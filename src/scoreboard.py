"""Persistent competition scoreboard.

Two files are kept side by side:

* ``scoreboard.json`` -- the canonical list, rewritten atomically on each
  submission so a crash mid-write can never truncate it.
* ``runs.jsonl``      -- an append-only log of every attempt, written first.

If the JSON file is ever lost or corrupted it can be rebuilt from the log, so
a day's worth of club-fair submissions is never a single fsync away from gone.
"""

import json
import os
import tempfile
import threading
import time
import uuid

MAX_NAME_LENGTH = 24
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def clean_name(raw):
    """Normalise a player name, or raise ValueError with a readable reason."""
    if not isinstance(raw, str):
        raise ValueError("Name must be text.")
    # Drop control characters, collapse whitespace.
    name = "".join(ch for ch in raw if ch.isprintable()).strip()
    name = " ".join(name.split())
    if not name:
        raise ValueError("Please enter a name.")
    if len(name) > MAX_NAME_LENGTH:
        name = name[:MAX_NAME_LENGTH].strip()
    return name


class Scoreboard:
    def __init__(self, directory=DEFAULT_DIR):
        self.directory = directory
        self.json_path = os.path.join(directory, "scoreboard.json")
        self.log_path = os.path.join(directory, "runs.jsonl")
        self._lock = threading.Lock()
        os.makedirs(directory, exist_ok=True)
        self._entries = self._load()

    # -- storage ---------------------------------------------------------

    def _load(self):
        try:
            with open(self.json_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return []
        except (json.JSONDecodeError, OSError):
            # Corrupt or unreadable: fall back to the append-only log rather
            # than starting the day from an empty board.
            return self._load_from_log()
        if isinstance(data, dict):
            data = data.get("entries", [])
        return data if isinstance(data, list) else []

    def _load_from_log(self):
        entries = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        return entries

    def _write_atomic(self):
        payload = {"entries": self._entries, "updated": time.time()}
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.directory,
            prefix=".scoreboard-", suffix=".tmp", delete=False,
        )
        try:
            with handle:
                json.dump(payload, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, self.json_path)
        except BaseException:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
            raise

    def _append_log(self, entry):
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    # -- api -------------------------------------------------------------

    def record(self, name, result, equation, params):
        """Store one attempt. Failed runs are kept and ranked too.

        Repeat names are intentionally kept as separate submissions.
        """
        finished = result.status == "finished"
        entry = {
            "id": uuid.uuid4().hex[:12],
            "name": name,
            "status": result.status,
            # Time only means something for a completed run; an unfinished run
            # shows n/a rather than a misleading "time until it crashed".
            "time": result.time if finished else None,
            "progress": result.progress,
            # Exactly 100 for reaching the finish, otherwise how far it got.
            "completion": 100.0 if finished else round(result.progress * 100.0, 1),
            "equation": equation,
            "params": params,
            "timestamp": time.time(),
        }
        with self._lock:
            # Log first: if the atomic rewrite fails, the attempt still exists.
            self._append_log(entry)
            self._entries.append(entry)
            self._write_atomic()
        return entry

    @staticmethod
    def _completion(entry):
        """Completion percentage, tolerating entries written before the field."""
        if entry.get("completion") is not None:
            return float(entry["completion"])
        if entry.get("status") == "finished":
            return 100.0
        return round(float(entry.get("progress") or 0.0) * 100.0, 1)

    def _sort_key(self, entry):
        """Furthest first; among equal completion, fastest first.

        Unfinished runs have no time, so they sort after any finisher at the
        same completion rather than jumping ahead of one.
        """
        completion = self._completion(entry)
        seconds = entry.get("time")
        return (
            -completion,
            float("inf") if seconds is None else float(seconds),
            entry.get("timestamp", 0),
        )

    def ranked(self, limit=None):
        """Every attempt, ranked by completion then time, tagged with its rank."""
        with self._lock:
            entries = list(self._entries)
        entries.sort(key=self._sort_key)
        ranked = []
        for index, entry in enumerate(entries, start=1):
            row = dict(entry)
            row["rank"] = index
            row["completion"] = self._completion(entry)
            ranked.append(row)
        return ranked[:limit] if limit else ranked

    def stats(self):
        with self._lock:
            total = len(self._entries)
            finished = sum(1 for e in self._entries if e.get("status") == "finished")
            names = {e.get("name") for e in self._entries if e.get("name")}
        return {"attempts": total, "finishes": finished, "players": len(names)}

    def rank_of(self, entry_id):
        for row in self.ranked():
            if row["id"] == entry_id:
                return row["rank"]
        return None
