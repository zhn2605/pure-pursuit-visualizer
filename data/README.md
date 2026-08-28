# Competition data

Written at runtime by `src/server.py`. Both files are gitignored.

- `scoreboard.json` — canonical leaderboard. Rewritten atomically (temp file +
  `os.replace`) on every submission, so an interrupted write can never
  truncate it.
- `runs.jsonl` — append-only log of **every** attempt, including crashes.
  Written *before* the JSON rewrite.

If `scoreboard.json` is ever deleted or corrupted, the server automatically
rebuilds the board from `runs.jsonl` on the next start. To reset the
competition, delete both files.
