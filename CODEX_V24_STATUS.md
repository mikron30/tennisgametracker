# Codex V24 status — 2026-09-13

## Promotion status

V24 is a **failed candidate** and must not be treated as the runtime default.
The promoted/runtime baseline remains **V20** on `codex/shallow-serve-turn-candidate`.

Focused tests in `test_recovered_serve_scoring.py` and
`test_out_bounce_motion_consistency.py` passed when pytest used a
workspace-local `--basetemp`, but that was not sufficient for promotion.

## Full regression attempt

A full 0–27920 candidate run was started under:

```powershell
python tools/auto_tracking_cycle.py --output-dir tmp/regression/v24_full_auto `
  --start-frame 0 --max-frames 27920 --max-events 1
```

The run was stopped after frame 6466 because several protected points had
already changed. Partial history and logs were retained in the Codex workspace.

Important result: the focused `4074->4260` point became winner **P1** in the
full run because earlier false points changed score/game and court-side
ownership. The same point is winner **P2** when starting directly at frame
4000. Therefore the V24 candidate failed the promotion gate.

## Rollback

The complete V24 code/test diff was saved in the Codex workspace as:

`tmp/v24_candidate.patch`

The three modified tracked files were then restored byte-for-byte from `HEAD`
(V20). User-modified HSV/player configuration was intentionally left untouched.
The V24 patch is not committed to this branch because it failed full-regression
promotion.

The V24 full-run source SHA-256 hashes were saved in:

`tmp/regression/v24_full_auto/candidate_source.json`

## Next validation step

Codex started a short V20 replay under:

`tmp/regression/v20_first350`

The purpose is to determine whether the first protected mismatch — runtime
`22->176` versus reviewed `22->322` — already exists in V20. That result should
be established before designing another tracking fix.

After that, any new candidate must again be checked against the protected
starts, endpoints, winners, reasons and categories. A focused fix must not be
promoted if earlier false points alter later score/game state or court-side
ownership.

## Practical consequence

`git pull` showing no runtime-code update after V20 is expected and correct.
There is currently no validated V21–V24 runtime change to pull. The next code
change should be based on the V20 baseline plus the V20 first-350 diagnosis,
not on the failed V24 candidate.
