# Court 1 locked regression through point 27

This fixture preserves the reviewed continuous run from frame 33700 through
the first-serve fault at frame 46445. It contains 27 point-history rows and
ends at score `3:0 0:0`, with P2 due to serve a second serve.

Before a regression, restore the learned player profile so repeated runs start
from the same state:

```powershell
Copy-Item regression_baselines\locked_through_point27\player_tracking_start.json `
  player_tracking_hsv_config_court1_trial.json -Force
```

Compare a completed prefix with the locked history using:

```powershell
python validate_court1_history.py <point_history.csv> `
  --reference regression_baselines\locked_through_point27\events.csv
```

The next reviewed point is expected to end at frame 46799 with P2 winning on a
net error, six recorded hits, and score `3:0 0:15`. The far-player occlusion
repair activates only after three completed games, for a verified incoming
near-end return. This keeps the locked prefix outside the new code path.

`hashes.csv` records the source and configuration state that produced the
locked prefix. `current_after_point28_hashes.csv` records the implementation
that adds the reviewed point-28 repair.
