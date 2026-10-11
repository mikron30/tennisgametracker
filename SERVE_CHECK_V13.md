# V13 rejects broad player-motion terminal recovery

The V12 focused audit confirms that the synthetic OUT at frame 4335 is now
suppressed.  The rally still remains on a false right-side track and eventually
ends with another false boundary-reversal OUT at frame 4425.

The earliest newly exposed cause is frame 4331.  After the tracker has been
stuck on the far-right artifact, `_resume_from_terminal_motion_candidate()`
accepts `(2983,733)` as a replacement with a displacement of about 850 px and a
local motion fraction of 0.795.  The event is tagged as
`terminal_motion_recovery`, but visually this is broad player/body motion rather
than compact ball motion.  Committing that recovery poisons the subsequent
motion history and allows the tracker to continue on right-side player/court
fragments.

V13 therefore adds a guard before a terminal-motion recovery mutates live state.
A large relocation (`>= max(300 px, 8% of image width)`) is rejected only when
its local motion is broad: motion fraction at least 0.65 anywhere, or at least
0.30 when the candidate lies in a tracked player/head/shoes/racket/body zone.
Large recoveries with compact local motion remain allowed.  There are no frame
or coordinate exceptions.

The focused tests cover both sides: a broad player-motion timeout recovery is
rejected without touching motion history, while the same large relocation with
compact local motion is still accepted and tagged as `terminal_motion_recovery`.

Run:

```powershell
python tracking_audit.py --court night --start-frame 4000 --max-frames 500 --output-dir tmp\serve_check_v13 --provider local --max-events 12 --endpoint-audit
```

Expected comparison with V12:

- preserve serve stance / tracking start around 4074-4077;
- preserve the service bounce at 4117;
- preserve receiver temporal recovery around 4133-4147;
- keep the false serve around 4375 absent;
- keep V9/V11/V12 OUT protections intact;
- at the frame-4331 timeout, reject the broad player-motion recovery instead of
  committing the ~850 px `(2983,733)` jump;
- inspect the resulting endpoint before any broader change.
