# Serve check V3 (experimental)

V2 did not pass video validation: the false endpoint shifted from frame 4292
to 4314, and walking generated another serve near 4375. Do not promote.

V3 derives the near player's baseline from bright court markings in the night
camera image, fitting a curve for lens distortion. The existing calibration
for other cameras is retained. Missing night geometry holds serve detection.
Both scanning and waiting now validate stance before collecting toss history.
Final motion logging is recomputed from the previous committed ball position
to the selected position, avoiding mixed candidate direction/displacement.

Validation: 54 focused tests passed. Full suite: 14 failed, 157 passed,
2 subtests passed; the same 14 failures existed before V3. Baseline estimates
at x=2250/2900 are consistent across supplied frames 4116, 4132 and 4292
(about y=1657/1640). This is not a full video test.

Remaining risks: person/racket false ball candidates and missed receiver
contact are not solved by these guards. Player tracking must be available.
Run the 4000–4499 audit first, confirm the actual serve/rally remains detected,
then follow REGRESSION_GUARDS.md including the full 0–27920 run before promotion.

PowerShell:

```powershell
python tracking_audit.py --court night --start-frame 4000 --max-frames 500 --output-dir tmp\serve_check_v3 --provider local --max-events 12 --endpoint-audit
Compress-Archive -Path "tmp\serve_check_v3\*" -DestinationPath "tmp\serve_check_v3.zip" -Force
```
