# V39 candidate status — 2026-09-18

V39 continues from committed V38 (`018795b`). It fixes the first protected
point, which V37 and V38 ended at frame 329 instead of the reviewed frame 322.

V38 had two ordering/evidence defects in its slow near-net turn guard:

- the check was nested too late in the net-classification path; and
- it recomputed the turn from adjacent motion vectors (51.6 degrees) instead
  of using the 129.1-degree direction change already accepted by
  `update_motion()`.

V39 evaluates the same narrow signature in the shared net path. It requires a
same-frame accepted direction change, both trajectory points within the 18px
calibrated net margin, previous speed at most 5px/frame, current speed at most
15px/frame, downward motion of at least 4px, and an accepted turn of at least
100 degrees. Recent ground and serve bounces remain excluded.

## Focused validation

The complete frames 0-349 source-video replay is in
`tmp/regression/v39b_first350` and `v39_first350.zip`.

- Serve start: 22
- Point end: 322
- Winner: P2
- Reason: `Ball hit the net`
- Category: `net_error`
- End position: `(2022, 513)`

The clean endpoint sheet visually matches the protected f322 net impact. The
runtime log records `[V39 SLOW NEAR-NET TURN]` with speeds 3.2/11.7px, accepted
angle 129.1 degrees, and signed net distances 9.4/-1.1px.

## Promotion gate

V39 remains an uncommitted candidate. The full frames 0-27920 regression
completed under `tmp/regression/v39_full_auto`, but the current V38-based
runtime failed the protected-point gate: only 4 of 28 protected starts matched
all endpoint fields, 5 matching starts changed, and 19 protected starts were
missing. The run itself exited successfully and processed all 27920 frames.

The V39 marker fired exactly once, at the intended f322 endpoint. It did not
cause the later mismatches; those expose accumulated drift already present in
the V38 line. V39 therefore fixes the focused point but cannot be promoted as
a fully validated runtime while its base fails the full ledger.

Exact full-run matches: starts 22, 1242, 2223 and 6813. The first subsequent
mismatch is start 696, ending at 749 with a direct service-box fault instead
of the protected f751 net-then-out fault.

Exact source/config hashes are stored in `candidate_source.json`.

The runtime diff is backed up at `tmp/v39_candidate.patch`. The user's modified
player-tracking configuration was not overwritten.
