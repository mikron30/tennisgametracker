# V15 preserves confirmed OUT endpoint provenance

The V14 focused audit confirms the OUT decision itself is now correct. At frame 4284 the descending right-side candidate is held as a pending OUT, and frame 4287 confirms it from a sharp bounded rebound. The point ends at frame 4287 with winner P1.

The remaining issue was reporting only. V14 already retained the original bounce position `(2922,729)`, but `_record_point_result()` later passed the endpoint through generic terminal player-overlap repair and the audit persisted `(2951,658)` instead. The point-end log and held visual also used the current rebound position rather than the original bounce source.

V15 keeps V14's scoring and OUT decision unchanged. It adds a same-frame, same-reason endpoint provenance lookup for confirmed pending OUTs. When that exact confirmation is being recorded, the original source bounce position is used consistently for:

- `[POINT_END]` logging;
- point-history / endpoint-audit `end_position`;
- the held endpoint visual marker.

The override is deliberately narrow: it is valid only on the frame where the pending OUT is confirmed and only for the matching `Ball bounced out of court (...)` reason. Other endpoint repairs and all other point-ending paths are unchanged.

Focused tests cover both the positive provenance case and rejection on a later frame or different end reason.

Run:

```powershell
python tracking_audit.py --court night --start-frame 4000 --max-frames 500 --output-dir tmp\serve_check_v15 --provider local --max-events 12 --endpoint-audit
```

Expected around frame 4287:

- `[OUT-BOUNCE PENDING CONFIRMED] source_f=4284 point=(2922, 729) ...`;
- `[POINT_END] ... pos=(2922, 729) ...`;
- `[POINT_END POSITION OVERRIDE] ... source=(2922, 729)`;
- `endpoint_audit.json` and the point-history CSV report `end_position=(2922,729)` while `end_frame` remains 4287 and winner remains P1.
