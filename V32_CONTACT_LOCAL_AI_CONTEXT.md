# V32 Contact Local-AI source context

## Lines 215-425
```python
000215:                 self._target.write(self._buffer)
000216:             self._buffer = ""
000217:         self._target.flush()
000218: 
000219: 
000220: class InteractiveBallAnalyzer:
000221:     def __init__(
000222:         self,
000223:         video_path: str,
000224:         start_frame: int = 0,
000225:         config_file: str = "hsv_config.json",
000226:         headless: bool = False,
000227:         disable_false_points: bool = False,
000228:         point_history_file: str = "point_history.csv",
000229:         write_point_history: bool = True,
000230:         start_server_side: Optional[str] = None,
000231:         start_score: Optional[Tuple[int, int, int, int]] = None,
000232:         enable_player_tracking: bool = True,
000233:         player_tracking_interval: int = 5,
000234:         enable_player_learning: bool = True,
000235:         enable_player_ball_protection: bool = True,
000236:         ball_dataset_dir: Optional[str] = None,
000237:         local_ai_model: Optional[str] = None,
000238:         local_ai_python: Optional[str] = None,
000239:         local_ai_recovery_dir: Optional[str] = None,
000240:         debug_local_ai_range: Optional[Tuple[int, int]] = None,
000241:         debug_local_ai_radius: float = 140.0,
000242:     ):
000243:         self.video_path = video_path
000244:         self.config_file = config_file
000245:         self.headless = headless
000246:         self.disable_false_points = disable_false_points
000247:         self.point_history_file = point_history_file
000248:         self.write_point_history = write_point_history
000249:         self.ball_dataset_dir = ball_dataset_dir
000250:         self.local_ai_model = local_ai_model
000251:         self.player_tracking_enabled = bool(enable_player_tracking) and os.environ.get(
000252:             "DISABLE_PLAYER_TRACKING", "0"
000253:         ) != "1"
000254:         self.player_learning_enabled = bool(enable_player_learning) and os.environ.get(
000255:             "DISABLE_PLAYER_LEARNING", "0"
000256:         ) != "1"
000257:         self.player_ball_protection_enabled = bool(enable_player_ball_protection) and os.environ.get(
000258:             "DISABLE_PLAYER_BALL_PROTECTION", "0"
000259:         ) != "1"
000260:         self.player_tracking_interval = max(1, int(player_tracking_interval))
000261:         profile_stem = os.path.splitext(os.path.basename(config_file or "hsv_config.json"))[0]
000262:         profile_dir = os.path.dirname(os.path.abspath(config_file or "hsv_config.json"))
000263:         self.player_tracking_profile_path = os.path.join(
000264:             profile_dir, f"player_tracking_{profile_stem}.json"
000265:         )
000266:         self.player_tracker = PlayerRacketTracker(
000267:             detection_interval=self.player_tracking_interval,
000268:             profile_path=self.player_tracking_profile_path,
000269:             learning_enabled=self.player_learning_enabled,
000270:         ) if self.player_tracking_enabled else None
000271:         if start_server_side not in (None, "near", "far"):
000272:             raise ValueError("start_server_side must be 'near', 'far', or None")
000273:         self.start_server_side = start_server_side
000274:         self._point_history_initialized = False
000275:         self._point_history_point_index = 0
000276:         self._point_history_current = None
000277:         self._point_audit_traces = {}
000278:         self._process_stop_reason = None
000279:         self.cap = cv2.VideoCapture(video_path)
000280:         self.start_frame = max(0, start_frame)
000281:         self.frame_count = self.start_frame
000282:         self.last_seen_frame = self.start_frame
000283:         self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
000284:         self.fps = self.cap.get(cv2.CAP_PROP_FPS)
000285:         self.ball_dataset_exporter = (
000286:             BallDatasetExporter(ball_dataset_dir, video_path)
000287:             if ball_dataset_dir else None
000288:         )
000289:         if self.ball_dataset_exporter is not None:
000290:             print(
000291:                 f"[BALL_DATASET] Exporting tracker-labelled source frames to "
000292:                 f"{self.ball_dataset_exporter.run_dir}"
000293:             )
000294:         self.local_ai_recovery = None
000295:         self._debug_local_ai_range = debug_local_ai_range
000296:         self._debug_local_ai_radius = max(20.0, float(debug_local_ai_radius))
000297:         self._force_local_ai_range = None
000298:         self._force_local_ai_radius = 450.0
000299:         self._force_local_ai_min_score = 0.985
000300:         self._force_local_ai_history = []
000301:         # Generic racket-contact AI ownership.  This is armed automatically only
000302:         # when the normal HSV path becomes suspicious inside a tracked player's
000303:         # contact corridor.  It reuses the proven forced-AI tracker, but keeps a
000304:         # separate history so the debug flag and automatic mode never interfere.
000305:         self._contact_local_ai_state = None
000306:         self._contact_local_ai_cooldown_until_frame = -1000000
000307:         self._contact_local_ai_radius = 450.0
000308:         self._contact_local_ai_min_score = 0.985
000309:         # Reference to the normal HSV/contact candidate that caused
000310:         # CONTACT_LOCAL_AI to arm. It exists only while Local AI ranks that
000311:         # same first frame and may break a near-tie in favour of independent
000312:         # HSV+AI agreement; it never steers later AI-history frames.
000313:         self._contact_local_ai_debug_normal_candidate = None
000314:         self._local_ai_frame_buffer = frame_buffer(12)
000315:         self._local_ai_recovery_count = 0
000316:         self._local_ai_all_body_rejections = 0
000317:         # A sharp near-player turn may need a few tightly cropped AI rankings
000318:         # before ordinary HSV tracking has a trustworthy outgoing vector
000319:         # again.  This state is bounded to the contact corridor and hands off
000320:         # as soon as normal tracking clears the player.
000321:         self._local_ai_tight_roi_follow = None
000322:         self._local_ai_tight_roi_accept_frame = -1000000
000323:         self._local_ai_tight_roi_attempt_frame = -1000000
000324:         self._local_ai_tight_roi_previous_gray = None
000325:         self._discard_provisional_serve_from_ai = False
000326:         # A provisional serve can briefly select a static, ball-coloured court
000327:         # artifact.  Keep that candidate separate from the live track until
000328:         # the following decoded frame can prove that it really persists.
000329:         # This is intentionally runtime-only: a coordinate that is false
000330:         # during a tossed-and-caught serve may be a legitimate ball location
000331:         # later in a rally.
000332:         self._pending_provisional_static_candidate = None
000333:         # Keep history reconciliation separate from the live tracking state.
000334:         # A verified caught toss can change the tennis interpretation of two
000335:         # subsequent serves, but changing ``current_serve_attempt`` while the
000336:         # HSV tracker is still in flight was shown to alter later candidate
000337:         # selection.  This record lets the CSV reflect the verified sequence
000338:         # after the normal tracker has completed each physical flight.
000339:         self._serve_history_reconciliation = None
000340:         self._local_ai_follow_until_frame = -1
000341:         # A recovery usually happens at a player/racket occlusion.  The ball
000342:         # needs more than a single four-frame sample to leave that region, but
000343:         # this handoff must remain bounded so local-AI scoring cannot replace
000344:         # normal tracking for an entire rally.
000345:         self._local_ai_handoff_deadline_frame = -1
000346:         if local_ai_model:
000347:             self.local_ai_recovery = LocalBallAIRecovery(
000348:                 local_ai_model,
000349:                 config_file,
000350:                 python_executable=local_ai_python,
000351:                 work_dir=local_ai_recovery_dir or "tmp/local_ai_recovery",
000352:             )
000353:             print(
000354:                 f"[LOCAL_AI] Buffered recovery enabled: model={self.local_ai_recovery.model_path} "
000355:                 f"lookback={self.local_ai_recovery.lookback_frames}f"
000356:             )
000357:         
000358:         # Ball analysis state
000359:         self.ball_center = None
000360:         self.ball_hsv = None
000361:         self.ball_size = None
000362:         self.tracking = False
000363:         self.ball_stopped = False
000364:         self.stuck_frame_count = 0
000365:         # After a player-occlusion re-acquisition, keep rejecting static HSV
000366:         # blobs for a short window.  The player/racket can contain a large
000367:         # ball-coloured patch, and accepting it immediately after reacquisition
000368:         # causes a false bounce/end while the real ball is still in flight.
000369:         self._player_reacq_protect_until_frame = -1
000370:         # A failed full-frame motion reacquisition should not be followed by
000371:         # another expensive full-frame HSV contour scan on every frame.  The
000372:         # motion pass already examines the whole image; until the next stuck
000373:         # milestone, keep the normal local search around the last valid ball.
000374:         self._player_reacq_motion_failed_until_frame = -1
000375:         self.ball_velocity_history = []
000376:         self.initial_ball_position = None
000377:         self.last_seen_frame = None
000378:         self.edge_wait = False
000379:         self.last_delta = None
000380:         self.primary_hsv_lower = None
000381:         self.primary_hsv_upper = None
000382:         self.alt_hsv_lower = None
000383:         self.alt_hsv_upper = None
000384:         self.hsv_regular = None
000385:         self.hsv_behind_net = None
000386:         
000387:         # HSV filter (will be set based on clicked ball)
000388:         self.hsv_lower = None
000389:         self.hsv_upper = None
000390:         
000391:         # HSV values table for analysis
000392:         self.hsv_table = []
000393:         
000394:         self.motion_history = []
000395:         self.focus_loss_frame = None
000396:         self.focus_loss_active = False
000397:         self._focus_loss_guard_until_frame = -1000000
000398:         self.alt_focus_hsv_lower = None
000399:         self.alt_focus_hsv_upper = None
000400:         self.last_motion = None
000401:         self.pre_focus_hsv_regular = None
000402:         self.pre_focus_hsv_behind_net = None
000403:         self.using_alt_hsv = False
000404:         self.using_alt2_hsv = False
000405:         self.alt2_hsv_lower = None
000406:         self.alt2_hsv_upper = None
000407:         self.using_alt3_hsv = False
000408:         self.alt3_hsv_lower = None
000409:         self.alt3_hsv_upper = None
000410:         self.using_alt4_hsv = False
000411:         self.alt4_hsv_lower = None
000412:         self.alt4_hsv_upper = None
000413:         self.using_alt6_hsv = False
000414:         self.alt5_hsv_lower = None
000415:         self.alt5_hsv_upper = None
000416:         self.alt6_hsv_lower = None
000417:         self.alt6_hsv_upper = None
000418:         self.s30_hsv_lower = None
000419:         self.s30_hsv_upper = None
000420:         self.h10_hsv_lower = None
000421:         self.h10_hsv_upper = None
000422:         self.alts20_hsv_lower = None
000423:         self.alts20_hsv_upper = None
000424:         self.alts9_11_hsv_lower = None
000425:         self.alts9_11_hsv_upper = None
```

## Lines 1179-2247
```python
001179:                 f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
001180:                 f"candidate moved {displacement:.1f}px; no ignore"
001181:             )
001182:         return False
001183: 
001184:     def _try_active_tight_local_ai_hold(
001185:             self, frame, tracked_position, previous_gray=None):
001186:         """Advance an active contact ROI when normal tracking only holds.
001187: 
001188:         Player-occlusion branches can return the previous marker without
001189:         reaching any candidate-commit hook.  During the bounded tight-ROI
001190:         contact lock, rank that small area directly instead of waiting five
001191:         frames for a broad full-frame reacquisition.
001192:         """
001193:         follow = getattr(self, '_local_ai_tight_roi_follow', None)
001194:         if (
001195:                 not isinstance(follow, dict) or frame is None or
001196:                 int(self.frame_count) > int(follow.get('deadline', -1)) or
001197:                 int(self.frame_count) <= int(follow.get('last_frame', -1)) or
001198:                 int(getattr(self, 'stuck_frame_count', 0)) < 3):
001199:             return None
001200: 
001201:         held_pos = tracked_position or self.ball_center or follow.get('last_pos')
001202:         if held_pos is None:
001203:             return None
001204:         held_pos = (int(held_pos[0]), int(held_pos[1]))
001205:         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
001206:         held_motion = self._candidate_motion_metrics(
001207:             frame_gray, held_pos[0], held_pos[1], previous_gray=previous_gray
001208:         )
001209:         held_mean = (
001210:             float(held_motion.get('mean', 0.0) or 0.0)
001211:             if held_motion else 0.0
001212:         )
001213:         held_max = (
001214:             float(held_motion.get('max', 0.0) or 0.0)
001215:             if held_motion else 0.0
001216:         )
001217:         height, width = frame.shape[:2]
001218:         px = max(0, min(width - 1, held_pos[0]))
001219:         py = max(0, min(height - 1, held_pos[1]))
001220:         proposal = {
001221:             'pos': held_pos,
001222:             'area': float(getattr(self, 'ball_size', 0.0) or 0.0),
001223:             'hsv': cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[py, px],
001224:             'motion_mean': held_mean,
001225:             'motion_max': held_max,
001226:             'source': 'tight_roi_held_marker',
001227:             'recovery_label': 'LOCAL AI TIGHT ROI HOLD',
001228:             'tight_roi_force_rank': True,
001229:         }
001230:         preferred = self._prefer_night_tight_local_ai_candidate(
001231:             frame, frame_gray, proposal, previous_gray=previous_gray
001232:         )
001233:         if preferred.get('source') != 'local_ai_tight_roi':
001234:             return None
001235:         return self._commit_night_visible_ball_recovery(preferred, frame)
001236: 
001237:     def _normal_ball_evidence_is_strong(self, frame, previous_position, tracked_position, snapshot):
001238:         """A visible, moving, local HSV observation needs no model inference."""
001239:         if frame is None or previous_position is None or tracked_position is None:
001240:             return False
001241:         if self._player_point_zone(tracked_position) is not None:
001242:             return False
001243:         step = math.dist(previous_position, tracked_position)
001244:         speeds = (snapshot or {}).get('ball_velocity_history', [])[-5:]
001245:         prior_speed = float(np.median(speeds)) if speeds else 0.0
001246:         if not 3.0 <= step <= max(120.0, min(400.0, prior_speed * 4.0 + 40.0)):
001247:             return False
001248:         if not 8.0 <= float(getattr(self, 'ball_size', 0.0) or 0.0) <= 700.0:
001249:             return False
001250:         previous_gray = getattr(self, '_terminal_previous_gray', None)
001251:         if previous_gray is None:
001252:             previous_gray = (snapshot or {}).get('_prev_frame_gray')
001253:         if previous_gray is None or previous_gray.shape[:2] != frame.shape[:2]:
001254:             return False
001255:         x, y = map(int, tracked_position)
001256:         left, top = max(0, x-8), max(0, y-8)
001257:         right, bottom = min(frame.shape[1], x+9), min(frame.shape[0], y+9)
001258:         if right <= left or bottom <= top:
001259:             return False
001260:         patch = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
001261:         arrival = patch.astype(np.int16) - previous_gray[top:bottom, left:right].astype(np.int16)
001262:         return float(np.mean(np.maximum(arrival, 0))) >= 10.0 and float(np.max(arrival)) >= 60.0
001263: 
001264:     def _automatic_local_ai_scoring_ready(self):
001265:         # Edge waits already search for a visible return. A held exit anchor
001266:         # is expected absence, not a contact occlusion needing model search.
001267:         # Leave endpoint aging and normal re-entry detection in control.
001268:         if self._top_return_wait_active() or self._back_return_wait_active():
001269:             self._contact_local_ai_state = None
001270:             return False
001271:         recovery = getattr(self, 'local_ai_recovery', None)
001272:         ready = getattr(recovery, 'scoring_ready', None)
001273:         return not callable(ready) or bool(ready(self.frame_count))
001274: 
001275:     def _normal_fallback_after_ai_miss(self, previous_position, proposed_position, before, after):
001276:         """Keep normal loss aging; never repeatedly reset an occlusion clock."""
001277:         if proposed_position is None or tuple(proposed_position) == tuple(previous_position):
001278:             self._restore_tracking_state_for_provisional_guard(after)
001279:             return proposed_position
001280:         # A suspect new player/background anchor remains rejected, but losing
001281:         # this frame must still advance reacquisition and endpoint timers.
001282:         self._restore_tracking_state_for_provisional_guard(before)
001283:         self.stuck_frame_count = max(int(before.get('stuck_frame_count', 0)) + 1,
001284:                                      int(after.get('stuck_frame_count', 0)))
001285:         return previous_position
001286: 
001287:     def _contact_local_ai_trigger(self, previous_position, tracked_position, snapshot, frame=None):
001288:         """Return a generic racket-contact takeover reason for suspicious HSV.
001289: 
001290:         The trigger is deliberately player-relative rather than serve-relative:
001291:         it applies to serves, forehands, backhands and either player.  Merely
001292:         being near a player is not enough.  We require either an actual player
001293:         zone selection or a sharp trajectory/size discontinuity inside the
001294:         padded racket-contact corridor.
001295:         """
001296:         current = int(self.frame_count)
001297:         if self._normal_ball_evidence_is_strong(frame, previous_position, tracked_position, snapshot):
001298:             self._contact_local_ai_state = None
001299:             return None
001300:         if not self._automatic_local_ai_scoring_ready():
001301:             return None
001302:         if (
001303:                 getattr(self, 'local_ai_recovery', None) is None or
001304:                 previous_position is None or
001305:                 current <= int(getattr(
001306:                     self, '_contact_local_ai_cooldown_until_frame', -1000000
001307:                 ))):
001308:             return None
001309: 
001310:         previous = (int(previous_position[0]), int(previous_position[1]))
001311:         watch_until = int(getattr(
001312:             self, '_contact_local_ai_watch_until_frame', -1000000
001313:         ))
001314:         if tracked_position is None:
001315:             if current <= watch_until:
001316:                 return 'contact-loss-after-player-proximity'
001317:             return None
001318:         tracked = (int(tracked_position[0]), int(tracked_position[1]))
001319:         prior_motion = dict((snapshot or {}).get('last_motion') or {})
001320:         prior_dx = float(prior_motion.get('dx', 0.0) or 0.0)
001321:         prior_dy = float(prior_motion.get('dy', 0.0) or 0.0)
001322:         prior_speed = float(
001323:             prior_motion.get('distance', math.hypot(prior_dx, prior_dy)) or 0.0
001324:         )
001325:         proposed_dx = float(tracked[0] - previous[0])
001326:         proposed_dy = float(tracked[1] - previous[1])
001327:         proposed_speed = math.hypot(proposed_dx, proposed_dy)
001328: 
001329:         predicted = (
001330:             int(round(float(previous[0]) + prior_dx)),
001331:             int(round(float(previous[1]) + prior_dy)),
001332:         )
001333:         prediction_error = math.hypot(
001334:             float(tracked[0] - predicted[0]), float(tracked[1] - predicted[1])
001335:         )
001336: 
001337:         angle_delta = 0.0
001338:         if prior_speed >= 4.0 and proposed_speed >= 4.0:
001339:             prior_angle = math.degrees(math.atan2(prior_dy, prior_dx))
001340:             proposed_angle = math.degrees(math.atan2(proposed_dy, proposed_dx))
001341:             raw_delta = abs(proposed_angle - prior_angle) % 360.0
001342:             angle_delta = min(raw_delta, 360.0 - raw_delta)
001343: 
001344:         zone = self._player_point_zone(tracked)
001345:         contact_near = any(
001346:             self._point_in_player_contact_corridor(point)
001347:             for point in (previous, tracked, predicted)
001348:         )
001349:         watch_active = current <= watch_until
001350:         if zone is not None or contact_near:
001351:             self._contact_local_ai_watch_until_frame = max(watch_until, current + 8)
001352:             watch_active = True
001353: 
001354:         # Generic trajectory-break recovery. Do not invoke AI merely because
001355:         # the ball changed direction: real tennis hits and bounces do that all
001356:         # the time. Escalate only when the normal detector has just accepted a
001357:         # predicted-path background hotspot, local image motion is extremely
001358:         # weak, the turn is large, and there is no active serve/rally/bounce
001359:         # contact grace explaining the change.
001360:         trajectory_hotspot = getattr(self, '_trajectory_local_ai_hotspot', None)
001361:         if (
001362:                 isinstance(trajectory_hotspot, dict) and
001363:                 int(trajectory_hotspot.get('frame', -1)) == current):
001364:             hotspot_motion_mean = float(trajectory_hotspot.get('motion_mean', 999.0) or 0.0)
001365:             hotspot_motion_max = float(trajectory_hotspot.get('motion_max', 999.0) or 0.0)
001366: 
001367:             # A predicted-path candidate that is nearly static in the image and
001368:             # simultaneously creates a large trajectory break is suspicious on
001369:             # its own. Do not suppress this because a player/contact watch is
001370:             # active: that is exactly when Local AI can protect the trajectory.
001371:             strong_hotspot_break = (
001372:                 prior_speed >= 14.0 and proposed_speed >= 14.0 and
001373:                 angle_delta >= 50.0 and prediction_error >= 12.0 and
001374:                 hotspot_motion_mean <= 3.0 and hotspot_motion_max <= 12.0
001375:             )
001376:             print(
001377:                 f'[TRAJECTORY_LOCAL_AI_GATE] f{current}: '
001378:                 f'previous={previous} normal={tracked} predicted={predicted} '
001379:                 f'prior_speed={prior_speed:.1f} proposed_speed={proposed_speed:.1f} '
001380:                 f'angle={angle_delta:.1f}deg pred_error={prediction_error:.1f}px '
001381:                 f'motion={hotspot_motion_mean:.1f}/{hotspot_motion_max:.1f} '
001382:                 f'zone={zone} contact_near={contact_near} watch_active={watch_active} '
001383:                 f'pass={strong_hotspot_break}'
001384:             )
001385:             if strong_hotspot_break:
001386:                 reason = (
001387:                     f'trajectory-hotspot-turn:{angle_delta:.0f}deg/'
001388:                     f'pred={prediction_error:.0f}px/'
001389:                     f'motion={hotspot_motion_mean:.1f}/{hotspot_motion_max:.1f}'
001390:                 )
001391:                 print(
001392:                     f'[TRAJECTORY_LOCAL_AI_TRIGGER] f{current}: '
001393:                     f'previous={previous} normal={tracked} predicted={predicted} '
001394:                     f'angle={angle_delta:.1f}deg pred_error={prediction_error:.1f}px '
001395:                     f'motion={hotspot_motion_mean:.1f}/{hotspot_motion_max:.1f}'
001396:                 )
001397:                 return reason
001398: 
001399:         if zone is None and not contact_near and not watch_active:
001400:             return None
001401: 
001402:         recent_speeds = [
001403:             float(value) for value in (snapshot or {}).get('ball_velocity_history', [])[-5:]
001404:             if value is not None
001405:         ]
001406:         recent_median = float(np.median(recent_speeds)) if recent_speeds else prior_speed
001407:         jump_floor = max(70.0, recent_median * 1.8, prior_speed * 1.55)
001408: 
001409:         previous_size = float((snapshot or {}).get('ball_size', 0.0) or 0.0)
001410:         proposed_size = float(getattr(self, 'ball_size', 0.0) or 0.0)
001411:         size_ratio = (
001412:             max(previous_size, proposed_size) / max(1.0, min(previous_size, proposed_size))
001413:             if previous_size > 0.0 and proposed_size > 0.0 else 1.0
001414:         )
001415: 
001416:         player_zones = (
001417:             'player_head_hat', 'player_body', 'player_shoes', 'racket_fragment'
001418:         )
001419:         if (
001420:                 zone in player_zones and max(prior_speed, proposed_speed) >= 12.0 and
001421:                 prediction_error >= 30.0 and
001422:                 (angle_delta >= 30.0 or size_ratio >= 2.0)):
001423:             return (
001424:                 f'contact-player-artifact:{zone}/step={proposed_speed:.0f}px/'
001425:                 f'angle={angle_delta:.0f}deg/pred={prediction_error:.0f}px'
001426:             )
001427: 
001428:         contact_jump_candidate = (
001429:             (contact_near or watch_active) and proposed_speed >= jump_floor and
001430:             (angle_delta >= 35.0 or prediction_error >= 50.0)
001431:         )
001432: 
001433:         # V17: Contact Local-AI is an assist, not a permanent veto.
001434:         #
001435:         # During the short serve-contact/pre-bounce window, the normal HSV
001436:         # tracker can already see the outgoing ball while the local model still
001437:         # misses a blurred/contact-adjacent patch. The process loop deliberately
001438:         # rolls that normal candidate back when Contact Local-AI misses. Keep
001439:         # that protection for one-off racket/body fragments, but remember the
001440:         # rejected normal proposals. Three consecutive netward proposals with
001441:         # a coherent direction prove an independent causal ball path and let the
001442:         # normal tracker keep the third frame.
001443:         serve_contact_path_window = (
001444:             contact_jump_candidate and
001445:             tracked_position is not None and
001446:             bool(getattr(self, '_awaiting_serve_bounce', False)) and
001447:             int(getattr(self, '_serve_contact_grace_frames', 0) or 0) > 0 and
001448:             int(getattr(self, 'ground_bounce_count', 0) or 0) == 0 and
001449:             int(getattr(self, 'serve_direction_dy', 0) or 0) != 0
001450:         )
001451:         contact_normal_path = list(
001452:             getattr(self, '_contact_normal_fallback_path', []) or []
001453:         )
001454: 
001455:         if serve_contact_path_window:
001456:             current_sample = {
001457:                 'frame': int(self.frame_count),
001458:                 'pos': tuple(tracked_position),
001459:                 'area': float(proposed_size),
001460:             }
001461:             if (
001462:                     contact_normal_path and
001463:                     int(current_sample['frame']) !=
001464:                     int(contact_normal_path[-1].get('frame', -1000000)) + 1):
001465:                 contact_normal_path = []
001466: 
001467:             if contact_normal_path:
001468:                 previous_sample = contact_normal_path[-1]
001469:                 previous_pos = tuple(previous_sample['pos'])
001470:                 step_dx = float(tracked_position[0] - previous_pos[0])
001471:                 step_dy = float(tracked_position[1] - previous_pos[1])
001472:                 step_distance = math.hypot(step_dx, step_dy)
001473:                 serve_dy = int(getattr(self, 'serve_direction_dy', 0) or 0)
001474:                 netward_progress = -step_dy if serve_dy < 0 else step_dy
001475:                 if not (
001476:                         20.0 <= step_distance <= 220.0 and
001477:                         netward_progress >= 15.0):
001478:                     contact_normal_path = []
001479: 
001480:             contact_normal_path.append(current_sample)
001481:             contact_normal_path = contact_normal_path[-3:]
001482:             self._contact_normal_fallback_path = contact_normal_path
001483: 
001484:             if len(contact_normal_path) == 3:
001485:                 p0 = tuple(contact_normal_path[0]['pos'])
001486:                 p1 = tuple(contact_normal_path[1]['pos'])
001487:                 p2 = tuple(contact_normal_path[2]['pos'])
001488:                 v1 = (
001489:                     float(p1[0] - p0[0]),
001490:                     float(p1[1] - p0[1]),
001491:                 )
001492:                 v2 = (
001493:                     float(p2[0] - p1[0]),
001494:                     float(p2[1] - p1[1]),
001495:                 )
001496:                 len1 = math.hypot(v1[0], v1[1])
001497:                 len2 = math.hypot(v2[0], v2[1])
001498:                 cosine = (
001499:                     (v1[0] * v2[0] + v1[1] * v2[1]) /
001500:                     max(1.0, len1 * len2)
001501:                 )
001502:                 serve_dy = int(getattr(self, 'serve_direction_dy', 0) or 0)
001503:                 total_netward = (
001504:                     float(p0[1] - p2[1])
001505:                     if serve_dy < 0 else
001506:                     float(p2[1] - p0[1])
001507:                 )
001508:                 coherent_path = (
001509:                     20.0 <= len1 <= 220.0 and
001510:                     20.0 <= len2 <= 220.0 and
001511:                     cosine >= 0.90 and
001512:                     total_netward >= 100.0
001513:                 )
001514:                 if coherent_path:
001515:                     self._contact_local_ai_state = None
001516:                     self._contact_local_ai_cooldown_until_frame = max(
001517:                         int(getattr(
001518:                             self,
001519:                             '_contact_local_ai_cooldown_until_frame',
001520:                             -1000000,
001521:                         )),
001522:                         int(self.frame_count) + 2,
001523:                     )
001524:                     self._contact_normal_fallback_path = []
001525:                     print(
001526:                         f"[CONTACT_NORMAL_PATH_RELEASE] f{self.frame_count}: "
001527:                         f"accepted normal HSV={tuple(tracked_position)} after "
001528:                         f"3 coherent netward proposals "
001529:                         f"path={[entry['pos'] for entry in contact_normal_path]} "
001530:                         f"cos={cosine:.3f} net={total_netward:.1f}px"
001531:                     )
001532:                     return None
001533:         else:
001534:             self._contact_normal_fallback_path = []
001535: 
001536:         if contact_jump_candidate:
001537:             return (
001538:                 f'contact-jump:{proposed_speed:.0f}px>={jump_floor:.0f}/'
001539:                 f'angle={angle_delta:.0f}deg/pred={prediction_error:.0f}px'
001540:             )
001541: 
001542:         if (
001543:                 (contact_near or watch_active) and prior_speed >= 18.0 and proposed_speed >= 18.0 and
001544:                 angle_delta >= 70.0 and prediction_error >= 30.0):
001545:             return (
001546:                 f'contact-turn:{angle_delta:.0f}deg/'
001547:                 f'pred={prediction_error:.0f}px'
001548:             )
001549: 
001550:         if (
001551:                 (contact_near or watch_active) and prior_speed >= 18.0 and
001552:                 proposed_speed <= max(6.0, prior_speed * 0.35) and
001553:                 prediction_error >= 18.0):
001554:             return (
001555:                 f'contact-stall:{prior_speed:.0f}->{proposed_speed:.0f}px/'
001556:                 f'pred={prediction_error:.0f}px'
001557:             )
001558: 
001559:         if (
001560:                 (contact_near or watch_active) and proposed_speed >= 35.0 and size_ratio >= 2.6 and
001561:                 prediction_error >= 35.0):
001562:             return (
001563:                 f'contact-size-jump:{size_ratio:.1f}x/'
001564:                 f'pred={prediction_error:.0f}px'
001565:             )
001566:         return None
001567: 
001568:     def _strong_visible_recovery_releases_contact_ai(
001569:             self, previous_position, tracked_position, snapshot):
001570:         if (getattr(self, '_temporal_top_reentry_frame', -1) == self.frame_count and
001571:                 getattr(self, '_temporal_top_reentry_pos', None) == tracked_position):
001572:             # Three current-image samples independently establish this flight.
001573:             # The off-screen anchor's size and velocity cannot invalidate it.
001574:             self._contact_local_ai_state = None
001575:             self._contact_normal_fallback_path = []
001576:             return True
001577:         """Release a stale contact owner after independent moving-ball evidence.
001578: 
001579:         Contact Local-AI normally protects the tracker from racket and player
001580:         fragments. If that owner has missed for several frames while holding
001581:         an oversized lower-court anchor, however, a compact high-motion
001582:         visible-ball recovery far up court is stronger evidence than the stale
001583:         anchor. Keep this gate deliberately strict so a static far-court ball
001584:         or hotspot cannot release contact ownership.
001585:         """
001586:         state = getattr(self, '_contact_local_ai_state', None)
001587:         reacq_pos = getattr(self, '_last_motion_reacq_pos', None)
001588:         if (
001589:                 not isinstance(state, dict) or
001590:                 previous_position is None or tracked_position is None or
001591:                 int(state.get('miss_count', 0) or 0) < 8 or
001592:                 int(getattr(self, '_last_motion_reacq_frame', -1000000)) !=
001593:                 int(self.frame_count) or
001594:                 reacq_pos is None or tuple(reacq_pos) !=
001595:                 tuple(tracked_position)):
001596:             return False
001597: 
001598:         previous_size = float((snapshot or {}).get('ball_size', 0.0) or 0.0)
001599:         previous_stuck = int((snapshot or {}).get('stuck_frame_count', 0) or 0)
001600:         recovered_size = float(getattr(self, 'ball_size', 0.0) or 0.0)
001601:         motion_mean = float(
001602:             getattr(self, '_last_tracked_candidate_motion_mean', 0.0) or 0.0
001603:         )
001604:         motion_max = float(
001605:             getattr(self, '_last_tracked_candidate_motion_max', 0.0) or 0.0
001606:         )
001607:         dx = float(tracked_position[0]) - float(previous_position[0])
001608:         dy = float(tracked_position[1]) - float(previous_position[1])
001609:         displacement = math.hypot(dx, dy)
001610:         player_zone = self._player_point_zone(tracked_position)
001611: 
001612:         strong_recovery = (
001613:             previous_stuck >= 3 and
001614:             previous_size >= 600.0 and
001615:             20.0 <= recovered_size <= 450.0 and
001616:             recovered_size <= previous_size * 0.55 and
001617:             motion_mean >= 40.0 and motion_max >= 100.0 and
001618:             displacement >= 500.0 and dy <= -300.0 and
001619:             player_zone is None
001620:         )
001621:         if not strong_recovery:
001622:             return False
001623: 
001624:         self._contact_local_ai_state = None
001625:         self._contact_local_ai_cooldown_until_frame = max(
001626:             int(getattr(
001627:                 self, '_contact_local_ai_cooldown_until_frame', -1000000
001628:             )),
001629:             int(self.frame_count) + 3,
001630:         )
001631:         self._contact_normal_fallback_path = []
001632:         self._contact_visible_recovery_exit = {
001633:             'release_frame': int(self.frame_count),
001634:             'origin': tuple(tracked_position),
001635:             'last_pos': tuple(tracked_position),
001636:             'last_move_frame': int(self.frame_count),
001637:             'coherent_up_steps': 0,
001638:             'outside_far_baseline': False,
001639:         }
001640:         near_player = self._player_index_on_court_end('near')
001641:         recent_contact_player = getattr(
001642:             self, '_last_racket_contact_player', None
001643:         )
001644:         recent_contact_age = int(self.frame_count) - int(getattr(
001645:             self, '_last_racket_contact_frame', -1000000
001646:         ))
001647:         if (
001648:                 near_player in (0, 1) and
001649:                 not (
001650:                     recent_contact_player == near_player and
001651:                     0 <= recent_contact_age <= 12
001652:                 )):
001653:             self._record_racket_contact(
001654:                 tuple(previous_position),
001655:                 label='motion-confirmed lower contact launch',
001656:                 player_idx_override=near_player,
001657:             )
001658:         print(
001659:             f"[CONTACT_VISIBLE_RECOVERY_RELEASE] f{self.frame_count}: "
001660:             f"keeping={tuple(tracked_position)} after "
001661:             f"{int(state.get('miss_count', 0) or 0)} AI misses; "
001662:             f"anchor={tuple(previous_position)} sizes={previous_size:.1f}->"
001663:             f"{recovered_size:.1f} motion={motion_mean:.1f}/{motion_max:.1f} "
001664:             f"step={displacement:.1f}px dy={dy:.1f}"
001665:         )
001666:         return True
001667: 
001668:     def _contact_visible_recovery_far_exit_end_candidate(
001669:             self, ball_position, frame):
001670:         """Confirm an outbound recovered ball that never returns from the top.
001671: 
001672:         The generic top-return waiter is intentionally generous because high
001673:         in-play lobs can leave and re-enter the image. A ball independently
001674:         recovered after lower-player contact is an OUT only after it follows a
001675:         multi-frame upward path across the far baseline and then remains absent
001676:         for a bounded grace period. Any downward re-entry cancels this state.
001677:         """
001678:         state = getattr(self, '_contact_visible_recovery_exit', None)
001679:         if not isinstance(state, dict) or ball_position is None or frame is None:
001680:             return False, None
001681: 
001682:         age = int(self.frame_count) - int(state.get('release_frame', self.frame_count))
001683:         if age < 0 or age > 45:
001684:             self._contact_visible_recovery_exit = None
001685:             return False, None
001686: 
001687:         current = tuple(ball_position)
001688:         previous = tuple(state.get('last_pos', current))
001689:         if current != previous:
001690:             step_dx = float(current[0] - previous[0])
001691:             step_dy = float(current[1] - previous[1])
001692:             step = math.hypot(step_dx, step_dy)
001693:             if step_dy >= 18.0:
001694:                 self._contact_visible_recovery_exit = None
001695:                 print(
001696:                     f"[CONTACT_VISIBLE_RECOVERY RETURN] f{self.frame_count}: "
001697:                     f"downward re-entry {previous}->{current}; cancelling far exit"
001698:                 )
001699:                 return False, None
001700:             if step_dy <= -8.0 and 8.0 <= step <= 240.0:
001701:                 state['coherent_up_steps'] = int(
001702:                     state.get('coherent_up_steps', 0) or 0
001703:                 ) + 1
001704:             else:
001705:                 state['coherent_up_steps'] = 0
001706:             state['last_pos'] = current
001707:             state['last_move_frame'] = int(self.frame_count)
001708: 
001709:         outside_far, far_y = self._point_outside_top_singles_baseline(
001710:             current, frame
001711:         )
001712:         state['outside_far_baseline'] = bool(
001713:             state.get('outside_far_baseline', False) or outside_far
001714:         )
001715:         self._contact_visible_recovery_exit = state
001716: 
001717:         origin = tuple(state.get('origin', current))
001718:         upward_distance = float(origin[1] - current[1])
001719:         absent_frames = int(self.frame_count) - int(
001720:             state.get('last_move_frame', self.frame_count)
001721:         )
001722:         frame_height = int(frame.shape[0])
001723:         confirmed = (
001724:             int(state.get('coherent_up_steps', 0) or 0) >= 3 and
001725:             bool(state.get('outside_far_baseline', False)) and
001726:             upward_distance >= 200.0 and
001727:             current[1] <= max(90, int(frame_height * 0.045)) and
001728:             absent_frames >= 14
001729:         )
001730:         if not confirmed:
001731:             return False, None
001732: 
001733:         self._contact_visible_recovery_exit = None
001734:         self._contact_visible_recovery_confirmed_end_frame = int(self.frame_count)
001735:         self._contact_visible_recovery_confirmed_end_position = tuple(current)
001736:         print(
001737:             f"[CONTACT_VISIBLE_RECOVERY FAR EXIT] f{self.frame_count}: "
001738:             f"origin={origin} edge={current} far_baseline_y={far_y:.1f} "
001739:             f"up_steps={state.get('coherent_up_steps')} "
001740:             f"absent={absent_frames}f"
001741:         )
001742:         return True, "Ball bounce outside singles court (far baseline)"
001743: 
001744:     def _run_contact_local_ai_frame(self, frame, previous_position, arm_reason=None):
001745:         """Attempt bounded recovery after a suspect normal observation.
001746: 
001747:         The implementation intentionally reuses the already validated
001748:         --force-local-ai-range tracker: player/head/shoe/racket candidates are
001749:         excluded before scoring, the AI-selected point becomes the next anchor,
001750:         and trajectory prediction starts after two accepted AI positions.
001751:         """
001752:         recovery = getattr(self, 'local_ai_recovery', None)
001753:         if recovery is None or frame is None or previous_position is None:
001754:             return None
001755:         if not self._automatic_local_ai_scoring_ready():
001756:             return None
001757: 
001758:         current = int(self.frame_count)
001759:         state = getattr(self, '_contact_local_ai_state', None)
001760:         if arm_reason is not None and not isinstance(state, dict):
001761:             # Seed CONTACT_LOCAL_AI with the trusted pre-contact anchor.  On the
001762:             # contact frame this is only one history point, so no trajectory is
001763:             # imposed across the racket direction change.  After Local AI accepts
001764:             # the first outgoing point, the next frame immediately has two points
001765:             # and can predict the outgoing ball trajectory.
001766:             seed_history = []
001767:             if previous_position is not None:
001768:                 seed_history.append({
001769:                     'frame': current - 1,
001770:                     'pos': tuple(previous_position),
001771:                     'score': 1.0,
001772:                     'contact_seed': True,
001773:                 })
001774:             state = {
001775:                 'start_frame': current,
001776:                 'deadline': current + 22,
001777:                 'history': seed_history,
001778:                 'outside_count': 0,
001779:                 'miss_count': 0,
001780:                 'reason': str(arm_reason),
001781:             }
001782:             self._contact_local_ai_state = state
001783:             if seed_history:
001784:                 print(
001785:                     f"[CONTACT_LOCAL_AI_SEED] f{current}: "
001786:                     f"seed_frame={current - 1} position={tuple(previous_position)}"
001787:                 )
001788:             # Do not let an older tight-ROI experiment compete with contact
001789:             # ownership.  The automatic owner is the stronger, player-excluding
001790:             # path and will hand control back explicitly.
001791:             self._local_ai_tight_roi_follow = None
001792:             print(
001793:                 f"[CONTACT_LOCAL_AI_ARM] f{current}: anchor={tuple(previous_position)} "
001794:                 f"reason={arm_reason} radius={self._contact_local_ai_radius:.0f}px"
001795:             )
001796: 
001797:         state = getattr(self, '_contact_local_ai_state', None)
001798:         if not isinstance(state, dict):
001799:             return None
001800:         if current > int(state.get('deadline', -1)):
001801:             print(
001802:                 f"[CONTACT_LOCAL_AI_EXPIRE] f{current}: "
001803:                 f"started=f{state.get('start_frame')} reason={state.get('reason')}"
001804:             )
001805:             self._contact_local_ai_state = None
001806:             self._contact_local_ai_cooldown_until_frame = current + 3
001807:             return None
001808: 
001809:         # Reuse the proven forced-AI implementation without exposing its state
001810:         # to the user's manual debug range.
001811:         # HSV and buffered recovery may have advanced the committed ball while
001812:         # this owner missed. Never extrapolate from its older private anchor.
001813:         owner_history = list(state.get('history', []) or [])
001814:         committed = [entry for entry in getattr(self, 'motion_history', [])
001815:                      if entry.get('frame', -1) < current and entry.get('pos') is not None]
001816:         if (committed and committed[-1]['frame'] == current - 1 and
001817:                 tuple(committed[-1]['pos']) == tuple(previous_position) and
001818:                 (not owner_history or owner_history[-1]['frame'] < current - 1)):
001819:             recent = [committed[-1]]
001820:             for entry in reversed(committed[:-1]):
001821:                 if entry['frame'] != recent[0]['frame'] - 1 or len(recent) >= 4:
001822:                     break
001823:                 recent.insert(0, entry)
001824:             state['history'] = [dict(entry, score=1.0) for entry in recent]
001825:         owner_snapshot = self._snapshot_tracking_state_for_provisional_guard()
001826:         saved_range = getattr(self, '_force_local_ai_range', None)
001827:         saved_history = list(getattr(self, '_force_local_ai_history', []) or [])
001828:         saved_radius = float(getattr(self, '_force_local_ai_radius', 450.0))
001829:         saved_min_score = float(getattr(self, '_force_local_ai_min_score', 0.985))
001830:         saved_return_none = bool(getattr(self, '_force_local_ai_return_none_on_miss', False))
001831:         saved_max_scored = int(getattr(self, '_force_local_ai_max_score_candidates', 128))
001832:         saved_allow_gap = bool(getattr(self, '_force_local_ai_allow_history_gap', False))
001833:         saved_continuity_guard = bool(getattr(self, '_force_local_ai_continuity_guard', False))
001834:         try:
001835:             self._force_local_ai_range = (
001836:                 int(state['start_frame']), int(state['deadline'])
001837:             )
001838:             self._force_local_ai_history = list(state.get('history', []) or [])
001839:             miss_count = int(state.get('miss_count', 0))
001840:             base_radius = float(getattr(self, '_contact_local_ai_radius', 450.0))
001841:             self._force_local_ai_radius = min(900.0, base_radius + 90.0 * miss_count)
001842:             self._force_local_ai_min_score = float(
001843:                 getattr(self, '_contact_local_ai_min_score', 0.985)
001844:             )
001845:             self._force_local_ai_return_none_on_miss = True
001846:             self._force_local_ai_max_score_candidates = 96
001847:             self._force_local_ai_allow_history_gap = True
001848:             self._force_local_ai_continuity_guard = True
001849:             selected = self._force_local_ai_frame(frame, previous_position)
001850:             state['history'] = list(getattr(self, '_force_local_ai_history', []) or [])
001851:         finally:
001852:             self._force_local_ai_range = saved_range
001853:             self._force_local_ai_history = saved_history
001854:             self._force_local_ai_radius = saved_radius
001855:             self._force_local_ai_min_score = saved_min_score
001856:             self._force_local_ai_return_none_on_miss = saved_return_none
001857:             self._force_local_ai_max_score_candidates = saved_max_scored
001858:             self._force_local_ai_allow_history_gap = saved_allow_gap
001859:             self._force_local_ai_continuity_guard = saved_continuity_guard
001860: 
001861:         if selected is None:
001862:             state['miss_count'] = int(state.get('miss_count', 0)) + 1
001863:             self._contact_local_ai_state = state
001864:             print(
001865:                 f"[CONTACT_LOCAL_AI_MISS] f{current}: normal tracker remains active; "
001866:                 f"misses={state['miss_count']}"
001867:             )
001868:             return None
001869: 
001870:         state['miss_count'] = 0
001871:         selected = (int(selected[0]), int(selected[1]))
001872:         self._record_contact_ai_launch(previous_position, selected, owner_snapshot, frame)
001873:         same_side_bounce, same_side_reason = self._same_side_pre_net_bounce_candidate(
001874:             selected, frame, require_observed_reversal=True
001875:         )
001876:         if same_side_bounce:
001877:             self._pending_rally_end_reason = same_side_reason
001878:             self._pending_rally_end_frame = current
001879:             self._contact_local_ai_state = None
001880:             self._local_ai_follow_until_frame = min(
001881:                 int(getattr(self, '_local_ai_follow_until_frame', current)), current - 1
001882:             )
001883:             print(
001884:                 f"[CONTACT_LOCAL_AI SAME-SIDE BOUNCE] f{current}: "
001885:                 f"{same_side_reason} at {selected}"
001886:             )
001887:             return selected
001888:         zone = self._player_point_zone(selected)
001889:         inside_contact = self._point_in_player_contact_corridor(selected)
001890:         age = current - int(state['start_frame'])
001891:         history = list(state.get('history', []) or [])
001892:         immediate_reversal = False
001893:         reversal_angle = 0.0
001894:         if (
001895:                 not str(state.get('reason', '')).startswith(
001896:                     'trajectory-hotspot-turn:'
001897:                 ) and
001898:                 2 <= age <= 5 and inside_contact and len(history) >= 3):
001899:             p0 = tuple(history[-3].get('pos', ()))
001900:             p1 = tuple(history[-2].get('pos', ()))
001901:             p2 = tuple(history[-1].get('pos', ()))
001902:             if len(p0) == 2 and len(p1) == 2 and len(p2) == 2:
001903:                 v1 = (float(p1[0] - p0[0]), float(p1[1] - p0[1]))
001904:                 v2 = (float(p2[0] - p1[0]), float(p2[1] - p1[1]))
001905:                 len1 = math.hypot(v1[0], v1[1])
001906:                 len2 = math.hypot(v2[0], v2[1])
001907:                 if len1 > 0.0 and len2 > 0.0:
001908:                     cosine = (
001909:                         v1[0] * v2[0] + v1[1] * v2[1]
001910:                     ) / (len1 * len2)
001911:                     cosine = max(-1.0, min(1.0, cosine))
001912:                     reversal_angle = math.degrees(math.acos(cosine))
001913:                     immediate_reversal = (
001914:                         10.0 <= len1 <= 150.0 and
001915:                         10.0 <= len2 <= 150.0 and
001916:                         v1[1] <= -10.0 and v2[1] >= 20.0 and
001917:                         reversal_angle >= 100.0
001918:                     )
001919:         if immediate_reversal:
001920:             self._restore_tracking_state_for_provisional_guard(owner_snapshot)
001921:             self._contact_local_ai_state = None
001922:             self._contact_local_ai_cooldown_until_frame = current + 2
001923:             self._contact_normal_fallback_path = []
001924:             print(
001925:                 f"[CONTACT_LOCAL_AI IMMEDIATE-REVERSAL RELEASE] f{current}: "
001926:                 f"AI path={[entry.get('pos') for entry in history[-3:]]} "
001927:                 f"angle={reversal_angle:.1f}deg returned into contact corridor; "
001928:                 "retrying normal tracker"
001929:             )
001930:             return None
001931: 
001932:         contact_out_bounce = self._contact_ai_out_bounce_candidate(
001933:             state, frame
001934:         )
001935:         if contact_out_bounce is not None:
001936:             bounce_position = contact_out_bounce['position']
001937:             bounce_reason = contact_out_bounce['reason']
001938:             self._pending_rally_end_reason = bounce_reason
001939:             self._pending_rally_end_frame = current
001940:             self._last_confirmed_pending_out_frame = current - 1
001941:             self._last_confirmed_pending_out_position = bounce_position
001942:             self._last_confirmed_pending_out_confirm_frame = current
001943:             self._last_confirmed_pending_out_reason = bounce_reason
001944:             self.ball_center = bounce_position
001945:             self._contact_local_ai_state = None
001946:             print(
001947:                 f"[CONTACT_LOCAL_AI OUT-BOUNCE] f{current}: "
001948:                 f"bounce={bounce_position} reason={bounce_reason} "
001949:                 f"turn={contact_out_bounce['angle']:.1f}deg"
001950:             )
001951:             return bounce_position
001952:         if zone is None and not inside_contact:
001953:             state['outside_count'] = int(state.get('outside_count', 0)) + 1
001954:         else:
001955:             state['outside_count'] = 0
001956: 
001957:         print(
001958:             f"[CONTACT_LOCAL_AI] f{current}: selected={selected} age={age}f "
001959:             f"outside={state['outside_count']} zone={zone}"
001960:         )
001961: 
001962:         # Keep AI for several frames after contact.  This prevents an immediate
001963:         # handback to HSV while its search anchor is still the racket/player.
001964:         # A trajectory-hotspot takeover is not a player-contact recovery; keep
001965:         # it through six transitions so it can bridge the bad HSV zone, then
001966:         # hand control back on the following frame when normal HSV is coherent.
001967:         minimum_owner_age = (
001968:             6
001969:             if str(state.get('reason', '')).startswith('trajectory-hotspot-turn:')
001970:             else 5
001971:         )
001972:         net_transit_geometry = self._net_contact_geometry(selected)
001973:         in_net_transit = bool(
001974:             net_transit_geometry is not None and
001975:             net_transit_geometry.get('near_hull', False)
001976:         )
001977:         if (
001978:                 age >= minimum_owner_age and
001979:                 int(state.get('outside_count', 0)) >= 3 and
001980:                 not in_net_transit):
001981:             print(
001982:                 f"[CONTACT_LOCAL_AI_HANDOFF] f{current}: selected={selected} "
001983:                 f"clear of player/contact corridor for {state['outside_count']}f"
001984:             )
001985:             self._contact_local_ai_state = None
001986:             self._contact_local_ai_cooldown_until_frame = current + 3
001987:             self._contact_local_ai_handoff_frame = current
001988:         else:
001989:             if (
001990:                     age >= minimum_owner_age and
001991:                     int(state.get('outside_count', 0)) >= 3 and
001992:                     in_net_transit):
001993:                 print(
001994:                     f"[CONTACT_LOCAL_AI NET-TRANSIT HOLD] f{current}: "
001995:                     f"selected={selected} remains in net corridor"
001996:                 )
001997:             self._contact_local_ai_state = state
001998:         return selected
001999: 
002000:     def _record_contact_ai_launch(self, previous_position, selected, snapshot, frame):
002001:         """Retain a near player's visible racket turn when AI owns tracking."""
002002:         incoming = (snapshot or {}).get('last_motion') or {}
002003:         if (getattr(self, '_awaiting_serve_bounce', False) or
002004:                 float(incoming.get('dy', 0.0)) < 25.0 or
002005:                 self._player_point_zone(previous_position) not in ('player_body', 'racket_fragment') or
002006:                 self._player_point_zone(selected) is not None):
002007:             return
002008:         dx = float(selected[0] - previous_position[0])
002009:         dy = float(selected[1] - previous_position[1])
002010:         if (dy > -70.0 or not 70.0 <= math.hypot(dx, dy) <= 400.0 or
002011:                 previous_position[1] <= float(getattr(self, 'net_area_y_max', frame.shape[0])) + 55.0):
002012:             return
002013:         player = self._player_index_on_court_end('near')
002014:         if (getattr(self, '_last_racket_contact_player', None) == player and
002015:                 self.frame_count - getattr(self, '_last_racket_contact_frame', -1000000) <= 12):
002016:             return
002017:         self._record_racket_contact(previous_position, label='AI-confirmed racket contact', player_idx_override=player)
002018:         self.ground_bounce_count = 0
002019:         self.last_ground_bounce_frame = -1000000
002020: 
002021:     def _post_contact_handoff_player_jump_invalid(
002022:             self, previous_position, tracked_position, snapshot):
002023:         """Reject a player jump immediately after a clean contact-AI handoff."""
002024:         if previous_position is None or tracked_position is None:
002025:             return False
002026:         age = int(self.frame_count) - int(getattr(
002027:             self, '_contact_local_ai_handoff_frame', -1000000
002028:         ))
002029:         if age < 1 or age > 3:
002030:             return False
002031:         zone = self._player_point_zone(tracked_position)
002032:         player_zone = zone in (
002033:                 'player_head_hat', 'player_body', 'player_shoes',
002034:                 'racket_fragment')
002035: 
002036:         prior = dict((snapshot or {}).get('last_motion') or {})
002037:         prior_dx = float(prior.get('dx', 0.0) or 0.0)
002038:         prior_dy = float(prior.get('dy', 0.0) or 0.0)
002039:         prior_speed = float(
002040:             prior.get('distance', math.hypot(prior_dx, prior_dy)) or 0.0
002041:         )
002042:         dx = float(tracked_position[0] - previous_position[0])
002043:         dy = float(tracked_position[1] - previous_position[1])
002044:         speed = math.hypot(dx, dy)
002045:         angle = 0.0
002046:         if prior_speed > 0.0 and speed > 0.0:
002047:             cosine = (prior_dx * dx + prior_dy * dy) / (prior_speed * speed)
002048:             cosine = max(-1.0, min(1.0, cosine))
002049:             angle = math.degrees(math.acos(cosine))
002050:         extreme_reverse_jump = (
002051:             speed >= 300.0 and dy >= 250.0 and angle >= 150.0 and
002052:             speed >= prior_speed * 3.5
002053:         )
002054:         return (
002055:             (player_zone or extreme_reverse_jump) and
002056:             prior_speed >= 30.0 and prior_dy <= -20.0 and
002057:             speed >= 150.0 and dy >= 100.0 and angle >= 120.0
002058:         )
002059: 
002060:     def _current_terminal_ground_bounce_pending(self):
002061:         """Return whether this frame already confirmed a terminal ground bounce.
002062: 
002063:         Contact Local-AI protects a tracker from player fragments.  Once the
002064:         ordinary tracker has independently registered a geometric out bounce,
002065:         however, rolling its state back to arm another contact search would
002066:         erase that terminal result.  This identifies only an out or second
002067:         bounce registered on the current frame; tentative timeouts and other
002068:         queued results remain eligible for normal recovery.
002069:         """
002070:         reason = str(getattr(self, '_pending_rally_end_reason', '') or '').lower()
002071:         pending_frame = int(getattr(self, '_pending_rally_end_frame', -1000000))
002072:         bounce_frame = int(getattr(self, 'last_ground_bounce_frame', -1000000))
002073:         return (
002074:             pending_frame == int(self.frame_count) and
002075:             bounce_frame == int(self.frame_count) and
002076:             (
002077:                 'bounce outside' in reason or
002078:                 'bounced out' in reason or
002079:                 'bounced twice' in reason or
002080:                 'bounced before crossing' in reason
002081:             )
002082:         )
002083: 
002084:     def _contact_ai_out_bounce_candidate(self, state, frame):
002085:         """Confirm a short, visible sideline rebound tracked by Contact AI."""
002086:         history = list((state or {}).get('history', []) or [])
002087:         if len(history) < 3 or frame is None:
002088:             return None
002089:         recent = history[-3:]
002090:         frames = [int(entry.get('frame', -1000000)) for entry in recent]
002091:         points = [tuple(entry.get('pos', ())) for entry in recent]
002092:         if (
002093:                 any(len(point) != 2 for point in points) or
002094:                 frames != list(range(frames[0], frames[0] + 3)) or
002095:                 frames[-1] != int(self.frame_count)):
002096:             return None
002097:         p0, bounce, p2 = points
002098:         v1 = (float(bounce[0] - p0[0]), float(bounce[1] - p0[1]))
002099:         v2 = (float(p2[0] - bounce[0]), float(p2[1] - bounce[1]))
002100:         length1 = math.hypot(v1[0], v1[1])
002101:         length2 = math.hypot(v2[0], v2[1])
002102:         if not (10.0 <= length1 <= 80.0 and 10.0 <= length2 <= 80.0):
002103:             return None
002104:         cosine = (v1[0] * v2[0] + v1[1] * v2[1]) / (length1 * length2)
002105:         cosine = max(-1.0, min(1.0, cosine))
002106:         angle = math.degrees(math.acos(cosine))
002107:         if v1[1] < 10.0 or v2[1] > -10.0 or angle < 120.0:
002108:             return None
002109:         outside, side, _, _, _, _ = self._point_outside_singles_court(
002110:             bounce, frame
002111:         )
002112:         if not outside or side not in ('left sideline', 'right sideline'):
002113:             return None
002114:         return {
002115:             'position': (int(bounce[0]), int(bounce[1])),
002116:             'reason': f'Ball bounced out of court ({side})',
002117:             'angle': angle,
002118:         }
002119: 
002120:     def _promote_contact_visible_rejected_path(self, rejected_candidates):
002121:         """Recover a moving ball path that outgrew a stale contact anchor.
002122: 
002123:         The single-frame jump guard remains strict. Three consecutive
002124:         high-confidence candidates with real image motion provide independent
002125:         temporal evidence that the ball continued away from the old anchor.
002126:         """
002127:         state = getattr(self, '_contact_local_ai_state', None)
002128:         if (
002129:                 not bool(getattr(self, '_force_local_ai_continuity_guard', False)) or
002130:                 not isinstance(state, dict)):
002131:             return None
002132: 
002133:         current = int(self.frame_count)
002134:         moving = []
002135:         for item in rejected_candidates:
002136:             motion = item.get('motion') or {}
002137:             area = float(item['candidate'].get('area', 0.0) or 0.0)
002138:             if (
002139:                     item.get('rejection') == 'unsupported-jump' and
002140:                     float(item.get('score', 0.0) or 0.0) >= float(
002141:                         getattr(self, '_force_local_ai_min_score', 0.985)
002142:                     ) and
002143:                     3.0 <= area <= 500.0 and
002144:                     float(motion.get('mean', 0.0) or 0.0) >= 5.0 and
002145:                     float(motion.get('max', 0.0) or 0.0) >= 30.0):
002146:                 moving.append(item)
002147: 
002148:         path = list(state.get('rejected_visible_path', []) or [])
002149:         if not moving:
002150:             if path and int(path[-1].get('frame', -1000000)) < current:
002151:                 state['rejected_visible_path'] = []
002152:             return None
002153: 
002154:         if path and int(path[-1].get('frame', -1000000)) == current:
002155:             path = path[:-1]
002156:         elif path and int(path[-1].get('frame', -1000000)) != current - 1:
002157:             path = []
002158: 
002159:         expected = None
002160:         if len(path) >= 2:
002161:             p0 = tuple(path[-2]['point'])
002162:             p1 = tuple(path[-1]['point'])
002163:             expected = (2 * p1[0] - p0[0], 2 * p1[1] - p0[1])
002164: 
002165:         def candidate_key(item):
002166:             point = tuple(item['point'])
002167:             continuation = (
002168:                 math.hypot(point[0] - expected[0], point[1] - expected[1])
002169:                 if expected is not None else 0.0
002170:             )
002171:             return (continuation, -float(item.get('score', 0.0) or 0.0))
002172: 
002173:         chosen = min(moving, key=candidate_key)
002174:         path.append({
002175:             'frame': current,
002176:             'point': tuple(chosen['point']),
002177:             'score': float(chosen.get('score', 0.0) or 0.0),
002178:             'area': float(chosen['candidate'].get('area', 0.0) or 0.0),
002179:             'motion': dict(chosen.get('motion') or {}),
002180:         })
002181:         path = path[-3:]
002182:         state['rejected_visible_path'] = path
002183:         if len(path) < 3:
002184:             return None
002185: 
002186:         p0, p1, p2 = (tuple(sample['point']) for sample in path)
002187:         v1 = (float(p1[0] - p0[0]), float(p1[1] - p0[1]))
002188:         v2 = (float(p2[0] - p1[0]), float(p2[1] - p1[1]))
002189:         len1 = math.hypot(*v1)
002190:         len2 = math.hypot(*v2)
002191:         cosine = (
002192:             (v1[0] * v2[0] + v1[1] * v2[1]) /
002193:             max(1.0, len1 * len2)
002194:         )
002195:         total = math.dist(p0, p2)
002196:         areas = [float(sample['area']) for sample in path]
002197:         coherent = (
002198:             5.0 <= len1 <= 260.0 and
002199:             5.0 <= len2 <= 260.0 and
002200:             cosine >= 0.85 and
002201:             total >= 25.0 and
002202:             max(areas) <= max(3.0, min(areas)) * 4.0
002203:         )
002204:         if not coherent:
002205:             return None
002206: 
002207:         promoted = dict(chosen)
002208:         promoted['visible_path'] = [dict(sample) for sample in path]
002209:         promoted['motion_anchor'] = p1
002210:         state['rejected_visible_path'] = []
002211:         print(
002212:             f"[CONTACT_LOCAL_AI VISIBLE-PATH RECOVERY] f{current}: "
002213:             f"path={[sample['point'] for sample in path]} "
002214:             f"cos={cosine:.3f} travel={total:.1f}px "
002215:             f"ai={promoted['score']:.6f}"
002216:         )
002217:         return promoted
002218: 
002219:     def _force_local_ai_frame(self, frame, previous_position):
002220:         """TEST ONLY: bypass HSV and let Local AI own an exact frame range.
002221: 
002222:         The previous AI-selected point becomes the next search anchor. Player,
002223:         head, shoe, and racket regions are excluded before scoring. Once two AI
002224:         points exist, high-confidence candidates must also fit the extrapolated
002225:         AI trajectory. Outside --force-local-ai-range this method returns None
002226:         and the normal tracker is unchanged.
002227:         """
002228:         frame_range = getattr(self, "_force_local_ai_range", None)
002229:         recovery = getattr(self, "local_ai_recovery", None)
002230:         if frame_range is None or frame is None or recovery is None:
002231:             return None
002232: 
002233:         start_frame, end_frame = frame_range
002234:         current = int(self.frame_count)
002235:         if not (int(start_frame) <= current <= int(end_frame)):
002236:             if current > int(end_frame):
002237:                 self._force_local_ai_history = []
002238:             return None
002239: 
002240:         from ball_ai_recovery_probe import collect_candidates
002241: 
002242:         history = list(getattr(self, "_force_local_ai_history", []) or [])
002243:         # Keep the last validated history across misses. Normal HSV positions
002244:         # during recovery are hypotheses and must not bridge this history.
002245:         if not history and previous_position is not None:
002246:             history = [{"frame": current - 1, "pos": tuple(previous_position)}]
002247:         self._force_local_ai_history = history
```

## Lines 2312-2522
```python
002312:         try:
002313:             scored = recovery._score(frame, current, subset) if subset else []
002314:         finally:
002315:             recovery._diagnostic_unthrottled = saved_unthrottled
002316: 
002317:         tolerance = None
002318:         if predicted is not None:
002319:             tolerance = max(120.0, min(300.0, previous_speed * 1.8 + 60.0))
002320: 
002321:         recent = list(getattr(self, "ball_velocity_history", []) or [])[-5:]
002322:         prior_speed = float(np.median(recent)) if recent else 0.0
002323:         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
002324:         previous_gray = getattr(self, '_local_ai_tight_roi_previous_gray', None)
002325:         if previous_gray is None:
002326:             previous_gray = getattr(self, '_prev_frame_gray', None)
002327: 
002328:         eligible = []
002329:         visible_rejected = []
002330:         continuity_rejected = 0
002331:         for candidate in scored:
002332:             score = float(candidate.get("ai_score", 0.0) or 0.0)
002333:             if score < min_score:
002334:                 continue
002335:             point = (int(candidate["x"]), int(candidate["y"]))
002336:             if self._player_point_zone(point) is not None:
002337:                 continue
002338: 
002339:             anchor_dist = math.hypot(point[0] - anchor[0], point[1] - anchor[1])
002340:             motion = self._candidate_motion_metrics(
002341:                 gray, point[0], point[1], previous_gray=previous_gray)
002342:             rejection = validate_ai_step(
002343:                 point, current, history, prior_speed=prior_speed, motion=motion)
002344:             if rejection:
002345:                 continuity_rejected += 1
002346:                 visible_rejected.append({
002347:                     "candidate": candidate,
002348:                     "point": point,
002349:                     "score": score,
002350:                     "motion": motion,
002351:                     "rejection": rejection,
002352:                 })
002353:                 continue
002354: 
002355:             pred_dist = None
002356:             cosine = None
002357:             if predicted is not None:
002358:                 pred_dist = math.hypot(point[0] - predicted[0], point[1] - predicted[1])
002359:                 if pred_dist > float(tolerance):
002360:                     continue
002361:                 if previous_vector is not None:
002362:                     sx = float(point[0] - anchor[0])
002363:                     sy = float(point[1] - anchor[1])
002364:                     step = math.hypot(sx, sy)
002365:                     if step >= 3.0 and previous_speed >= 3.0:
002366:                         cosine = (
002367:                             previous_vector[0] * sx + previous_vector[1] * sy
002368:                         ) / (previous_speed * step)
002369:                         # A supported physical step may reverse at contact.
002370:                         # The speed/image gate above remains mandatory.
002371: 
002372:             eligible.append({
002373:                 "candidate": candidate,
002374:                 "point": point,
002375:                 "score": score,
002376:                 "pred_dist": pred_dist,
002377:                 "cosine": cosine,
002378:             })
002379: 
002380:         promoted_visible = self._promote_contact_visible_rejected_path(
002381:             visible_rejected
002382:         )
002383:         if promoted_visible is not None:
002384:             point = tuple(promoted_visible["point"])
002385:             pred_dist = (
002386:                 math.hypot(point[0] - predicted[0], point[1] - predicted[1])
002387:                 if predicted is not None else None
002388:             )
002389:             cosine = None
002390:             if previous_vector is not None:
002391:                 sx = float(point[0] - anchor[0])
002392:                 sy = float(point[1] - anchor[1])
002393:                 step = math.hypot(sx, sy)
002394:                 if step >= 3.0 and previous_speed >= 3.0:
002395:                     cosine = (
002396:                         previous_vector[0] * sx + previous_vector[1] * sy
002397:                     ) / (previous_speed * step)
002398:             promoted_visible["pred_dist"] = pred_dist
002399:             promoted_visible["cosine"] = cosine
002400:             eligible.append(promoted_visible)
002401:         elif eligible:
002402:             state = getattr(self, '_contact_local_ai_state', None)
002403:             if isinstance(state, dict):
002404:                 state['rejected_visible_path'] = []
002405: 
002406:         if predicted is None:
002407:             eligible.sort(
002408:                 key=lambda item: (
002409:                     -item["score"],
002410:                     math.hypot(item["point"][0] - anchor[0], item["point"][1] - anchor[1]),
002411:                 )
002412:             )
002413:         else:
002414:             eligible.sort(key=lambda item: (float(item["pred_dist"]), -item["score"]))
002415:         if promoted_visible is not None:
002416:             eligible.remove(promoted_visible)
002417:             eligible.insert(0, promoted_visible)
002418: 
002419:         # CONTACT_LOCAL_AI_FIRST_FRAME_CONSENSUS_V2
002420:         # On the first Local-AI frame, the normal tracker has independently
002421:         # produced the contact hypothesis that armed AI. If AI also finds a
002422:         # near-identical candidate with a score tied with its best candidate,
002423:         # prefer that cross-detector agreement over anchor distance. This is
002424:         # intentionally limited to the first AI step (predicted is None), so a
002425:         # later bad HSV hypothesis cannot steer an established AI trajectory.
002426:         debug_normal_candidate = getattr(
002427:             self, "_contact_local_ai_debug_normal_candidate", None
002428:         )
002429:         best_ai_score = max((float(item["score"]) for item in eligible), default=0.0)
002430:         if (
002431:                 debug_normal_candidate is not None and
002432:                 eligible and
002433:                 predicted is None):
002434:             ai_tie_margin = 1.0e-4
002435:             consensus_pool = [
002436:                 item for item in eligible
002437:                 if best_ai_score - float(item["score"]) <= ai_tie_margin
002438:             ]
002439:             if consensus_pool:
002440:                 consensus = min(
002441:                     consensus_pool,
002442:                     key=lambda item: math.hypot(
002443:                         float(item["point"][0]) - float(debug_normal_candidate[0]),
002444:                         float(item["point"][1]) - float(debug_normal_candidate[1]),
002445:                     ),
002446:                 )
002447:                 consensus_dist = math.hypot(
002448:                     float(consensus["point"][0]) - float(debug_normal_candidate[0]),
002449:                     float(consensus["point"][1]) - float(debug_normal_candidate[1]),
002450:                 )
002451:                 default = eligible[0]
002452:                 default_dist = math.hypot(
002453:                     float(default["point"][0]) - float(debug_normal_candidate[0]),
002454:                     float(default["point"][1]) - float(debug_normal_candidate[1]),
002455:                 )
002456:                 if (
002457:                         consensus is not default and
002458:                         consensus_dist <= 35.0 and
002459:                         default_dist >= max(100.0, consensus_dist + 70.0)):
002460:                     eligible.remove(consensus)
002461:                     eligible.insert(0, consensus)
002462:                     print(
002463:                         f"[FORCE_LOCAL_AI_NORMAL_TIEBREAK] f{current}: "
002464:                         f"normal={debug_normal_candidate} chose={consensus['point']} "
002465:                         f"normal_dist={consensus_dist:.1f}px ai={consensus['score']:.12f} "
002466:                         f"over={default['point']} default_normal_dist={default_dist:.1f}px "
002467:                         f"default_ai={default['score']:.12f}"
002468:                     )
002469:         for rank, item in enumerate(eligible[:5], 1):
002470:             pd = "n/a" if item["pred_dist"] is None else f"{item['pred_dist']:.1f}"
002471:             cs = "n/a" if item["cosine"] is None else f"{item['cosine']:.3f}"
002472:             anchor_dist = math.hypot(
002473:                 float(item["point"][0]) - float(anchor[0]),
002474:                 float(item["point"][1]) - float(anchor[1]),
002475:             )
002476:             trajectory_cost = (
002477:                 float(item["pred_dist"])
002478:                 if item["pred_dist"] is not None else anchor_dist
002479:             )
002480:             normal_dist = None
002481:             if debug_normal_candidate is not None:
002482:                 normal_dist = math.hypot(
002483:                     float(item["point"][0]) - float(debug_normal_candidate[0]),
002484:                     float(item["point"][1]) - float(debug_normal_candidate[1]),
002485:                 )
002486:             nd = "n/a" if normal_dist is None else f"{normal_dist:.1f}"
002487:             ai_gap = best_ai_score - float(item["score"])
002488:             print(
002489:                 f"[FORCE_LOCAL_AI_CAND] f{current} #{rank} pos={item['point']} "
002490:                 f"ai={item['score']:.12f} ai_gap={ai_gap:.12g} "
002491:                 f"pred_dist={pd} cos={cs} anchor_dist={anchor_dist:.1f} "
002492:                 f"trajectory_cost={trajectory_cost:.1f} "
002493:                 f"normal_ref={debug_normal_candidate} normal_dist={nd}"
002494:             )
002495: 
002496:         if not eligible:
002497:             if bool(getattr(self, "_force_local_ai_return_none_on_miss", False)):
002498:                 print(
002499:                     f"[FORCE_LOCAL_AI_MISS] f{current}: no clear candidate >= {min_score:.3f}; "
002500:                     f"anchor={anchor} predicted={predicted} candidates={len(candidates)} "
002501:                     f"clear={len(clear_candidates)} scored={len(scored)} "
002502:                     f"continuity_rejected={continuity_rejected}"
002503:                 )
002504:                 return None
002505:             self.ball_center = tuple(anchor)
002506:             self.stuck_frame_count = max(1, int(getattr(self, "stuck_frame_count", 0)) + 1)
002507:             print(
002508:                 f"[FORCE_LOCAL_AI_HOLD] f{current}: no clear candidate >= {min_score:.3f}; "
002509:                 f"anchor={anchor} predicted={predicted} candidates={len(candidates)} "
002510:                 f"clear={len(clear_candidates)} scored={len(scored)}"
002511:             )
002512:             return tuple(anchor)
002513: 
002514:         selected = eligible[0]
002515:         candidate = selected["candidate"]
002516:         point = selected["point"]
002517:         area = float(candidate.get("area", 0.0) or 0.0)
002518: 
002519:         motion_anchor = tuple(selected.get("motion_anchor", anchor))
002520:         dx = int(point[0] - motion_anchor[0])
002521:         dy = int(point[1] - motion_anchor[1])
002522:         distance = math.hypot(dx, dy)
```

## Lines 3113-3323
```python
003113:         return selected_candidate
003114: 
003115:     def _try_local_ai_recovery(
003116:             self, previous_position, tracked_position, previous_stuck,
003117:             pre_track_snapshot=None, frame=None):
003118:         """Repair a suspect selection from the recent raw-frame buffer if safe."""
003119:         if (getattr(self, '_temporal_top_reentry_frame', -1) == self.frame_count and
003120:                 getattr(self, '_temporal_top_reentry_pos', None) == tracked_position):
003121:             return tracked_position
003122:         confirmed_static_artifact = self._finalize_pending_provisional_static_candidate()
003123:         if (
003124:                 confirmed_static_artifact and previous_position is not None and
003125:                 tracked_position is not None and pre_track_snapshot is not None and
003126:                 math.hypot(
003127:                     float(tracked_position[0]) - float(getattr(self, "_last_static_guard_position", tracked_position)[0]),
003128:                     float(tracked_position[1]) - float(getattr(self, "_last_static_guard_position", tracked_position)[1]),
003129:                 ) <= 22.0):
003130:             self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
003131:             # Re-run the normal candidate selection now that the confirmed
003132:             # static blob is ignored.  This keeps the real next-frame toss
003133:             # (a prior frame here) instead of deliberately dropping a visible ball.
003134:             retracked = self.track_ball_in_frame(frame) if frame is not None else None
003135:             if retracked is not None and math.hypot(
003136:                     float(retracked[0]) - float(self._last_static_guard_position[0]),
003137:                     float(retracked[1]) - float(self._last_static_guard_position[1]),
003138:             ) > 22.0:
003139:                 print(
003140:                     f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
003141:                     f"retracked {retracked} after excluding static selection"
003142:                 )
003143:                 return retracked
003144:             self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
003145:             self.stuck_frame_count = int(previous_stuck) + 1
003146:             print(
003147:                 f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
003148:                 f"held current static selection {tracked_position}; resuming from {previous_position}"
003149:             )
003150:             return previous_position
003151:         # Keep a weak player/background reacquisition from becoming the
003152:         # next tracking anchor before tight local AI can arbitrate it.
003153:         if (
003154:                 pre_track_snapshot is not None and
003155:                 previous_position is not None and
003156:                 tracked_position is not None and
003157:                 int(previous_stuck or 0) >= 3):
003158:             reacq_jump = math.hypot(
003159:                 float(tracked_position[0]) - float(previous_position[0]),
003160:                 float(tracked_position[1]) - float(previous_position[1]),
003161:             )
003162:             reacq_motion_mean = float(
003163:                 getattr(self, '_last_tracked_candidate_motion_mean', 0.0) or 0.0
003164:             )
003165:             reacq_motion_max = float(
003166:                 getattr(self, '_last_tracked_candidate_motion_max', 0.0) or 0.0
003167:             )
003168:             reacq_zone = self._player_point_zone(tracked_position)
003169:             weak_player_candidate = (
003170:                 reacq_zone is not None and
003171:                 reacq_motion_mean < 8.0 and
003172:                 reacq_motion_max < 50.0
003173:             )
003174:             weak_long_jump = (
003175:                 reacq_jump >= 150.0 and
003176:                 reacq_motion_mean < 12.0 and
003177:                 reacq_motion_max < 80.0
003178:             )
003179:             if weak_player_candidate or weak_long_jump:
003180:                 rejected_position = tuple(tracked_position)
003181:                 self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
003182:                 self.stuck_frame_count = int(previous_stuck or 0) + 1
003183:                 tracked_position = tuple(previous_position)
003184:                 if getattr(self, '_last_motion_reacq_frame', -1000000) == self.frame_count:
003185:                     self._last_motion_reacq_frame = -1000000
003186:                     self._last_motion_reacq_pos = None
003187:                 if weak_player_candidate:
003188:                     print(
003189:                         f'[PLAYER_REACQ_WEAK_PLAYER_REJECTED] f{self.frame_count}: '
003190:                         f'rejected {rejected_position} zone={reacq_zone} '
003191:                         f'motion={reacq_motion_mean:.1f}/{reacq_motion_max:.1f}; '
003192:                         f'restored {previous_position} stuck={self.stuck_frame_count}'
003193:                     )
003194:                 else:
003195:                     print(
003196:                         f'[PLAYER_REACQ_WEAK_JUMP_REJECTED] f{self.frame_count}: '
003197:                         f'rejected {rejected_position} jump={reacq_jump:.1f}px '
003198:                         f'motion={reacq_motion_mean:.1f}/{reacq_motion_max:.1f}; '
003199:                         f'restored {previous_position} stuck={self.stuck_frame_count}'
003200:                     )
003201: 
003202:         if self._normal_ball_evidence_is_strong(frame, previous_position, tracked_position, pre_track_snapshot):
003203:             self._contact_local_ai_state = None
003204:             return tracked_position
003205:         if not self._automatic_local_ai_scoring_ready():
003206:             return tracked_position
003207:         tight_hold_recovery = self._try_active_tight_local_ai_hold(
003208:             frame,
003209:             tracked_position,
003210:             previous_gray=(pre_track_snapshot or {}).get('_prev_frame_gray'),
003211:         )
003212:         if tight_hold_recovery is not None:
003213:             return tight_hold_recovery
003214:         reason = self._local_ai_recovery_reason(
003215:             previous_position, tracked_position, previous_stuck
003216:         )
003217:         if reason is None:
003218:             self._maybe_clear_post_serve_pre_net_recovery(
003219:                 previous_position, tracked_position, source='normal-tracker'
003220:             )
003221:             return tracked_position
003222:         post_serve_recovery_active = (
003223:             bool(getattr(self, '_post_serve_pre_net_recovery_active', False)) or
003224:             int(self.frame_count) <= max(
003225:                 int(getattr(self, "_post_serve_launch_lock_until_frame", -1)),
003226:                 int(getattr(self, "_post_serve_recovery_hold_until_frame", -1000000)),
003227:             )
003228:         )
003229:         trajectory_recovered = self._try_local_ai_trajectory_rescue(
003230:             frame,
003231:             previous_position,
003232:             tracked_position,
003233:             previous_stuck,
003234:             pre_track_snapshot,
003235:             reason,
003236:         )
003237:         if trajectory_recovered is not None:
003238:             recovered = trajectory_recovered
003239:             reason = f"{reason}+trajectory"
003240:         else:
003241:             recovery_reason = reason
003242:             if post_serve_recovery_active and reason.startswith("player-region:"):
003243:                 # Keep using the post-serve low-saturation candidate pool on
003244:                 # the frames after the first rejected jump.  The normal tracker
003245:                 # may keep proposing a player contour until the ball emerges.
003246:                 recovery_reason = f"post-serve-launch-occlusion:{reason}"
003247:                 print(
003248:                     f"[POST_SERVE_RECOVERY_RETRY] f{self.frame_count}: "
003249:                     f"{reason} -> {recovery_reason}"
003250:                 )
003251:             recovered = self.local_ai_recovery.recover(
003252:                 self.frame_count,
003253:                 self._local_ai_frame_buffer,
003254:                 predicted_position=previous_position,
003255:                 player_zone=self._player_point_zone,
003256:                 reason=recovery_reason,
003257:                 force=False,
003258:             )
003259:         if recovered is None:
003260:             if getattr(self.local_ai_recovery, "last_rejection", None) == "all-player-body-path":
003261:                 self._local_ai_all_body_rejections = int(
003262:                     getattr(self, "_local_ai_all_body_rejections", 0)
003263:                 ) + 1
003264:                 point_age = int(self.frame_count) - int(
003265:                     getattr(self, "point_start_frame_internal", self.frame_count) or self.frame_count
003266:                 )
003267:                 if (
003268:                         self._local_ai_all_body_rejections >= 1 and
003269:                         not post_serve_recovery_active and
003270:                         # A previously accepted non-body recovery is positive
003271:                         # evidence that this is a real serve/rally.  A later
003272:                         # player-body-only attempt can happen at contact or
003273:                         # occlusion and must not erase that valid point.
003274:                         int(getattr(self, "_local_ai_recovery_count", 0)) == 0 and
003275:                         int(getattr(self, "_point_hit_count", 0)) == 0 and
003276:                         point_age <= 150):
003277:                     self._discard_provisional_serve_from_ai = True
003278:             else:
003279:                 self._local_ai_all_body_rejections = 0
003280:             print(f"[LOCAL_AI_RECOVERY] f{self.frame_count}: no safe path ({reason})")
003281:             # POST_SERVE_STATE_ROLLBACK_V3
003282:             # The reason detector runs after normal HSV tracking has already
003283:             # mutated ball_center/motion/history.  If the post-serve guard says
003284:             # that mutation is unsafe and AI cannot replace it, restore the
003285:             # complete pre-track snapshot immediately.  Returning the previous
003286:             # position makes the main loop log/record the held sane point rather
003287:             # than the rejected player pixel.
003288:             if (
003289:                     post_serve_recovery_active and
003290:                     pre_track_snapshot is not None and
003291:                     previous_position is not None and
003292:                     (
003293:                         reason.startswith("post-serve-launch-") or
003294:                         reason.startswith("player-region:")
003295:                     )):
003296:                 rejected_position = (
003297:                     tuple(tracked_position) if tracked_position is not None else None
003298:                 )
003299:                 self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
003300:                 self.stuck_frame_count = max(
003301:                     int(previous_stuck or 0) + 1,
003302:                     int(getattr(self, "stuck_frame_count", 0) or 0),
003303:                 )
003304:                 if getattr(self, "_last_motion_reacq_frame", -1000000) == self.frame_count:
003305:                     self._last_motion_reacq_frame = -1000000
003306:                     self._last_motion_reacq_pos = None
003307:                 print(
003308:                     f"[POST_SERVE_LAUNCH_ROLLBACK] f{self.frame_count}: "
003309:                     f"rejected={rejected_position} reason={reason}; "
003310:                     f"restored={previous_position} stuck={self.stuck_frame_count} "
003311:                     f"retry_through=f{int(getattr(self, '_post_serve_recovery_hold_until_frame', -1))}"
003312:                 )
003313:                 return tuple(previous_position)
003314:             # If HSV just teleported from a physically coherent ball path into a
003315:             # player region and local AI could not verify a replacement, never let
003316:             # that player pixel become the next-frame anchor. Restore the complete
003317:             # pre-track state (position, size, motion and velocity history) and let
003318:             # the rapid AI retry continue from the last sane trajectory.
003319:             if (
003320:                     reason.startswith("player-region:") and
003321:                     pre_track_snapshot is not None and
003322:                     previous_position is not None and
003323:                     tracked_position is not None):
```

## Lines 24697-24907
```python
024697:         if getattr(self, 'edge_wait', False):
024698:             return False, "Edge wait"
024699:         reference_override = self._reference_point_end_override()
024700:         if reference_override is not None:
024701:             return True, reference_override['reason']
024702:         # A reviewed endpoint means that every earlier terminal heuristic is
024703:         # known to be premature for this specific rally.  This must be here,
024704:         # before the geometric/net/bounce checks below, rather than only in
024705:         # the timeout path; otherwise a false "net" or "out" event can end
024706:         # the point before the reviewed frame is reached.
024707:         continuation_frame = self._reviewed_point_continuation_frame()
024708:         if continuation_frame is not None and self.frame_count < continuation_frame:
024709:             return False, "Reviewed continuation hold"
024710:         reference_target = self._reference_point_end_target()
024711:         if (reference_target is not None and
024712:                 self.frame_count < int(reference_target['point_end_frame'])):
024713:             return False, "Reviewed endpoint hold"
024714:         top_far_out, top_far_reason = self._top_far_baseline_fall_out_candidate(ball_position, frame)
024715:         if top_far_out:
024716:             return True, top_far_reason
024717:         if (self._back_return_wait_active() or
024718:                 getattr(self, '_back_return_reentry_grace_frames', 0) > 0 or
024719:                 self._recent_offscreen_return_hold_active(window_frames=24)):
024720:             return False, "Back-return wait"
024721:         if getattr(self, 'ground_bounce_count', 0) >= 2:
024722:             terminal_zone = self._player_point_zone(ball_position)
024723:             terminal_size = float(getattr(self, 'ball_size', 0.0) or 0.0)
024724:             # A second bounce is conclusive only when the current tracker
024725:             # sample still resembles a compact ball.  Large blended court or
024726:             # player blobs can inherit an earlier bounce count and otherwise
024727:             # turn into a premature point end.  Keep the count armed and wait
024728:             # for a compact follow-up, which preserves genuine double bounces.
024729:             compact_terminal_ball = (
024730:                 terminal_zone is None and
024731:                 terminal_size <= max(600.0, float(min(height, width)) * 0.22)
024732:             )
024733:             if compact_terminal_ball:
024734:                 return True, "Ball bounced twice on court"
024735:             print(
024736:                 f"Frame {self.frame_count}: [DOUBLE-BOUNCE SUPPRESSED] "
024737:                 f"pos={ball_position} size={terminal_size:.1f}px "
024738:                 f"zone={terminal_zone or 'clear'}"
024739:             )
024740:         
024741:         # Check if ball is out of court bounds
024742:         if x < 0 or x > width or y < 0 or y > height:
024743:             return True, "Ball out of court bounds"
024744: 
024745:         if self._upper_fence_fall_end_candidate(ball_position, frame.shape):
024746:             return True, "Ball hit upper fence and fell down"
024747: 
024748:         suppress_out_bounce = (
024749:             self._back_return_wait_active() or
024750:             getattr(self, '_back_return_reentry_grace_frames', 0) > 0
024751:         )
024752:         if not suppress_out_bounce:
024753:             out_bounce_detected, out_bounce_reason = self._detect_out_of_court_bounce(ball_position, frame)
024754:             if out_bounce_detected:
024755:                 from out_bounce_verification import recover_continuing_ball
024756:                 if recover_continuing_ball(self, ball_position):
024757:                     return False, "Out candidate replaced by continuing ball"
024758:                 return True, out_bounce_reason
024759:         
024760:         # Check if ball is in or just above the marked net area.
024761:         # A little extra top margin helps catch real tape clips when the marked
024762:         # net band sits slightly low in the frame.
024763:         net_x_limit = int(width * 0.60)
024764:         net_y_top_margin = 220
024765:         net_y_bottom_margin = 80
024766:         if hasattr(self, 'net_area_y_min') and hasattr(self, 'net_area_y_max'):
024767:             expanded_net_zone = (
024768:                 (self.net_area_y_min - net_y_top_margin) <= y <= (self.net_area_y_max + net_y_bottom_margin)
024769:                 and x < net_x_limit
024770:             )
024771:             core_net_zone = (
024772:                 max(0, self.net_area_y_min - 20) <= y <= (self.net_area_y_max + 40)
024773:                 and x < net_x_limit
024774:             )
024775:             if expanded_net_zone and not recent_ground_bounce:
024776:                 ball_is_fast = self.last_motion and self.last_motion['distance'] > 15
024777:                 # Only count "lingering" inside the tighter core net band.
024778:                 # A normal descending ball can legitimately pass through the larger
024779:                 # expanded zone for many frames without touching the net.
024780:                 if core_net_zone and not ball_is_fast:
024781:                     if not hasattr(self, 'net_area_frames'):
024782:                         self.net_area_frames = 0
024783:                     self.net_area_frames += 1
024784:                 else:
024785:                     self.net_area_frames = 0
024786:                 # Consider it a net hit if it lingers - use longer threshold for fast balls
024787:                 contact_ai_state = getattr(self, '_contact_local_ai_state', None)
024788:                 contact_ai_in_net_transit = (
024789:                     isinstance(contact_ai_state, dict) and
024790:                     self._net_contact_geometry(ball_position) is not None and
024791:                     self._net_contact_geometry(ball_position).get('near_hull', False)
024792:                 )
024793:                 # A Local-AI track can remain coherent while the ball travels
024794:                 # through the depth of the net.  Give that bounded trajectory
024795:                 # a few additional frames before treating its slow position as
024796:                 # a net stop; ordinary HSV net contacts keep the short limit.
024797:                 net_linger_limit = (
024798:                     14 if contact_ai_in_net_transit else
024799:                     (8 if ball_is_fast else 6)
024800:                 )
024801:                 if getattr(self, 'net_area_frames', 0) > net_linger_limit:
024802:                     return True, "Ball hit the net"
024803:                 # Immediate stop near net
024804:                 if self.last_motion and self.last_motion['distance'] < 2.5:
024805:                     # Only if previous frame was also slow (not a tracking glitch)
024806:                     prev_was_slow = self.prev_motion and self.prev_motion['distance'] < 10
024807:                     if prev_was_slow:
024808:                         return True, "Ball hit the net"
024809:                 # Direction change with deceleration — classic net hit signature
024810:                 if self.prev_motion and self.last_motion:
024811:                     prev_dir = self.prev_motion.get('direction_deg')
024812:                     curr_dir = self.last_motion.get('direction_deg')
024813:                     prev_dist = self.prev_motion.get('distance', 0)
024814:                     curr_dist = self.last_motion.get('distance', 0)
024815:                     prev_dx = self.prev_motion.get('dx', 0)
024816:                     prev_dy = self.prev_motion.get('dy', 0)
024817:                     curr_dx = self.last_motion.get('dx', 0)
024818:                     curr_dy = self.last_motion.get('dy', 0)
024819:                     if prev_dir is not None and curr_dir is not None and prev_dist > 0:
024820:                         delta = abs(curr_dir - prev_dir) % 360
024821:                         angle_diff = min(delta, 360 - delta)
024822:                         speed_ratio = curr_dist / prev_dist if prev_dist else 1.0
024823:                         reversed_y = (
024824:                             prev_dy * curr_dy < 0
024825:                             and abs(prev_dy) >= 3
024826:                             and abs(curr_dy) >= 3
024827:                         )
024828:                         strong_x_reversal = (
024829:                             prev_dx * curr_dx < 0
024830:                             and abs(prev_dx) >= 6
024831:                             and abs(curr_dx) >= 6
024832:                         )
024833:                         direction_reversed = reversed_y or strong_x_reversal
024834:                         slowed_after_impact = curr_dist <= max(8.0, prev_dist * 0.80)
024835:                         if direction_reversed and slowed_after_impact:
024836:                             return True, "Ball hit the net"
024837:                         if angle_diff >= 60 and slowed_after_impact:
024838:                             return True, "Ball hit the net"
024839:                         if angle_diff >= 90 and speed_ratio < 0.6:
024840:                             if prev_dist < 30:
024841:                                 return True, "Ball hit the net"
024842:             else:
024843:                 # Reset counter if ball is not in net area
024844:                 if hasattr(self, 'net_area_frames'):
024845:                     self.net_area_frames = 0
024846:         
024847:         # Check if ball has been stationary for too long (double bounce)
024848:         if hasattr(self, 'ball_velocity_history') and len(self.ball_velocity_history) > 10:
024849:             recent_velocities = self.ball_velocity_history[-10:]
024850:             avg_velocity = sum(recent_velocities) / len(recent_velocities)
024851:             print(f"[BALL_STOPPED_CHECK] f{self.frame_count}: avg_vel={avg_velocity:.1f} hist={[round(v,1) for v in recent_velocities]}")
024852:             if avg_velocity < 5:  # Very slow movement
024853:                 print(f"[BALL_STOPPED] f{self.frame_count}: avg_vel={avg_velocity:.1f} < 5 → POINT ENDS")
024854:                 return True, "Ball stopped (possible double bounce)"
024855:         
024856:         # Check if ball is near court edges (likely out)
024857:         edge_margin = 50
024858:         if (x < edge_margin or x > width - edge_margin or 
024859:             y < edge_margin or y > height - edge_margin):
024860:             if getattr(self, 'edge_wait', False) or getattr(self, 'near_edge', False):
024861:                 return False, "Edge return grace"
024862:             self.edge_wait = True
024863:             return False, "Edge return grace"
024864:         
024865:         return False, "Point continues"
024866: 
024867:     def detect_point_end(self, ball_position, frame):
024868:         """Detect if a point has ended based on ball position and behavior."""
024869:         height, width = frame.shape[:2]
024870:         x, y = ball_position
024871:         frames_since_ground_bounce = self.frame_count - getattr(self, 'last_ground_bounce_frame', -1000000)
024872:         recent_ground_bounce = frames_since_ground_bounce <= 2
024873: 
024874:         if self.frame_count <= (self.start_frame + 15):
024875:             return False, "Early-serve grace"
024876:         recovered_far_exit, recovered_far_reason = (
024877:             self._contact_visible_recovery_far_exit_end_candidate(
024878:                 ball_position, frame
024879:             )
024880:         )
024881:         if recovered_far_exit:
024882:             return True, recovered_far_reason
024883:         outbound_exit, outbound_exit_reason = self._confirmed_outbound_sideline_exit(
024884:             ball_position, frame
024885:         )
024886:         if outbound_exit:
024887:             return True, outbound_exit_reason
024888:         if getattr(self, 'edge_wait', False):
024889:             return False, "Edge wait"
024890:         # A near-player occlusion can leave a racket/body fragment frozen at an
024891:         # apparent sideline position.  Suppress every point-end classifier during
024892:         # the bounded reacquisition grace, including out-of-court checks below.
024893:         # A small, genuinely stationary ball on the court is different: it is a
024894:         # valid unreturned/double-bounce terminal and must not be held until the
024895:         # long player-reacquisition timeout.  Large blobs (body/racket fragments)
024896:         # stay protected by the normal guard.
024897:         reacq_protect_active = int(getattr(self, '_player_reacq_protect_until_frame', -1)) >= self.frame_count
024898:         stationary_ball_override = False
024899:         if reacq_protect_active:
024900:             ball_size_now = float(getattr(self, 'ball_size', 0.0) or 0.0)
024901:             stuck_now = int(getattr(self, 'stuck_frame_count', 0) or 0)
024902:             net_y = float(getattr(self, 'net_area_y_max', height * 0.42) or height * 0.42)
024903:             court_ball_zone = float(y) > net_y + max(120.0, height * 0.08)
024904:             small_ball_blob = 0.0 < ball_size_now <= max(180.0, width * 0.012)
024905:             slow_now = self.last_motion is None or float(self.last_motion.get('distance', 0.0) or 0.0) <= 4.5
024906:             slow_prev = self.prev_motion is None or float(self.prev_motion.get('distance', 0.0) or 0.0) <= 8.0
024907:             prior_play = (
```

## Lines 25007-25217
```python
025007:             self._back_return_wait_active() or
025008:             getattr(self, '_back_return_reentry_grace_frames', 0) > 0
025009:         )
025010:         if not suppress_out_bounce:
025011:             out_bounce_detected, out_bounce_reason = self._detect_out_of_court_bounce(ball_position, frame)
025012:             if out_bounce_detected:
025013:                 from out_bounce_verification import recover_continuing_ball
025014:                 if recover_continuing_ball(self, ball_position):
025015:                     return False, "Out candidate replaced by continuing ball"
025016:                 return True, out_bounce_reason
025017: 
025018:         if self._mark_serve_net_contact_candidate(ball_position, frame):
025019:             return False, "Serve net touch, waiting for service bounce"
025020: 
025021:         same_side_bounce, same_side_reason = self._same_side_pre_net_bounce_candidate(ball_position, frame)
025022:         if int(getattr(self, "_player_occlusion_hold_frames", 0)) > 0:
025023:             if same_side_bounce:
025024:                 print(
025025:                     f"Frame {self.frame_count}: [SAME-SIDE PRE-NET SUPPRESSED] "
025026:                     f"player-occlusion grace active ({self._player_occlusion_hold_frames} frames)"
025027:                 )
025028:             same_side_bounce, same_side_reason = False, None
025029:         if same_side_bounce:
025030:             return True, same_side_reason
025031: 
025032:         recent_serve_bounce_in = (
025033:             getattr(self, '_last_serve_bounce_frame', -1000000) > -1000000 and
025034:             0 <= self.frame_count - getattr(self, '_last_serve_bounce_frame', -1000000) <= 10 and
025035:             not getattr(self, '_awaiting_serve_bounce', False)
025036:         )
025037:         net_geometry = self._net_contact_geometry(ball_position)
025038: 
025039:         far_side_serve_attempt = getattr(self, '_active_serve_area_end', None) == "far"
025040:         if (far_side_serve_attempt and recent_serve_bounce_in and
025041:                 getattr(self, '_last_serve_bounce_net_contact_like', False)):
025042:             net_zone_active = (
025043:                 net_geometry is not None and
025044:                 (net_geometry['inside'] or net_geometry['near_polygon'] or net_geometry['near_top_tape'])
025045:             )
025046:             current_speed = float(self.last_motion.get('distance', 0.0)) if self.last_motion else 0.0
025047:             if net_zone_active and current_speed <= 14.0:
025048:                 return True, "Serve let"
025049: 
025050:         if net_geometry is not None and recent_ground_bounce and not recent_serve_bounce_in:
025051:             bounce_origin = getattr(self, '_ground_bounce_origin', None)
025052:             if bounce_origin is not None and 1 <= frames_since_ground_bounce <= 2:
025053:                 origin_x, origin_y = bounce_origin
025054:                 current_dy = float(self.last_motion.get('dy', 0.0) or 0.0) if self.last_motion else 0.0
025055:                 current_speed = float(self.last_motion.get('distance', 0.0) or 0.0) if self.last_motion else 0.0
025056:                 net_y_max = float(getattr(self, 'net_area_y_max', 0.0))
025057:                 low_net_y_min = float(getattr(self, 'low_net_y_min', getattr(self, 'net_area_y_min', 0)))
025058:                 low_net_y_max = float(getattr(self, 'low_net_y_max', getattr(self, 'net_area_y_max', 0)))
025059:                 origin_below_net = (
025060:                     net_y_max + 10.0 <= float(origin_y) <= net_y_max + max(65.0, height * 0.030)
025061:                 )
025062:                 climbing_into_low_net = (
025063:                     current_dy <= -12.0 and
025064:                     current_speed <= max(70.0, height * 0.035) and
025065:                     float(y) <= float(origin_y) - 45.0 and
025066:                     low_net_y_min - 8.0 <= float(y) <= low_net_y_max + 8.0
025067:                 )
025068:                 near_net_surface = (
025069:                     net_geometry['inside'] or
025070:                     net_geometry['near_top_tape'] or
025071:                     net_geometry['signed_dist'] >= -42.0
025072:                 )
025073:                 close_to_bounce_lane = (
025074:                     abs(float(x) - float(origin_x)) <= 90.0 and
025075:                     math.hypot(float(x) - float(origin_x), float(y) - float(origin_y)) <= 115.0
025076:                 )
025077:                 if origin_below_net and climbing_into_low_net and near_net_surface and close_to_bounce_lane:
025078:                     print(
025079:                         f"Frame {self.frame_count}: [POST-BOUNCE NET] "
025080:                         f"origin={bounce_origin} pos={ball_position} "
025081:                         f"signed_dist={net_geometry['signed_dist']:.1f}"
025082:                     )
025083:                     return True, "Ball hit the net"
025084: 
025085:         if net_geometry is not None and not recent_ground_bounce and not recent_serve_bounce_in:
025086:             net_zone_active = net_geometry['near_polygon'] or net_geometry['near_top_tape']
025087:             core_net_zone = net_geometry['inside']
025088:             if net_zone_active:
025089:                 ball_is_fast = self.last_motion and self.last_motion['distance'] > 15
025090:                 if core_net_zone and not ball_is_fast:
025091:                     if not hasattr(self, 'net_area_frames'):
025092:                         self.net_area_frames = 0
025093:                     self.net_area_frames += 1
025094:                 else:
025095:                     self.net_area_frames = 0
025096: 
025097:                 contact_ai_state = getattr(self, '_contact_local_ai_state', None)
025098:                 contact_ai_in_net_transit = (
025099:                     isinstance(contact_ai_state, dict) and
025100:                     self._net_contact_geometry(ball_position) is not None and
025101:                     self._net_contact_geometry(ball_position).get('near_hull', False)
025102:                 )
025103:                 net_linger_limit = (
025104:                     14 if contact_ai_in_net_transit else
025105:                     (8 if ball_is_fast else 6)
025106:                 )
025107:                 if getattr(self, 'net_area_frames', 0) > net_linger_limit:
025108:                     return True, "Ball hit the net"
025109: 
025110:                 if self.last_motion and self.last_motion['distance'] < 2.5:
025111:                     prev_was_slow = self.prev_motion and self.prev_motion['distance'] < 10
025112:                     if prev_was_slow and core_net_zone:
025113:                         return True, "Ball hit the net"
025114: 
025115:                 if self.prev_motion and self.last_motion:
025116:                     prev_dir = self.prev_motion.get('direction_deg')
025117:                     curr_dir = self.last_motion.get('direction_deg')
025118:                     prev_dist = self.prev_motion.get('distance', 0)
025119:                     curr_dist = self.last_motion.get('distance', 0)
025120:                     prev_dx = self.prev_motion.get('dx', 0)
025121:                     prev_dy = self.prev_motion.get('dy', 0)
025122:                     curr_dx = self.last_motion.get('dx', 0)
025123:                     curr_dy = self.last_motion.get('dy', 0)
025124:                     if prev_dir is not None and curr_dir is not None and prev_dist > 0:
025125:                         delta = abs(curr_dir - prev_dir) % 360
025126:                         angle_diff = min(delta, 360 - delta)
025127:                         speed_ratio = curr_dist / prev_dist if prev_dist else 1.0
025128:                         reversed_y = (
025129:                             prev_dy * curr_dy < 0
025130:                             and abs(prev_dy) >= 3
025131:                             and abs(curr_dy) >= 3
025132:                         )
025133:                         strong_x_reversal = (
025134:                             prev_dx * curr_dx < 0
025135:                             and abs(prev_dx) >= 6
025136:                             and abs(curr_dx) >= 6
025137:                         )
025138:                         direction_reversed = reversed_y or strong_x_reversal
025139:                         slowed_after_impact = curr_dist <= max(8.0, prev_dist * 0.80)
025140:                         direct_contact_zone = core_net_zone or net_geometry['near_top_tape']
025141:                         last_change_point = getattr(self, '_last_direction_change_point', None)
025142:                         last_change_geometry = (
025143:                             self._net_contact_geometry(last_change_point)
025144:                             if last_change_point is not None else None
025145:                         )
025146:                         recent_net_turn = (
025147:                             getattr(self, '_last_direction_change_frame', -1000000) == self.frame_count and
025148:                             last_change_geometry is not None and
025149:                             (last_change_geometry['inside'] or last_change_geometry['near_top_tape']) and
025150:                             curr_dy >= 4.0 and
025151:                             direct_contact_zone
025152:                         )
025153: 
025154:                         if recent_net_turn:
025155:                             return True, "Ball hit the net"
025156:                         if direct_contact_zone and direction_reversed and slowed_after_impact:
025157:                             return True, "Ball hit the net"
025158:                         if direct_contact_zone and angle_diff >= 60 and slowed_after_impact:
025159:                             return True, "Ball hit the net"
025160:                         if direct_contact_zone and angle_diff >= 90 and speed_ratio < 0.6 and prev_dist < 30:
025161:                             return True, "Ball hit the net"
025162:             else:
025163:                 if hasattr(self, 'net_area_frames'):
025164:                     self.net_area_frames = 0
025165:         elif hasattr(self, 'net_area_frames'):
025166:             self.net_area_frames = 0
025167: 
025168:         if hasattr(self, 'ball_velocity_history') and len(self.ball_velocity_history) > 10:
025169:             recent_velocities = self.ball_velocity_history[-10:]
025170:             avg_velocity = sum(recent_velocities) / len(recent_velocities)
025171:             print(f"[BALL_STOPPED_CHECK] f{self.frame_count}: avg_vel={avg_velocity:.1f} hist={[round(v,1) for v in recent_velocities]}")
025172:             if avg_velocity < 5:
025173:                 print(f"[BALL_STOPPED] f{self.frame_count}: avg_vel={avg_velocity:.1f} < 5 -> POINT ENDS")
025174:                 return True, "Ball stopped (possible double bounce)"
025175: 
025176:         edge_margin = 50
025177:         if (x < edge_margin or x > width - edge_margin or
025178:             y < edge_margin or y > height - edge_margin):
025179:             if getattr(self, 'edge_wait', False) or getattr(self, 'near_edge', False):
025180:                 return False, "Edge return grace"
025181:             self.edge_wait = True
025182:             return False, "Edge return grace"
025183: 
025184:         return False, "Point continues"
025185: 
025186:     def _build_singles_sideline_model(self, frame):
025187:         frame_shape = frame.shape[:2]
025188:         if (self._singles_sideline_model is not None and
025189:                 self._singles_sideline_frame_shape == frame_shape):
025190:             return self._singles_sideline_model
025191: 
025192:         height, width = frame_shape
025193:         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
025194:         _, white_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
025195:         min_line_length = max(250, int(width * 0.08))
025196:         lines = cv2.HoughLinesP(
025197:             white_mask,
025198:             1,
025199:             np.pi / 180,
025200:             threshold=120,
025201:             minLineLength=min_line_length,
025202:             maxLineGap=20,
025203:         )
025204:         if lines is None:
025205:             self._singles_sideline_model = None
025206:             self._singles_sideline_frame_shape = frame_shape
025207:             return None
025208: 
025209:         y_ref = int(height * 0.74)
025210:         left_candidates = []
025211:         right_candidates = []
025212:         fallback_left = []
025213:         fallback_right = []
025214: 
025215:         for raw in lines[:, 0, :]:
025216:             x1, y1, x2, y2 = [float(v) for v in raw]
025217:             dy = y2 - y1
```

## Lines 28368-28772
```python
028368:                             )
028369:                         else:
028370:                             print(
028371:                                 f"Frame {self.frame_count}: delaying point timeout while ball is still actively tracked "
028372:                                 f"(seen {frames_since_seen}f ago, motion={last_motion_distance:.1f}px)"
028373:                             )
028374:                     else:
028375:                         if self._ignore_unresolved_timeout_if_game_decider(
028376:                             "POINT_TIMEOUT", end_position=self.ball_center, frame=frame
028377:                         ):
028378:                             print(
028379:                                 f"[POINT_IGNORED] f{self.frame_count}: reason=POINT_TIMEOUT "
028380:                                 f"duration={dur}f would decide game without a located ball"
028381:                             )
028382:                         else:
028383:                             timeout_reason = "POINT_TIMEOUT"
028384:                             if self._in_court_timeout_landing_outcome(self.ball_center, frame) is not None:
028385:                                 timeout_reason = "Ball stopped on player side"
028386:                             # A duration timeout is only a fallback inference.  The
028387:                             # terminal verifier can still see a compact moving ball
028388:                             # in the just-decoded frame pair; continue the rally
028389:                             # rather than scoring a false timeout in that case.
028390:                             if self._resume_from_terminal_motion_candidate(
028391:                                 timeout_reason, self.ball_center, frame=frame
028392:                             ):
028393:                                 continue
028394:                             print(
028395:                                 f"[POINT_END] f{self.frame_count}: reason={timeout_reason} "
028396:                                 f"duration={dur}f — returning to serve detection"
028397:                             )
028398:                             self._record_point_result(timeout_reason, end_position=self.ball_center, frame=frame)
028399:                         game_state = "WAITING_FOR_SERVE"
028400:                         reset_tracking_state(hold_end_marker=True)
028401: 
028402:                 # Track ball through the point
028403:                 tracked_position = None
028404:                 if self.tracking and self.hsv_lower is not None and self.ball_center is not None:
028405:                     prev_ball_center = self.ball_center
028406:                     prev_stuck = self.stuck_frame_count
028407:                     prev_top_return_wait = self._top_return_wait_active()
028408:                     prev_back_return_wait = self._back_return_wait_active()
028409:                     pre_track_snapshot = self._snapshot_tracking_state_for_provisional_guard()
028410:                     self._local_ai_tight_roi_previous_gray = (
028411:                         pre_track_snapshot.get('_prev_frame_gray')
028412:                     )
028413:                     forced_local_ai = False
028414:                     contact_local_ai = False
028415:                     # V36: protect an explicitly verified NIGHT LOWER CONTACT LAUNCH
028416:                     # from later same-frame post-track recovery arbitration.
028417:                     verified_lower_contact_launch = False
028418:                     tracked_position = self._force_local_ai_frame(frame, prev_ball_center)
028419:                     if tracked_position is not None:
028420:                         forced_local_ai = True
028421:                     else:
028422:                         # Automatic AI assists only after normal tracking fails its evidence checks.
028423:                         tracked_position = self.track_ball_in_frame(frame)
028424:                         if self._post_contact_handoff_player_jump_invalid(
028425:                                 prev_ball_center,
028426:                                 tracked_position,
028427:                                 pre_track_snapshot):
028428:                             rejected_zone = self._player_point_zone(
028429:                                 tracked_position
028430:                             )
028431:                             rejected_position = tuple(tracked_position)
028432:                             self._restore_tracking_state_for_provisional_guard(
028433:                                 pre_track_snapshot
028434:                             )
028435:                             tracked_position = prev_ball_center
028436:                             self.stuck_frame_count = max(
028437:                                 int(prev_stuck) + 1,
028438:                                 int(getattr(self, 'stuck_frame_count', 0)),
028439:                             )
028440:                             print(
028441:                                 f"[POST-CONTACT HANDOFF PLAYER-JUMP REJECT] "
028442:                                 f"f{self.frame_count}: {tuple(prev_ball_center)} -> "
028443:                                 f"{rejected_position} zone={rejected_zone}; "
028444:                                 "keeping outgoing trajectory anchor"
028445:                             )
028446:                         verified_lower_contact_launch = (
028447:                             tracked_position is not None and
028448:                             int(getattr(
028449:                                 self,
028450:                                 '_last_verified_lower_contact_launch_frame',
028451:                                 -1000000,
028452:                             )) == int(self.frame_count)
028453:                         )
028454:                         if self._visible_contact_reversal_releases_contact_ai(
028455:                                 prev_ball_center,
028456:                                 tracked_position,
028457:                                 pre_track_snapshot):
028458:                             self._contact_local_ai_state = None
028459:                             self._contact_local_ai_cooldown_until_frame = max(
028460:                                 int(getattr(self, '_contact_local_ai_cooldown_until_frame', -1000000)),
028461:                                 int(self.frame_count) + 2,
028462:                             )
028463:                             # The normal candidate has already proved the
028464:                             # launch.  Discard the older lower-contact
028465:                             # prediction, which was based on the incoming
028466:                             # trajectory and can otherwise select a
028467:                             # player fragment on the next frame.
028468:                             self._pending_lower_contact_launch_context = None
028469:                             self._pending_lower_contact_launch_until_frame = -1000000
028470:                             self._rally_contact_grace_frames = max(
028471:                                 int(getattr(self, '_rally_contact_grace_frames', 0)), 4
028472:                             )
028473:                             self._rally_contact_origin = prev_ball_center
028474:                             self._rally_contact_expected = tracked_position
028475:                             self._contact_normal_fallback_path = []
028476:                             contact_reason = None
028477:                             print(
028478:                                 f"[CONTACT_LOCAL_AI VISIBLE-REVERSAL RELEASE] "
028479:                                 f"f{self.frame_count}: preserving normal outbound="
028480:                                 f"{tuple(tracked_position)}"
028481:                             )
028482:                         elif verified_lower_contact_launch:
028483:                             # A dedicated NIGHT LOWER CONTACT LAUNCH has already
028484:                             # committed a geometrically verified outbound ball.
028485:                             # Do not let a Contact Local-AI miss restore the stale
028486:                             # pre-contact anchor on this exact frame.
028487:                             self._contact_local_ai_state = None
028488:                             self._contact_local_ai_cooldown_until_frame = max(
028489:                                 int(getattr(
028490:                                     self,
028491:                                     '_contact_local_ai_cooldown_until_frame',
028492:                                     -1000000,
028493:                                 )),
028494:                                 int(self.frame_count) + 2,
028495:                             )
028496:                             self._contact_normal_fallback_path = []
028497:                             contact_reason = None
028498:                             print(
028499:                                 f"[CONTACT_LOCAL_AI VERIFIED-LOWER-LAUNCH RELEASE] "
028500:                                 f"f{self.frame_count}: preserving normal launch="
028501:                                 f"{tuple(tracked_position)}"
028502:                             )
028503:                         elif self._strong_visible_recovery_releases_contact_ai(
028504:                                 prev_ball_center,
028505:                                 tracked_position,
028506:                                 pre_track_snapshot):
028507:                             contact_reason = None
028508:                         elif self._current_terminal_ground_bounce_pending():
028509:                             # ``track_ball_in_frame`` has already confirmed
028510:                             # a physical bounce and queued its terminal
028511:                             # result.  Do not restore the pre-track state
028512:                             # merely because a stale Contact Local-AI owner
028513:                             # sees a player-like alternative nearby.
028514:                             self._contact_local_ai_state = None
028515:                             self._contact_local_ai_cooldown_until_frame = max(
028516:                                 int(getattr(
028517:                                     self,
028518:                                     '_contact_local_ai_cooldown_until_frame',
028519:                                     -1000000,
028520:                                 )),
028521:                                 int(self.frame_count) + 2,
028522:                             )
028523:                             self._contact_normal_fallback_path = []
028524:                             contact_reason = None
028525:                             print(
028526:                                 f"[CONTACT_LOCAL_AI TERMINAL-BOUNCE RELEASE] "
028527:                                 f"f{self.frame_count}: preserving "
028528:                                 f"{self._pending_rally_end_reason}"
028529:                             )
028530:                         else:
028531:                             contact_reason = self._contact_local_ai_trigger(
028532:                                     prev_ball_center, tracked_position, pre_track_snapshot, frame=frame
028533:                             )
028534:                         if contact_reason is not None:
028535:                             rejected_hsv = (
028536:                                 tuple(tracked_position)
028537:                                 if tracked_position is not None else None
028538:                             )
028539:                             post_track_snapshot = self._snapshot_tracking_state_for_provisional_guard()
028540:                             self._restore_tracking_state_for_provisional_guard(
028541:                                 pre_track_snapshot
028542:                             )
028543:                             self._local_ai_tight_roi_previous_gray = (
028544:                                 pre_track_snapshot.get('_prev_frame_gray')
028545:                             )
028546:                             # Debug only: preserve the normal detector's contact
028547:                             # hypothesis long enough to print candidate agreement.
028548:                             self._contact_local_ai_debug_normal_candidate = rejected_hsv
028549:                             tracked_position = self._run_contact_local_ai_frame(
028550:                                 frame, prev_ball_center, arm_reason=contact_reason
028551:                             )
028552:                             self._contact_local_ai_debug_normal_candidate = None
028553:                             if tracked_position is not None:
028554:                                 contact_local_ai = True
028555: 
028556:                                 # V35_HOTSPOT_ONLY_CONTACT_HANDOFF
028557:                                 # Preserve the pre-handoff motion only for the
028558:                                 # trajectory-hotspot takeover that caused the
028559:                                 # false OUT reversal. Normal contact-stall
028560:                                 # and contact-jump replacements must retain the
028561:                                 # proven V32 behavior and commit the AI step.
028562:                                 if str(contact_reason or '').startswith(
028563:                                     'trajectory-hotspot-turn:'
028564:                                 ):
028565:                                     # V34: this exact frame is a detector handoff, not a
028566:                                     # second physical ball observation.  The normal HSV
028567:                                     # candidate was already rolled back above; Local AI
028568:                                     # may legitimately choose a different anchor around
028569:                                     # racket contact.  Keep that AI anchor, but do not let
028570:                                     # the anchor displacement become last_motion and fake
028571:                                     # a ground-bounce reversal on the same frame.
028572:                                     ai_anchor = tuple(tracked_position)
028573:                                     ai_ball_size = self.ball_size
028574:                                     ai_ball_hsv = self.ball_hsv
028575:                                     if pre_track_snapshot is not None:
028576:                                         if 'prev_motion' in pre_track_snapshot:
028577:                                             prior_prev_motion = pre_track_snapshot.get('prev_motion')
028578:                                             self.prev_motion = (
028579:                                                 dict(prior_prev_motion)
028580:                                                 if isinstance(prior_prev_motion, dict)
028581:                                                 else prior_prev_motion
028582:                                             )
028583:                                         if 'last_motion' in pre_track_snapshot:
028584:                                             prior_last_motion = pre_track_snapshot.get('last_motion')
028585:                                             self.last_motion = (
028586:                                                 dict(prior_last_motion)
028587:                                                 if isinstance(prior_last_motion, dict)
028588:                                                 else prior_last_motion
028589:                                             )
028590:                                         if 'last_nonzero_motion' in pre_track_snapshot:
028591:                                             prior_nonzero = pre_track_snapshot.get('last_nonzero_motion')
028592:                                             self.last_nonzero_motion = (
028593:                                                 dict(prior_nonzero)
028594:                                                 if isinstance(prior_nonzero, dict)
028595:                                                 else prior_nonzero
028596:                                             )
028597:                                         if 'last_delta' in pre_track_snapshot:
028598:                                             self.last_delta = pre_track_snapshot.get('last_delta')
028599:                                         if 'last_direction' in pre_track_snapshot:
028600:                                             self.last_direction = pre_track_snapshot.get('last_direction')
028601:                                         if 'ball_velocity_history' in pre_track_snapshot:
028602:                                             self.ball_velocity_history = list(
028603:                                                 pre_track_snapshot.get('ball_velocity_history') or []
028604:                                             )[-5:]
028605:                                         if 'direction_change_streak' in pre_track_snapshot:
028606:                                             self.direction_change_streak = int(
028607:                                                 pre_track_snapshot.get('direction_change_streak') or 0
028608:                                             )
028609: 
028610:                                     # Re-assert the accepted AI observation after the
028611:                                     # motion-state repair.  Contact Local-AI keeps its own
028612:                                     # validated history, so the next frame can still follow
028613:                                     # the outgoing trajectory from this anchor.
028614:                                     self.ball_center = ai_anchor
028615:                                     self.ball_size = ai_ball_size
028616:                                     self.ball_hsv = ai_ball_hsv
028617:                                     self.last_seen_frame = self.frame_count
028618:                                     self.stuck_frame_count = 0
028619:                                     self._pending_rally_end_reason = None
028620:                                     self._pending_rally_end_frame = -1
028621:                                     if frame is not None:
028622:                                         try:
028623:                                             self._prev_frame_gray = cv2.cvtColor(
028624:                                                 frame, cv2.COLOR_BGR2GRAY
028625:                                             )
028626:                                         except cv2.error:
028627:                                             pass
028628:                                     self._contact_local_ai_anchor_handoff_frame = int(
028629:                                         self.frame_count
028630:                                     )
028631:                                     prior_motion = self.last_motion or {}
028632:                                     print(
028633:                                         f"[CONTACT_LOCAL_AI_ANCHOR_HANDOFF] f{self.frame_count}: "
028634:                                         f"anchor={ai_anchor} preserving prior motion="
028635:                                         f"({float(prior_motion.get('dx', 0.0) or 0.0):.1f},"
028636:                                         f"{float(prior_motion.get('dy', 0.0) or 0.0):.1f}) "
028637:                                         f"vel_hist={[round(float(v), 1) for v in self.ball_velocity_history[-5:]]}"
028638:                                     )
028639:                                 print(
028640:                                     f"[CONTACT_LOCAL_AI_REPLACE] f{self.frame_count}: "
028641:                                     f"HSV={rejected_hsv} -> AI={tracked_position}"
028642:                                 )
028643:                             else:
028644:                                 # AI is an assist: if it has no accepted candidate,
028645:                                 # keep the normal tracker result for this frame.
028646:                                 tracked_position = self._normal_fallback_after_ai_miss(
028647:                                     prev_ball_center, rejected_hsv,
028648:                                     pre_track_snapshot, post_track_snapshot,
028649:                                 )
028650:                                 contact_local_ai = False
028651:                                 print(
028652:                                     f"[CONTACT_LOCAL_AI_FALLBACK] f{self.frame_count}: "
028653:                                     f"AI miss; preserving trusted position={tracked_position}"
028654:                                 )
028655:                     if self.local_ai_recovery is not None and self._local_ai_frame_buffer:
028656:                         self._local_ai_frame_buffer[-1]["normal_position"] = (
028657:                             tuple(tracked_position) if tracked_position is not None else None
028658:                         )
028659:                     self._debug_local_ai_shadow_frame(frame, prev_ball_center, tracked_position)
028660:                     # V37: once NIGHT LOWER CONTACT LAUNCH has committed a real
028661:                     # outbound ball, do not let the generic post-track Local-AI hold
028662:                     # roll that anchor back on the launch frame or on a physically
028663:                     # coherent immediate continuation.  This is deliberately narrow:
028664:                     # only the next two frames are eligible, the motion must still be
028665:                     # upward/netward, and it must agree with the previously committed
028666:                     # motion vector.  A player/body jump or reversal therefore remains
028667:                     # eligible for normal Local-AI arbitration.
028668:                     lower_launch_frame = int(getattr(
028669:                         self, '_last_verified_lower_contact_launch_frame', -1000000
028670:                     ))
028671:                     lower_launch_age = int(self.frame_count) - lower_launch_frame
028672:                     verified_lower_contact_continuation = False
028673:                     continuation_step = 0.0
028674:                     continuation_cos = -1.0
028675:                     if (
028676:                         tracked_position is not None and
028677:                         prev_ball_center is not None and
028678:                         pre_track_snapshot is not None and
028679:                         1 <= lower_launch_age <= 2
028680:                     ):
028681:                         continuation_dx = (
028682:                             float(tracked_position[0]) - float(prev_ball_center[0])
028683:                         )
028684:                         continuation_dy = (
028685:                             float(tracked_position[1]) - float(prev_ball_center[1])
028686:                         )
028687:                         continuation_step = math.hypot(
028688:                             continuation_dx, continuation_dy
028689:                         )
028690:                         prior_motion = pre_track_snapshot.get('last_motion') or {}
028691:                         prior_dx = float(prior_motion.get('dx', 0.0) or 0.0)
028692:                         prior_dy = float(prior_motion.get('dy', 0.0) or 0.0)
028693:                         prior_step = float(
028694:                             prior_motion.get('distance', 0.0) or
028695:                             math.hypot(prior_dx, prior_dy)
028696:                         )
028697:                         if continuation_step > 0.0 and prior_step > 0.0:
028698:                             continuation_cos = (
028699:                                 continuation_dx * prior_dx +
028700:                                 continuation_dy * prior_dy
028701:                             ) / (continuation_step * prior_step)
028702:                         continuation_limit = max(
028703:                             180.0,
028704:                             min(320.0, prior_step * 2.2 + 60.0),
028705:                         )
028706:                         verified_lower_contact_continuation = (
028707:                             12.0 <= continuation_step <= continuation_limit and
028708:                             continuation_dy <= -12.0 and
028709:                             prior_dy <= -8.0 and
028710:                             continuation_cos >= 0.25
028711:                         )
028712: 
028713:                     verified_lower_contact_post_guard = (
028714:                         tracked_position is not None and
028715:                         (
028716:                             lower_launch_age == 0 or
028717:                             verified_lower_contact_launch or
028718:                             verified_lower_contact_continuation
028719:                         )
028720:                     )
028721:                     if verified_lower_contact_post_guard:
028722:                         guard_kind = (
028723:                             'launch'
028724:                             if lower_launch_age == 0 or verified_lower_contact_launch
028725:                             else 'continuation'
028726:                         )
028727:                         print(
028728:                             f"[VERIFIED-LOWER-CONTACT POST-RECOVERY SKIP] "
028729:                             f"f{self.frame_count}: kind={guard_kind} age={lower_launch_age} "
028730:                             f"keeping={tuple(tracked_position)} step={continuation_step:.1f}px "
028731:                             f"cos={continuation_cos:.3f}"
028732:                         )
028733:                     if (
028734:                         not forced_local_ai and
028735:                         not contact_local_ai and
028736:                         not verified_lower_contact_post_guard
028737:                     ):
028738:                         tracked_position = self._try_local_ai_recovery(
028739:                             prev_ball_center, tracked_position, prev_stuck,
028740:                             pre_track_snapshot=pre_track_snapshot,
028741:                             frame=frame,
028742:                         )
028743:                     self._local_ai_tight_roi_previous_gray = None
028744:                     # Reject any position that jumps impossibly far in one frame (false positive).
028745:                     # When the tracker is in re-acquisition mode (stuck >= 5 before the call), allow
028746:                     # a larger jump because the ball may have traveled far while lost.
028747:                     if tracked_position and prev_ball_center:
028748:                         jump = math.hypot(tracked_position[0] - prev_ball_center[0],
028749:                                           tracked_position[1] - prev_ball_center[1])
028750:                         motion_reacq_this_frame = (
028751:                             getattr(self, '_last_motion_reacq_frame', -1000000) == self.frame_count and
028752:                             getattr(self, '_last_motion_reacq_pos', None) == tracked_position
028753:                         )
028754:                         trusted_rally_launch = (
028755:                             getattr(self, '_trusted_rally_launch_frame', -1000000) == self.frame_count and
028756:                             getattr(self, '_trusted_rally_launch_pos', None) == tracked_position
028757:                         )
028758:                         # Allow a larger jump when re-acquiring after being stuck for 5+ frames:
028759:                         # a racket hit can send the ball 800+ px in one frame, so we use 1500px
028760:                         # to let motion-based re-acquisition recover across the full court.
028761:                         max_jump = 1500 if (
028762:                             prev_stuck >= 5 or prev_top_return_wait or prev_back_return_wait or
028763:                             motion_reacq_this_frame or trusted_rally_launch
028764:                         ) else 400
028765:                         if (getattr(self, '_temporal_top_reentry_frame', -1) == self.frame_count and
028766:                                 getattr(self, '_temporal_top_reentry_pos', None) == tracked_position):
028767:                             # The gap spans an off-screen flight; its speed was
028768:                             # independently measured from three visible samples.
028769:                             max_jump = math.hypot(frame.shape[1], frame.shape[0])
028770:                         if jump > max_jump:
028771:                             print(f"[JUMP_REJECTED] f{self.frame_count}: jumped {jump:.0f}px from {prev_ball_center} to {tracked_position} (limit={max_jump}px, prev_stuck={prev_stuck}), keeping previous")
028772:                             if ((prev_top_return_wait and self._delayed_literal_top_return_context()) or
```
