# V32 Contact Local-AI source context

## Lines 215-425
```python
000215:             self._buffer = ""
000216:         self._target.flush()
000217: 
000218: 
000219: class InteractiveBallAnalyzer:
000220:     def __init__(
000221:         self,
000222:         video_path: str,
000223:         start_frame: int = 0,
000224:         config_file: str = "hsv_config.json",
000225:         headless: bool = False,
000226:         disable_false_points: bool = False,
000227:         point_history_file: str = "point_history.csv",
000228:         write_point_history: bool = True,
000229:         start_server_side: Optional[str] = None,
000230:         start_score: Optional[Tuple[int, int, int, int]] = None,
000231:         enable_player_tracking: bool = True,
000232:         player_tracking_interval: int = 5,
000233:         enable_player_learning: bool = True,
000234:         enable_player_ball_protection: bool = True,
000235:         ball_dataset_dir: Optional[str] = None,
000236:         local_ai_model: Optional[str] = None,
000237:         local_ai_python: Optional[str] = None,
000238:         local_ai_recovery_dir: Optional[str] = None,
000239:         debug_local_ai_range: Optional[Tuple[int, int]] = None,
000240:         debug_local_ai_radius: float = 140.0,
000241:     ):
000242:         self.video_path = video_path
000243:         self.config_file = config_file
000244:         self.headless = headless
000245:         self.disable_false_points = disable_false_points
000246:         self.point_history_file = point_history_file
000247:         self.write_point_history = write_point_history
000248:         self.ball_dataset_dir = ball_dataset_dir
000249:         self.local_ai_model = local_ai_model
000250:         self.player_tracking_enabled = bool(enable_player_tracking) and os.environ.get(
000251:             "DISABLE_PLAYER_TRACKING", "0"
000252:         ) != "1"
000253:         self.player_learning_enabled = bool(enable_player_learning) and os.environ.get(
000254:             "DISABLE_PLAYER_LEARNING", "0"
000255:         ) != "1"
000256:         self.player_ball_protection_enabled = bool(enable_player_ball_protection) and os.environ.get(
000257:             "DISABLE_PLAYER_BALL_PROTECTION", "0"
000258:         ) != "1"
000259:         self.player_tracking_interval = max(1, int(player_tracking_interval))
000260:         profile_stem = os.path.splitext(os.path.basename(config_file or "hsv_config.json"))[0]
000261:         profile_dir = os.path.dirname(os.path.abspath(config_file or "hsv_config.json"))
000262:         self.player_tracking_profile_path = os.path.join(
000263:             profile_dir, f"player_tracking_{profile_stem}.json"
000264:         )
000265:         self.player_tracker = PlayerRacketTracker(
000266:             detection_interval=self.player_tracking_interval,
000267:             profile_path=self.player_tracking_profile_path,
000268:             learning_enabled=self.player_learning_enabled,
000269:         ) if self.player_tracking_enabled else None
000270:         if start_server_side not in (None, "near", "far"):
000271:             raise ValueError("start_server_side must be 'near', 'far', or None")
000272:         self.start_server_side = start_server_side
000273:         self._point_history_initialized = False
000274:         self._point_history_point_index = 0
000275:         self._point_history_current = None
000276:         self._point_audit_traces = {}
000277:         self._process_stop_reason = None
000278:         self.cap = cv2.VideoCapture(video_path)
000279:         self.start_frame = max(0, start_frame)
000280:         self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
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

## Lines 1366-1771
```python
001366:                 f'angle={angle_delta:.0f}deg/pred={prediction_error:.0f}px'
001367:             )
001368: 
001369:         contact_jump_candidate = (
001370:             (contact_near or watch_active) and proposed_speed >= jump_floor and
001371:             (angle_delta >= 35.0 or prediction_error >= 50.0)
001372:         )
001373: 
001374:         # V17: Contact Local-AI is an assist, not a permanent veto.
001375:         #
001376:         # During the short serve-contact/pre-bounce window, the normal HSV
001377:         # tracker can already see the outgoing ball while the local model still
001378:         # misses a blurred/contact-adjacent patch. The process loop deliberately
001379:         # rolls that normal candidate back when Contact Local-AI misses. Keep
001380:         # that protection for one-off racket/body fragments, but remember the
001381:         # rejected normal proposals. Three consecutive netward proposals with
001382:         # a coherent direction prove an independent causal ball path and let the
001383:         # normal tracker keep the third frame.
001384:         serve_contact_path_window = (
001385:             contact_jump_candidate and
001386:             tracked_position is not None and
001387:             bool(getattr(self, '_awaiting_serve_bounce', False)) and
001388:             int(getattr(self, '_serve_contact_grace_frames', 0) or 0) > 0 and
001389:             int(getattr(self, 'ground_bounce_count', 0) or 0) == 0 and
001390:             int(getattr(self, 'serve_direction_dy', 0) or 0) != 0
001391:         )
001392:         contact_normal_path = list(
001393:             getattr(self, '_contact_normal_fallback_path', []) or []
001394:         )
001395: 
001396:         if serve_contact_path_window:
001397:             current_sample = {
001398:                 'frame': int(self.frame_count),
001399:                 'pos': tuple(tracked_position),
001400:                 'area': float(proposed_size),
001401:             }
001402:             if (
001403:                     contact_normal_path and
001404:                     int(current_sample['frame']) !=
001405:                     int(contact_normal_path[-1].get('frame', -1000000)) + 1):
001406:                 contact_normal_path = []
001407: 
001408:             if contact_normal_path:
001409:                 previous_sample = contact_normal_path[-1]
001410:                 previous_pos = tuple(previous_sample['pos'])
001411:                 step_dx = float(tracked_position[0] - previous_pos[0])
001412:                 step_dy = float(tracked_position[1] - previous_pos[1])
001413:                 step_distance = math.hypot(step_dx, step_dy)
001414:                 serve_dy = int(getattr(self, 'serve_direction_dy', 0) or 0)
001415:                 netward_progress = -step_dy if serve_dy < 0 else step_dy
001416:                 if not (
001417:                         20.0 <= step_distance <= 220.0 and
001418:                         netward_progress >= 15.0):
001419:                     contact_normal_path = []
001420: 
001421:             contact_normal_path.append(current_sample)
001422:             contact_normal_path = contact_normal_path[-3:]
001423:             self._contact_normal_fallback_path = contact_normal_path
001424: 
001425:             if len(contact_normal_path) == 3:
001426:                 p0 = tuple(contact_normal_path[0]['pos'])
001427:                 p1 = tuple(contact_normal_path[1]['pos'])
001428:                 p2 = tuple(contact_normal_path[2]['pos'])
001429:                 v1 = (
001430:                     float(p1[0] - p0[0]),
001431:                     float(p1[1] - p0[1]),
001432:                 )
001433:                 v2 = (
001434:                     float(p2[0] - p1[0]),
001435:                     float(p2[1] - p1[1]),
001436:                 )
001437:                 len1 = math.hypot(v1[0], v1[1])
001438:                 len2 = math.hypot(v2[0], v2[1])
001439:                 cosine = (
001440:                     (v1[0] * v2[0] + v1[1] * v2[1]) /
001441:                     max(1.0, len1 * len2)
001442:                 )
001443:                 serve_dy = int(getattr(self, 'serve_direction_dy', 0) or 0)
001444:                 total_netward = (
001445:                     float(p0[1] - p2[1])
001446:                     if serve_dy < 0 else
001447:                     float(p2[1] - p0[1])
001448:                 )
001449:                 coherent_path = (
001450:                     20.0 <= len1 <= 220.0 and
001451:                     20.0 <= len2 <= 220.0 and
001452:                     cosine >= 0.90 and
001453:                     total_netward >= 100.0
001454:                 )
001455:                 if coherent_path:
001456:                     self._contact_local_ai_state = None
001457:                     self._contact_local_ai_cooldown_until_frame = max(
001458:                         int(getattr(
001459:                             self,
001460:                             '_contact_local_ai_cooldown_until_frame',
001461:                             -1000000,
001462:                         )),
001463:                         int(self.frame_count) + 2,
001464:                     )
001465:                     self._contact_normal_fallback_path = []
001466:                     print(
001467:                         f"[CONTACT_NORMAL_PATH_RELEASE] f{self.frame_count}: "
001468:                         f"accepted normal HSV={tuple(tracked_position)} after "
001469:                         f"3 coherent netward proposals "
001470:                         f"path={[entry['pos'] for entry in contact_normal_path]} "
001471:                         f"cos={cosine:.3f} net={total_netward:.1f}px"
001472:                     )
001473:                     return None
001474:         else:
001475:             self._contact_normal_fallback_path = []
001476: 
001477:         if contact_jump_candidate:
001478:             return (
001479:                 f'contact-jump:{proposed_speed:.0f}px>={jump_floor:.0f}/'
001480:                 f'angle={angle_delta:.0f}deg/pred={prediction_error:.0f}px'
001481:             )
001482: 
001483:         if (
001484:                 (contact_near or watch_active) and prior_speed >= 18.0 and proposed_speed >= 18.0 and
001485:                 angle_delta >= 70.0 and prediction_error >= 30.0):
001486:             return (
001487:                 f'contact-turn:{angle_delta:.0f}deg/'
001488:                 f'pred={prediction_error:.0f}px'
001489:             )
001490: 
001491:         if (
001492:                 (contact_near or watch_active) and prior_speed >= 18.0 and
001493:                 proposed_speed <= max(6.0, prior_speed * 0.35) and
001494:                 prediction_error >= 18.0):
001495:             return (
001496:                 f'contact-stall:{prior_speed:.0f}->{proposed_speed:.0f}px/'
001497:                 f'pred={prediction_error:.0f}px'
001498:             )
001499: 
001500:         if (
001501:                 (contact_near or watch_active) and proposed_speed >= 35.0 and size_ratio >= 2.6 and
001502:                 prediction_error >= 35.0):
001503:             return (
001504:                 f'contact-size-jump:{size_ratio:.1f}x/'
001505:                 f'pred={prediction_error:.0f}px'
001506:             )
001507:         return None
001508: 
001509:     def _run_contact_local_ai_frame(self, frame, previous_position, arm_reason=None):
001510:         """Let Local AI own a short generic racket-contact sequence.
001511: 
001512:         The implementation intentionally reuses the already validated
001513:         --force-local-ai-range tracker: player/head/shoe/racket candidates are
001514:         excluded before scoring, the AI-selected point becomes the next anchor,
001515:         and trajectory prediction starts after two accepted AI positions.
001516:         """
001517:         recovery = getattr(self, 'local_ai_recovery', None)
001518:         if recovery is None or frame is None or previous_position is None:
001519:             return None
001520: 
001521:         current = int(self.frame_count)
001522:         state = getattr(self, '_contact_local_ai_state', None)
001523:         if arm_reason is not None and not isinstance(state, dict):
001524:             # Seed CONTACT_LOCAL_AI with the trusted pre-contact anchor.  On the
001525:             # contact frame this is only one history point, so no trajectory is
001526:             # imposed across the racket direction change.  After Local AI accepts
001527:             # the first outgoing point, the next frame immediately has two points
001528:             # and can predict the outgoing ball trajectory.
001529:             seed_history = []
001530:             if previous_position is not None:
001531:                 seed_history.append({
001532:                     'frame': current - 1,
001533:                     'pos': tuple(previous_position),
001534:                     'score': 1.0,
001535:                     'contact_seed': True,
001536:                 })
001537:             state = {
001538:                 'start_frame': current,
001539:                 'deadline': current + 22,
001540:                 'history': seed_history,
001541:                 'outside_count': 0,
001542:                 'miss_count': 0,
001543:                 'reason': str(arm_reason),
001544:             }
001545:             self._contact_local_ai_state = state
001546:             if seed_history:
001547:                 print(
001548:                     f"[CONTACT_LOCAL_AI_SEED] f{current}: "
001549:                     f"seed_frame={current - 1} position={tuple(previous_position)}"
001550:                 )
001551:             # Do not let an older tight-ROI experiment compete with contact
001552:             # ownership.  The automatic owner is the stronger, player-excluding
001553:             # path and will hand control back explicitly.
001554:             self._local_ai_tight_roi_follow = None
001555:             print(
001556:                 f"[CONTACT_LOCAL_AI_ARM] f{current}: anchor={tuple(previous_position)} "
001557:                 f"reason={arm_reason} radius={self._contact_local_ai_radius:.0f}px"
001558:             )
001559: 
001560:         state = getattr(self, '_contact_local_ai_state', None)
001561:         if not isinstance(state, dict):
001562:             return None
001563:         if current > int(state.get('deadline', -1)):
001564:             print(
001565:                 f"[CONTACT_LOCAL_AI_EXPIRE] f{current}: "
001566:                 f"started=f{state.get('start_frame')} reason={state.get('reason')}"
001567:             )
001568:             self._contact_local_ai_state = None
001569:             self._contact_local_ai_cooldown_until_frame = current + 3
001570:             return None
001571: 
001572:         # Reuse the proven forced-AI implementation without exposing its state
001573:         # to the user's manual debug range.
001574:         saved_range = getattr(self, '_force_local_ai_range', None)
001575:         saved_history = list(getattr(self, '_force_local_ai_history', []) or [])
001576:         saved_radius = float(getattr(self, '_force_local_ai_radius', 450.0))
001577:         saved_min_score = float(getattr(self, '_force_local_ai_min_score', 0.985))
001578:         saved_return_none = bool(getattr(self, '_force_local_ai_return_none_on_miss', False))
001579:         saved_max_scored = int(getattr(self, '_force_local_ai_max_score_candidates', 128))
001580:         saved_allow_gap = bool(getattr(self, '_force_local_ai_allow_history_gap', False))
001581:         saved_continuity_guard = bool(getattr(self, '_force_local_ai_continuity_guard', False))
001582:         try:
001583:             self._force_local_ai_range = (
001584:                 int(state['start_frame']), int(state['deadline'])
001585:             )
001586:             self._force_local_ai_history = list(state.get('history', []) or [])
001587:             miss_count = int(state.get('miss_count', 0))
001588:             base_radius = float(getattr(self, '_contact_local_ai_radius', 450.0))
001589:             self._force_local_ai_radius = min(900.0, base_radius + 90.0 * miss_count)
001590:             self._force_local_ai_min_score = float(
001591:                 getattr(self, '_contact_local_ai_min_score', 0.985)
001592:             )
001593:             self._force_local_ai_return_none_on_miss = True
001594:             self._force_local_ai_max_score_candidates = 384
001595:             self._force_local_ai_allow_history_gap = True
001596:             self._force_local_ai_continuity_guard = True
001597:             selected = self._force_local_ai_frame(frame, previous_position)
001598:             state['history'] = list(getattr(self, '_force_local_ai_history', []) or [])
001599:         finally:
001600:             self._force_local_ai_range = saved_range
001601:             self._force_local_ai_history = saved_history
001602:             self._force_local_ai_radius = saved_radius
001603:             self._force_local_ai_min_score = saved_min_score
001604:             self._force_local_ai_return_none_on_miss = saved_return_none
001605:             self._force_local_ai_max_score_candidates = saved_max_scored
001606:             self._force_local_ai_allow_history_gap = saved_allow_gap
001607:             self._force_local_ai_continuity_guard = saved_continuity_guard
001608: 
001609:         if selected is None:
001610:             state['miss_count'] = int(state.get('miss_count', 0)) + 1
001611:             self._contact_local_ai_state = state
001612:             print(
001613:                 f"[CONTACT_LOCAL_AI_MISS] f{current}: normal tracker remains active; "
001614:                 f"misses={state['miss_count']}"
001615:             )
001616:             return None
001617: 
001618:         state['miss_count'] = 0
001619:         selected = (int(selected[0]), int(selected[1]))
001620:         zone = self._player_point_zone(selected)
001621:         inside_contact = self._point_in_player_contact_corridor(selected)
001622:         age = current - int(state['start_frame'])
001623:         if zone is None and not inside_contact:
001624:             state['outside_count'] = int(state.get('outside_count', 0)) + 1
001625:         else:
001626:             state['outside_count'] = 0
001627: 
001628:         print(
001629:             f"[CONTACT_LOCAL_AI] f{current}: selected={selected} age={age}f "
001630:             f"outside={state['outside_count']} zone={zone}"
001631:         )
001632: 
001633:         # Keep AI for several frames after contact.  This prevents an immediate
001634:         # handback to HSV while its search anchor is still the racket/player.
001635:         # A trajectory-hotspot takeover is not a player-contact recovery; keep
001636:         # it through six transitions so it can bridge the bad HSV zone, then
001637:         # hand control back on the following frame when normal HSV is coherent.
001638:         minimum_owner_age = (
001639:             6
001640:             if str(state.get('reason', '')).startswith('trajectory-hotspot-turn:')
001641:             else 5
001642:         )
001643:         if age >= minimum_owner_age and int(state.get('outside_count', 0)) >= 3:
001644:             print(
001645:                 f"[CONTACT_LOCAL_AI_HANDOFF] f{current}: selected={selected} "
001646:                 f"clear of player/contact corridor for {state['outside_count']}f"
001647:             )
001648:             self._contact_local_ai_state = None
001649:             self._contact_local_ai_cooldown_until_frame = current + 3
001650:         else:
001651:             self._contact_local_ai_state = state
001652:         return selected
001653: 
001654:     def _force_local_ai_frame(self, frame, previous_position):
001655:         """TEST ONLY: bypass HSV and let Local AI own an exact frame range.
001656: 
001657:         The previous AI-selected point becomes the next search anchor. Player,
001658:         head, shoe, and racket regions are excluded before scoring. Once two AI
001659:         points exist, high-confidence candidates must also fit the extrapolated
001660:         AI trajectory. Outside --force-local-ai-range this method returns None
001661:         and the normal tracker is unchanged.
001662:         """
001663:         frame_range = getattr(self, "_force_local_ai_range", None)
001664:         recovery = getattr(self, "local_ai_recovery", None)
001665:         if frame_range is None or frame is None or recovery is None:
001666:             return None
001667: 
001668:         start_frame, end_frame = frame_range
001669:         current = int(self.frame_count)
001670:         if not (int(start_frame) <= current <= int(end_frame)):
001671:             if current > int(end_frame):
001672:                 self._force_local_ai_history = []
001673:             return None
001674: 
001675:         from ball_ai_recovery_probe import collect_candidates
001676: 
001677:         history = list(getattr(self, "_force_local_ai_history", []) or [])
001678:         # Keep the last validated history across misses. Normal HSV positions
001679:         # during recovery are hypotheses and must not bridge this history.
001680:         if not history and previous_position is not None:
001681:             history = [{"frame": current - 1, "pos": tuple(previous_position)}]
001682:         self._force_local_ai_history = history
001683: 
001684:         anchor = tuple(history[-1]["pos"]) if history else (
001685:             tuple(previous_position) if previous_position is not None else None
001686:         )
001687:         if anchor is None:
001688:             print(f"[FORCE_LOCAL_AI_HOLD] f{current}: no anchor available")
001689:             return previous_position
001690: 
001691:         radius = max(40.0, float(getattr(self, "_force_local_ai_radius", 450.0)))
001692:         min_score = float(getattr(self, "_force_local_ai_min_score", 0.985))
001693: 
001694:         config = dict(recovery._config)
001695:         config["force_low_sat"] = {
001696:             "h_min": 80, "h_max": 135,
001697:             "s_min": 10, "s_max": 60,
001698:             "v_min": 110, "v_max": 255,
001699:         }
001700:         modes = (
001701:             "regular_court", "alt1", "alt2", "alt3", "s_30", "h_10",
001702:             "force_low_sat",
001703:         )
001704: 
001705:         predicted = None
001706:         previous_speed = 0.0
001707:         previous_vector = None
001708:         if len(history) >= 2:
001709:             p0 = tuple(history[-2]["pos"])
001710:             p1 = tuple(history[-1]["pos"])
001711:             vx = float(p1[0] - p0[0])
001712:             vy = float(p1[1] - p0[1])
001713:             previous_speed = math.hypot(vx, vy)
001714:             if previous_speed >= 3.0:
001715:                 previous_vector = (vx, vy)
001716:                 predicted = (int(round(p1[0] + vx)), int(round(p1[1] + vy)))
001717: 
001718:         candidates = collect_candidates(
001719:             frame, config, modes=modes,
001720:             min_area=3.0, max_area=1200.0,
001721:             around=anchor, radius=radius,
001722:             dedup_distance=4.0,
001723:         )
001724: 
001725:         clear_candidates = []
001726:         for candidate in candidates:
001727:             point = (int(candidate["x"]), int(candidate["y"]))
001728:             if self._player_point_zone(point) is None:
001729:                 clear_candidates.append(candidate)
001730: 
001731:         sort_anchor = predicted if predicted is not None else anchor
001732:         clear_candidates.sort(
001733:             key=lambda item: math.hypot(
001734:                 float(item["x"]) - float(sort_anchor[0]),
001735:                 float(item["y"]) - float(sort_anchor[1]),
001736:             )
001737:         )
001738:         max_scored = max(32, min(512, int(getattr(
001739:             self, "_force_local_ai_max_score_candidates", 128
001740:         ))))
001741:         subset = clear_candidates[:max_scored]
001742:         scored = recovery._score(frame, current, subset) if subset else []
001743: 
001744:         tolerance = None
001745:         if predicted is not None:
001746:             tolerance = max(120.0, min(300.0, previous_speed * 1.8 + 60.0))
001747: 
001748:         recent = list(getattr(self, "ball_velocity_history", []) or [])[-5:]
001749:         prior_speed = float(np.median(recent)) if recent else 0.0
001750:         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
001751:         previous_gray = getattr(self, '_local_ai_tight_roi_previous_gray', None)
001752:         if previous_gray is None:
001753:             previous_gray = getattr(self, '_prev_frame_gray', None)
001754: 
001755:         eligible = []
001756:         continuity_rejected = 0
001757:         for candidate in scored:
001758:             score = float(candidate.get("ai_score", 0.0) or 0.0)
001759:             if score < min_score:
001760:                 continue
001761:             point = (int(candidate["x"]), int(candidate["y"]))
001762:             if self._player_point_zone(point) is not None:
001763:                 continue
001764: 
001765:             anchor_dist = math.hypot(point[0] - anchor[0], point[1] - anchor[1])
001766:             motion = self._candidate_motion_metrics(
001767:                 gray, point[0], point[1], previous_gray=previous_gray)
001768:             rejection = validate_ai_step(
001769:                 point, current, history, prior_speed=prior_speed, motion=motion)
001770:             if rejection:
001771:                 continuity_rejected += 1
```

## Lines 26679-26889
```python
026679:                         if self._ignore_unresolved_timeout_if_game_decider(
026680:                             "POINT_TIMEOUT", end_position=self.ball_center, frame=frame
026681:                         ):
026682:                             print(
026683:                                 f"[POINT_IGNORED] f{self.frame_count}: reason=POINT_TIMEOUT "
026684:                                 f"duration={dur}f would decide game without a located ball"
026685:                             )
026686:                         else:
026687:                             timeout_reason = "POINT_TIMEOUT"
026688:                             if self._in_court_timeout_landing_outcome(self.ball_center, frame) is not None:
026689:                                 timeout_reason = "Ball stopped on player side"
026690:                             # A duration timeout is only a fallback inference.  The
026691:                             # terminal verifier can still see a compact moving ball
026692:                             # in the just-decoded frame pair; continue the rally
026693:                             # rather than scoring a false timeout in that case.
026694:                             if self._resume_from_terminal_motion_candidate(
026695:                                 timeout_reason, self.ball_center, frame=frame
026696:                             ):
026697:                                 continue
026698:                             print(
026699:                                 f"[POINT_END] f{self.frame_count}: reason={timeout_reason} "
026700:                                 f"duration={dur}f — returning to serve detection"
026701:                             )
026702:                             self._record_point_result(timeout_reason, end_position=self.ball_center, frame=frame)
026703:                         game_state = "WAITING_FOR_SERVE"
026704:                         reset_tracking_state(hold_end_marker=True)
026705: 
026706:                 # Track ball through the point
026707:                 tracked_position = None
026708:                 if self.tracking and self.hsv_lower is not None and self.ball_center is not None:
026709:                     prev_ball_center = self.ball_center
026710:                     prev_stuck = self.stuck_frame_count
026711:                     prev_top_return_wait = self._top_return_wait_active()
026712:                     prev_back_return_wait = self._back_return_wait_active()
026713:                     pre_track_snapshot = self._snapshot_tracking_state_for_provisional_guard()
026714:                     self._local_ai_tight_roi_previous_gray = (
026715:                         pre_track_snapshot.get('_prev_frame_gray')
026716:                     )
026717:                     forced_local_ai = False
026718:                     contact_local_ai = False
026719:                     tracked_position = self._force_local_ai_frame(frame, prev_ball_center)
026720:                     if tracked_position is not None:
026721:                         forced_local_ai = True
026722:                     else:
026723:                         # Once armed, contact Local AI owns the next few frames
026724:                         # before HSV is allowed to mutate the anchor again.
026725:                         tracked_position = self._run_contact_local_ai_frame(
026726:                             frame, prev_ball_center
026727:                         )
026728:                         if tracked_position is not None:
026729:                             contact_local_ai = True
026730:                         else:
026731:                             tracked_position = self.track_ball_in_frame(frame)
026732:                             contact_reason = self._contact_local_ai_trigger(
026733:                                 prev_ball_center, tracked_position, pre_track_snapshot
026734:                             )
026735:                             if contact_reason is not None:
026736:                                 rejected_hsv = (
026737:                                     tuple(tracked_position)
026738:                                     if tracked_position is not None else None
026739:                                 )
026740:                                 post_track_snapshot = self._snapshot_tracking_state_for_provisional_guard()
026741:                                 self._restore_tracking_state_for_provisional_guard(
026742:                                     pre_track_snapshot
026743:                                 )
026744:                                 self._local_ai_tight_roi_previous_gray = (
026745:                                     pre_track_snapshot.get('_prev_frame_gray')
026746:                                 )
026747:                                 # Debug only: preserve the normal detector's contact
026748:                                 # hypothesis long enough to print candidate agreement.
026749:                                 self._contact_local_ai_debug_normal_candidate = rejected_hsv
026750:                                 tracked_position = self._run_contact_local_ai_frame(
026751:                                     frame, prev_ball_center, arm_reason=contact_reason
026752:                                 )
026753:                                 self._contact_local_ai_debug_normal_candidate = None
026754:                                 if tracked_position is not None:
026755:                                     contact_local_ai = True
026756:                                     print(
026757:                                         f"[CONTACT_LOCAL_AI_REPLACE] f{self.frame_count}: "
026758:                                         f"HSV={rejected_hsv} -> AI={tracked_position}"
026759:                                     )
026760:                                 else:
026761:                                     # AI is an assist: if it has no accepted candidate,
026762:                                     # keep the normal tracker result for this frame.
026763:                                     self._restore_tracking_state_for_provisional_guard(
026764:                                         pre_track_snapshot
026765:                                     )
026766:                                     tracked_position = prev_ball_center
026767:                                     contact_local_ai = False
026768:                                     print(
026769:                                         f"[CONTACT_LOCAL_AI_FALLBACK] f{self.frame_count}: "
026770:                                         f"AI miss; preserving trusted position={tracked_position}"
026771:                                     )
026772:                     if self.local_ai_recovery is not None and self._local_ai_frame_buffer:
026773:                         self._local_ai_frame_buffer[-1]["normal_position"] = (
026774:                             tuple(tracked_position) if tracked_position is not None else None
026775:                         )
026776:                     self._debug_local_ai_shadow_frame(frame, prev_ball_center, tracked_position)
026777:                     if not forced_local_ai and not contact_local_ai:
026778:                         tracked_position = self._try_local_ai_recovery(
026779:                             prev_ball_center, tracked_position, prev_stuck,
026780:                             pre_track_snapshot=pre_track_snapshot,
026781:                             frame=frame,
026782:                         )
026783:                     self._local_ai_tight_roi_previous_gray = None
026784:                     # Reject any position that jumps impossibly far in one frame (false positive).
026785:                     # When the tracker is in re-acquisition mode (stuck >= 5 before the call), allow
026786:                     # a larger jump because the ball may have traveled far while lost.
026787:                     if tracked_position and prev_ball_center:
026788:                         jump = math.hypot(tracked_position[0] - prev_ball_center[0],
026789:                                           tracked_position[1] - prev_ball_center[1])
026790:                         motion_reacq_this_frame = (
026791:                             getattr(self, '_last_motion_reacq_frame', -1000000) == self.frame_count and
026792:                             getattr(self, '_last_motion_reacq_pos', None) == tracked_position
026793:                         )
026794:                         trusted_rally_launch = (
026795:                             getattr(self, '_trusted_rally_launch_frame', -1000000) == self.frame_count and
026796:                             getattr(self, '_trusted_rally_launch_pos', None) == tracked_position
026797:                         )
026798:                         # Allow a larger jump when re-acquiring after being stuck for 5+ frames:
026799:                         # a racket hit can send the ball 800+ px in one frame, so we use 1500px
026800:                         # to let motion-based re-acquisition recover across the full court.
026801:                         max_jump = 1500 if (
026802:                             prev_stuck >= 5 or prev_top_return_wait or prev_back_return_wait or
026803:                             motion_reacq_this_frame or trusted_rally_launch
026804:                         ) else 400
026805:                         if jump > max_jump:
026806:                             print(f"[JUMP_REJECTED] f{self.frame_count}: jumped {jump:.0f}px from {prev_ball_center} to {tracked_position} (limit={max_jump}px, prev_stuck={prev_stuck}), keeping previous")
026807:                             self.ball_center = prev_ball_center
026808:                             self.stuck_frame_count = max(self.stuck_frame_count, prev_stuck + 1)
026809:                             tracked_position = prev_ball_center
026810: 
026811:                     # Validate only the position that survived the physical
026812:                     # jump gate above.  Previously the provisional serve
026813:                     # state consumed a 425px false contour at a prior frame before
026814:                     # that contour was rejected below.  Its bogus vector then
026815:                     # hid the real upward reversal at a prior frame.
026816:                     # Once a far toss is independently marked as caught, it
026817:                     # becomes a *history* question only.  Do not continue to
026818:                     # mutate the live provisional flags: the previous attempt
026819:                     # cleared those flags at a prior frame and changed the otherwise
026820:                     # stable HSV path of the following real rally.
026821:                     tainted_far_toss_active = (
026822:                         isinstance(
026823:                             getattr(self, '_tainted_provisional_serve_start', None),
026824:                             dict,
026825:                         ) and
026826:                         getattr(self, '_provisional_serve_start_kind', None) ==
026827:                         'far-top-post-hit'
026828:                     )
026829:                     provisional_kind_before_validation = getattr(
026830:                         self, '_provisional_serve_start_kind', None
026831:                     )
026832:                     provisional_reject_reason = (
026833:                         None if tainted_far_toss_active else
026834:                         self._validate_provisional_serve_start(
026835:                             prev_ball_center, tracked_position
026836:                         )
026837:                     )
026838:                     # A pending far-toss token is solely a fallback for a
026839:                     # *rejected* practice toss.  Once this tentative flight
026840:                     # has earned three coherent netward steps, it is a real
026841:                     # serve (for example a prior frame -> a prior frame) and the token must
026842:                     # not survive its fault/end state.  Leaving it alive
026843:                     # caused the local model to re-use that old start at
026844:                     # a prior frame and inject a second, false point.
026845:                     if provisional_reject_reason is not None:
026846:                         provisional_kind = getattr(self, '_provisional_serve_start_kind', None)
026847:                         rejected_start = self._current_history_serve_start_frame()
026848:                         if provisional_kind == 'far-top-post-hit':
026849:                             # This is a caught far-side toss.  Do *not* reset
026850:                             # to WAITING_FOR_SERVE: that was the rejected
026851:                             # approach and it changed later, already-verified
026852:                             # points.  Keep the normal HSV flow alive and ask
026853:                             # local AI to prove a later outgoing path in
026854:                             # shadow mode.  Only proven evidence may alter the
026855:                             # final history/serve-fault interpretation.
026856:                             self._tainted_provisional_serve_start = {
026857:                                 'start_frame': rejected_start,
026858:                                 'detected_frame': int(self.frame_count),
026859:                                 'anchor': tuple(prev_ball_center),
026860:                                 'reason': provisional_reject_reason,
026861:                             }
026862:                             print(
026863:                                 f"[SERVE_START_TAINTED] f{self.frame_count}: "
026864:                                 f"keeping tracker/provisional state for start f{rejected_start}; "
026865:                                 f"{provisional_reject_reason}"
026866:                             )
026867:                         else:
026868:                             print(
026869:                                 f"[SERVE_START_REJECTED] f{self.frame_count}: "
026870:                                 f"discarding provisional start f{rejected_start}; "
026871:                                 f"{provisional_reject_reason}"
026872:                             )
026873:                             self._point_history_current = None
026874:                             self._provisional_serve_start_kind = None
026875:                             self._serve_start_requires_confirmation = False
026876:                             game_state = "WAITING_FOR_SERVE"
026877:                             clear_waiting_serve_history()
026878:                             reset_tracking_state()
026879:                             continue
026880:                     # Shadow verification is intentionally side-effect free
026881:                     # for ball position/game state.  It only supplies a
026882:                     # later verified launch frame for history/fault scoring.
026883:                     reconcile_tainted_far_serve_start()
026884:                     if getattr(self, "_discard_provisional_serve_from_ai", False):
026885:                         self._discard_provisional_serve_from_ai = False
026886:                         self._ignore_unconfirmed_serve_start_result(
026887:                             "local AI rejected repeated all-player-body recovery path"
026888:                         )
026889:                         print(
```
