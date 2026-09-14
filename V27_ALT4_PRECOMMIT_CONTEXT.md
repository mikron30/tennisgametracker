# V27 ALT4 precommit context

## ALT4 HSV OVERRIDE around source line 20610
```python
020480:             # Skip this during full-frame scan recovery - ball direction changed after player hit.
020481:             # Also skip during post-reacquire window: the serve contact can instantly reverse
020482:             # the ball, so a direction / speed change here is expected and correct.
020483:             skip_upper_wall_override = upper_wall_search_context and best_source in ('alt4', 'alt5', 'alt6')
020484:             if (not serve_contact_grace and not rally_contact_grace and not ground_bounce_grace and not serve_direction_search
020485:                     and lower_contact_launch_context is None and not _in_post_reacq and self.last_motion
020486:                     and not skip_upper_wall_override and not upper_contact_turn_commit
020487:                     and not large_lower_launch_override
020488:                     and not large_lower_bounce_launch_override
020489:                     and not upper_far_player_escape_override
020490:                     and not upper_dynamic_departure_override
020491:                     and not accepted_top_return_reentry and not top_return_reentry_grace
020492:                     and not accepted_back_return_reentry and not back_return_reentry_grace
020493:                     and self.ball_center and self.stuck_frame_count < 5):
020494:                 lm_dx = self.last_motion['dx']
020495:                 lm_dy = self.last_motion['dy']
020496:                 lm_dist = self.last_motion['distance']
020497:                 mv_dx = cx - self.ball_center[0]
020498:                 mv_dy = cy - self.ball_center[1]
020499:                 dot = lm_dx * mv_dx + lm_dy * mv_dy
020500:                 speed_diff = abs(velocity - lm_dist) if lm_dist else 0
020501:                 angle_jump = 0.0
020502:                 if self.last_direction is not None and direction_deg is not None:
020503:                     delta = abs(direction_deg - self.last_direction) % 360
020504:                     angle_jump = min(delta, 360 - delta)
020505:                 speed_ratio = (velocity / lm_dist) if lm_dist else 1.0
020506:                 if (dot < 0 or angle_jump > 45 or speed_ratio > 1.8 or speed_ratio < 0.6 or
020507:                         (lm_dist and speed_diff > lm_dist * 0.6)):
020508:                     retrack = self.retrack_with_alt_hsv(
020509:                         search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
020510:                         frame_gray=frame_gray, filter_key="alt1"
020511:                     )
020512:                     override_applied = False
020513:                     if retrack is not None and self._should_accept_hsv_override(
020514:                             "alt1", retrack, current_pos, current_area, prev_pos, predicted_point,
020515:                             frame_gray, current_filter_key=current_filter_key):
020516:                         cx, cy = retrack['pos']
020517:                         hsv_values = retrack['hsv']
020518:                         bulb_size = retrack['area']
020519:                         self.using_alt_hsv = True
020520:                         self.using_alt2_hsv = False
020521:                         self.using_alt3_hsv = False
020522:                         self.using_alt4_hsv = False
020523:                         self.using_alt6_hsv = False
020524:                         if self.alt_focus_hsv_lower is not None and self.alt_focus_hsv_upper is not None:
020525:                             self.hsv_lower = self.alt_focus_hsv_lower
020526:                             self.hsv_upper = self.alt_focus_hsv_upper
020527:                         print(f"Frame {self.frame_count}: [ALT HSV OVERRIDE] Ball at ({cx}, {cy})")
020528:                         override_applied = True
020529:                         hsv_override_applied = True
020530:                     if (not override_applied and self.alt2_hsv_lower is not None and
020531:                             self.alt2_hsv_upper is not None):
020532:                         retrack2 = self.retrack_with_alt2_hsv(
020533:                             search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
020534:                             frame_gray=frame_gray, filter_key="alt2"
020535:                         )
020536:                         if retrack2 is not None and self._player_reacq_static_candidate(
020537:                                 retrack2, predicted_point=predicted_point):
020538:                             print(
020539:                                 f"  DEBUG: Rejecting alt2 HSV override during player-reacq guard "
020540:                                 f"at {retrack2['pos']} motion="
020541:                                 f"{retrack2.get('motion_mean', 0.0):.1f}/"
020542:                                 f"{retrack2.get('motion_max', 0.0):.1f}"
020543:                             )
020544:                             retrack2 = None
020545:                         if retrack2 is not None and self._should_accept_hsv_override(
020546:                                 "alt2", retrack2, current_pos, current_area, prev_pos, predicted_point,
020547:                                 frame_gray, current_filter_key=current_filter_key):
020548:                             cx, cy = retrack2['pos']
020549:                             hsv_values = retrack2['hsv']
020550:                             bulb_size = retrack2['area']
020551:                             self.using_alt_hsv = False
020552:                             self.using_alt2_hsv = True
020553:                             self.using_alt3_hsv = False
020554:                             self.using_alt6_hsv = False
020555:                             self.hsv_lower = self.alt2_hsv_lower
020556:                             self.hsv_upper = self.alt2_hsv_upper
020557:                             print(f"Frame {self.frame_count}: [ALT2 HSV OVERRIDE] Ball at ({cx}, {cy})")
020558:                             override_applied = True
020559:                             hsv_override_applied = True
020560:                             if self.last_motion and self.alt3_hsv_lower is not None and self.alt3_hsv_upper is not None:
020561:                                 lm_dist = self.last_motion['distance']
020562:                                 mv_dx = cx - self.ball_center[0]
020563:                                 mv_dy = cy - self.ball_center[1]
020564:                                 mv_dist = math.hypot(mv_dx, mv_dy)
020565:                                 speed_ratio = (mv_dist / lm_dist) if lm_dist else 1.0
020566:                                 angle_jump = 0.0
020567:                                 if self.last_direction is not None:
020568:                                     direction_deg = math.degrees(math.atan2(mv_dy, mv_dx))
020569:                                     delta = abs(direction_deg - self.last_direction) % 360
020570:                                     angle_jump = min(delta, 360 - delta)
020571:                                 if angle_jump > 45 or speed_ratio > 1.8 or speed_ratio < 0.6:
020572:                                     retrack3 = self.retrack_with_alt2_hsv(
020573:                                         search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
020574:                                         lower=self.alt3_hsv_lower, upper=self.alt3_hsv_upper, frame_gray=frame_gray,
020575:                                         filter_key="alt3"
020576:                                     )
020577:                                     if retrack3 is not None and self._should_accept_hsv_override(
020578:                                             "alt3", retrack3, current_pos, current_area, prev_pos,
020579:                                             predicted_point, frame_gray,
020580:                                             current_filter_key=current_filter_key):
020581:                                         cx, cy = retrack3['pos']
020582:                                         hsv_values = retrack3['hsv']
020583:                                         bulb_size = retrack3['area']
020584:                                         self.using_alt2_hsv = False
020585:                                         self.using_alt3_hsv = True
020586:                                         self.using_alt6_hsv = False
020587:                                         self.hsv_lower = self.alt3_hsv_lower
020588:                                         self.hsv_upper = self.alt3_hsv_upper
020589:                                         print(f"Frame {self.frame_count}: [ALT3 HSV OVERRIDE] Ball at ({cx}, {cy})")
020590:                     if (not override_applied and self.alt4_hsv_lower is not None and
020591:                             self.alt4_hsv_upper is not None):
020592:                         retrack4 = self.retrack_with_alt2_hsv(
020593:                             search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
020594:                             lower=self.alt4_hsv_lower, upper=self.alt4_hsv_upper, frame_gray=frame_gray,
020595:                             filter_key="alt4"
020596:                         )
020597:                         if retrack4 is not None and self._should_accept_hsv_override(
020598:                                 "alt4", retrack4, current_pos, current_area, prev_pos, predicted_point,
020599:                                 frame_gray, current_filter_key=current_filter_key):
020600:                             cx, cy = retrack4['pos']
020601:                             hsv_values = retrack4['hsv']
020602:                             bulb_size = retrack4['area']
020603:                             self.using_alt_hsv = False
020604:                             self.using_alt2_hsv = False
020605:                             self.using_alt3_hsv = False
020606:                             self.using_alt4_hsv = True
020607:                             self.using_alt6_hsv = False
020608:                             self.hsv_lower = self.alt4_hsv_lower
020609:                             self.hsv_upper = self.alt4_hsv_upper
020610:                             print(f"Frame {self.frame_count}: [ALT4 HSV OVERRIDE] Ball at ({cx}, {cy})")
020611:                             override_applied = True
020612:                             hsv_override_applied = True
020613:                     if (not override_applied and self.alt6_hsv_lower is not None and
020614:                             self.alt6_hsv_upper is not None):
020615:                         retrack6 = self.retrack_with_alt2_hsv(
020616:                             search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
020617:                             lower=self.alt6_hsv_lower, upper=self.alt6_hsv_upper, frame_gray=frame_gray,
020618:                             filter_key="alt6"
020619:                         )
020620:                         if retrack6 is not None and self._should_accept_hsv_override(
020621:                                 "alt6", retrack6, current_pos, current_area, prev_pos, predicted_point,
020622:                                 frame_gray, current_filter_key=current_filter_key):
020623:                             cx, cy = retrack6['pos']
020624:                             hsv_values = retrack6['hsv']
020625:                             bulb_size = retrack6['area']
020626:                             self.using_alt_hsv = False
020627:                             self.using_alt2_hsv = False
020628:                             self.using_alt3_hsv = False
020629:                             self.using_alt6_hsv = True
020630:                             self.hsv_lower = self.alt6_hsv_lower
020631:                             self.hsv_upper = self.alt6_hsv_upper
020632:                             print(f"Frame {self.frame_count}: [ALT6 HSV OVERRIDE] Ball at ({cx}, {cy})")
020633:                             override_applied = True
020634:                             hsv_override_applied = True
020635:                     if self._should_try_alts9_11_override(frame.shape, allow_inactive):
020636:                         specialist_current_pos = (cx, cy)
020637:                         specialist_current_area = bulb_size
020638:                         retrack_s911 = self.retrack_with_alt2_hsv(
020639:                             search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
020640:                             lower=self.alts9_11_hsv_lower, upper=self.alts9_11_hsv_upper,
020641:                             frame_gray=frame_gray, filter_key="alts9_11", sparse_mode=True
020642:                         )
020643:                         if retrack_s911 is not None and self._should_accept_hsv_override(
020644:                                 "alts9_11", retrack_s911, specialist_current_pos, specialist_current_area,
020645:                                 prev_pos, predicted_point,
020646:                                 frame_gray, current_filter_key=current_filter_key):
020647:                             cx, cy = retrack_s911['pos']
020648:                             hsv_values = retrack_s911['hsv']
020649:                             bulb_size = retrack_s911['area']
020650:                             print(f"Frame {self.frame_count}: [ALTS9_11 HSV OVERRIDE] Ball at ({cx}, {cy})")
020651:                             override_applied = True
020652:                             hsv_override_applied = True
020653:                     if (not override_applied and
020654:                             (self.alt6_hsv_lower is None or self.alt6_hsv_upper is None) and
020655:                             self.alt3_hsv_lower is not None and self.alt3_hsv_upper is not None):
020656:                         retrack3 = self.retrack_with_alt2_hsv(
020657:                             search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
020658:                             lower=self.alt3_hsv_lower, upper=self.alt3_hsv_upper, frame_gray=frame_gray,
020659:                             filter_key="alt3"
020660:                         )
020661:                         if retrack3 is not None and self._should_accept_hsv_override(
020662:                                 "alt3", retrack3, current_pos, current_area, prev_pos, predicted_point,
020663:                                 frame_gray, current_filter_key=current_filter_key):
020664:                             cx, cy = retrack3['pos']
020665:                             hsv_values = retrack3['hsv']
020666:                             bulb_size = retrack3['area']
020667:                             self.using_alt_hsv = False
020668:                             self.using_alt2_hsv = False
020669:                             self.using_alt3_hsv = True
020670:                             self.using_alt6_hsv = False
020671:                             self.hsv_lower = self.alt3_hsv_lower
020672:                             self.hsv_upper = self.alt3_hsv_upper
020673:                             print(f"Frame {self.frame_count}: [ALT3 HSV OVERRIDE] Ball at ({cx}, {cy})")
020674: 
020675:             # HSV overrides can replace a stale contour with the real ball during
020676:             # top-screen return recovery. Keep that narrow path aligned with the
020677:             # final candidate, or it may hold the old marker because it is still
020678:             # scoring the pre-override fragment.
020679:             recent_top_return_override = (
020680:                 hsv_override_applied and
020681:                 self._recent_offscreen_return_hold_active(window_frames=12)
020682:             )
020683:             override_final_dx = override_final_dy = 0
020684:             override_final_velocity = None
020685:             if hsv_override_applied and prev_pos is not None:
020686:                 override_final_dx = cx - prev_pos[0]
020687:                 override_final_dy = cy - prev_pos[1]
020688:                 override_final_velocity = math.hypot(override_final_dx, override_final_dy)
020689:             last_motion_distance = (
020690:                 float(self.last_motion.get('distance', 0.0) or 0.0)
020691:                 if self.last_motion is not None else 0.0
020692:             )
020693:             override_replaced_false_jump = (
020694:                 hsv_override_applied and
020695:                 override_final_velocity is not None and
020696:                 velocity >= max(150.0, last_motion_distance * 4.0) and
020697:                 override_final_velocity <= max(40.0, last_motion_distance * 2.0)
020698:             )
020699:             if recent_top_return_override or override_replaced_false_jump:
020700:                 current_pos = (cx, cy)
020701:                 current_area = bulb_size
020702:                 dx = override_final_dx
020703:                 dy = override_final_dy
020704:                 velocity = override_final_velocity if override_final_velocity is not None else 0
020705:                 direction_deg = (
020706:                     math.degrees(math.atan2(dy, dx))
020707:                     if velocity > 0 else 0.0
020708:                 )
020709:             if recent_top_return_override:
020710:                 if (
020711:                         prev_pos is not None and
020712:                         dy >= max(8.0, frame_height * 0.004) and
020713:                         velocity <= max(70.0, float(self.last_motion.get('distance', 0.0) or 0.0) * 1.25)
020714:                         and bulb_size <= max(14.0, float(prev_ball_size or 0.0) * 3.5)
020715:                         and max(175, int(frame_height * 0.081)) <= cy <= max(270, int(frame_height * 0.13))):
020716:                     existing_defer = int(getattr(self, '_top_far_out_defer_until_frame', -1000000))
020717:                     if self.frame_count > existing_defer:
020718:                         self._top_far_out_defer_until_frame = self.frame_count + 3
020719:                         self._top_far_out_deferred_candidate = {
020720:                             'frame': self.frame_count,
020721:                             'pos': (cx, cy),
020722:                         }
020723:                         print(
020724:                             f"Frame {self.frame_count}: [TOP-FAR-OUT DEFER] "
020725:                             f"smooth HSV return continuation at ({cx},{cy})"
020726:                         )
020727: 
020728:             # Detect sudden ball size drop (occlusion by player)
020729:             # If ball was > 30px and now < 5px, it's being occluded — don't trust this detection
020730:             small_upper_flight_candidate = (
020731:                 (prev_ball_size is not None and prev_ball_size <= 25) or
020732:                 bulb_size <= 25
020733:             )
020734:             soft_upper_contact_recover = (
020735:                 not allow_inactive and
020736:                 contact_reacquire_bounds is not None and
020737:                 self.ball_center is not None and
020738:                 self.last_motion is not None and
020739:                 self.last_motion.get('dy', 0.0) < -2.0 and
020740:                 self.ground_bounce_count > 0 and
020741:                 small_upper_flight_candidate and
020742:                 not accepted_top_return_reentry and not accepted_back_return_reentry
020743:             )
020744:             if soft_upper_contact_recover:
020745:                 soft_recover_applied = False
020746:                 contact_reference = self.ball_center
020747:                 soft_recover = self._recover_contact_phase_ball(
020748:                     frame, contact_reference, frame_gray,
020749:                     predicted_point=None,
020750:                     max_prev_speed=45.0,
020751:                     upper_stuck_mode=True,
020752:                 )
020753:                 if soft_recover is not None:
020754:                     trusted_predicted_point = (
020755:                         int(self.ball_center[0] + self.last_motion.get('dx', 0.0)),
020756:                         int(self.ball_center[1] + self.last_motion.get('dy', 0.0)),
020757:                     )
020758:                     current_predicted_distance = float('inf')
020759:                     recovered_predicted_distance = float('inf')
```

## Direction change candidate around source line 21335
```python
021205:                         bounce_dist <= max(20.0, ground_bounce_context['min_launch_dist'] * 2.4) and
021206:                         expected_distance <= max(32.0, ground_bounce_context['expected_cap']) and
021207:                         candidate_not_tiny and
021208:                         (motion_max >= 70.0 or motion_mean >= 12.0)
021209:                     )
021210:                     if upper_soft_ground_bounce_candidate or upper_post_miss_bounce_seed:
021211:                         ground_bounce_candidate = True
021212:                         change_detected = True
021213:                         if upper_post_miss_bounce_seed:
021214:                             racket_mean = (
021215:                                 post_miss_racket_debug.get('mean', 0.0)
021216:                                 if post_miss_racket_debug is not None else 0.0
021217:                             )
021218:                             racket_max = (
021219:                                 post_miss_racket_debug.get('max', 0.0)
021220:                                 if post_miss_racket_debug is not None else 0.0
021221:                             )
021222:                             print(
021223:                                 f"Frame {self.frame_count}: [UPPER BOUNCE SEED] accepting close post-miss "
021224:                                 f"candidate ({cx},{cy}) from origin=({origin_x},{origin_y}) "
021225:                                 f"motion={motion_mean:.1f}/{motion_max:.1f} "
021226:                                 f"racket_below={racket_mean:.1f}/{racket_max:.1f}"
021227:                             )
021228:                 if predicted_point is not None and not predicted_turn_candidate:
021229:                     predicted_path_distance = math.hypot(cx - predicted_point[0], cy - predicted_point[1])
021230:                     predicted_path_cap = max(70, int(frame_width * 0.02))
021231:                     smooth_angle_continuation = angle_jump <= 30
021232:                     moderate_speed_drop = speed_ratio >= 0.30
021233:                     predicted_continuation_candidate = (
021234:                         smooth_angle_continuation and
021235:                         moderate_speed_drop and
021236:                         predicted_path_distance <= predicted_path_cap
021237:                     )
021238:                     if (not predicted_continuation_candidate and contact_reacquire_bounds is not None and
021239:                             self.ground_bounce_count > 0 and self.last_motion is not None and
021240:                             self.last_motion.get('dy', 0.0) < -2.0 and prev_ball_size is not None and
021241:                             prev_ball_size <= 12 and bulb_size <= 12):
021242:                         bounce_soft_angle = angle_jump <= 40
021243:                         bounce_soft_speed = speed_ratio >= 0.20
021244:                         bounce_soft_cap = max(110, int(frame_width * 0.03))
021245:                         predicted_continuation_candidate = (
021246:                             bounce_soft_angle and
021247:                             bounce_soft_speed and
021248:                             predicted_path_distance <= bounce_soft_cap
021249:                         )
021250:                 if upper_far_player_escape_override:
021251:                     predicted_continuation_candidate = True
021252:                 recent_bounce_reversal_candidate = False
021253:                 recent_bounce_continue_needed = self._recent_return_bounce_recover_active(window_frames=2)
021254:                 if recent_bounce_continue_needed:
021255:                     recent_bounce_pred_distance = (
021256:                         math.hypot(cx - predicted_point[0], cy - predicted_point[1])
021257:                         if predicted_point is not None else velocity
021258:                     )
021259:                     recent_bounce_regular_candidate = (
021260:                         recent_regular_single is not None and
021261:                         (best_source in ("regular", "primary") or current_filter_key in ("regular_court", "primary"))
021262:                     )
021263:                     recent_bounce_reversal_candidate = (
021264:                         not recent_bounce_regular_candidate and
021265:                         velocity >= 18.0 and
021266:                         dy <= -18.0 and
021267:                         recent_bounce_pred_distance <= max(90, int(frame_width * 0.03)) and
021268:                         (motion_mean >= 10.0 or motion_max >= 60.0) and
021269:                         bulb_size >= max(4.0, min(max(float(prev_ball_size or 0.0), 4.0) * 0.55, 18.0))
021270:                     )
021271:                     recent_bounce_continue_needed = (
021272:                         not recent_bounce_reversal_candidate and
021273:                         not recent_bounce_regular_candidate and
021274:                         (velocity <= 16.0 or abs(dy) <= 12.0 or (motion_mean < 18.0 and motion_max < 90.0))
021275:                     )
021276:                 else:
021277:                     recent_bounce_regular_candidate = False
021278: 
021279:                 serve_bounce_in_event = self._serve_bounce_in_from_turn_candidate(
021280:                     (cx, cy), frame, dx, dy, angle_jump, velocity
021281:                 )
021282:                 serve_bounce_out_event = self._serve_bounce_out_from_turn_candidate(
021283:                     (cx, cy), frame, dx, dy, angle_jump, velocity
021284:                 )
021285:                 if near_camera_regular_ball_override:
021286:                     # The selected point is a regular/Alt2 airborne ball, not
021287:                     # the near-player racket blob.  The service-box geometry
021288:                     # must not be evaluated against the stale pre-contact
021289:                     # marker on this frame; let subsequent frames establish a
021290:                     # real bounce or out-of-court event.
021291:                     serve_bounce_in_event = None
021292:                     serve_bounce_out_event = None
021293:                     print(
021294:                         f"Frame {self.frame_count}: [AIRBORNE REGULAR OVERRIDE] "
021295:                         "deferring serve-bounce classification"
021296:                     )
021297:                 impact_event = None
021298:                 if ((hold_change_detected or upper_focus_loss_guard_candidate)
021299:                         and not predicted_turn_candidate and not predicted_continuation_candidate
021300:                         and not lower_contact_launch_candidate and not large_lower_launch_candidate
021301:                         and not upper_racket_contact_turn_candidate
021302:                         and not upper_slow_arc_candidate and not ground_bounce_candidate
021303:                         and serve_bounce_in_event is None
021304:                         and serve_bounce_out_event is None
021305:                         and not recent_bounce_reversal_candidate and not recent_bounce_regular_candidate
021306:                         and not night_top_band_continuation_candidate):
021307:                     self.direction_change_streak += 1
021308:                     max_hold = 1 if immediate_lower_racket_contact_turn_candidate else (
021309:                         2 if (near_net or lower_racket_contact_turn_candidate) else 3
021310:                     )
021311:                     if upper_focus_loss_guard_candidate:
021312:                         max_hold = max(max_hold, 3)
021313:                         self.direction_change_streak = min(self.direction_change_streak, max_hold - 1)
021314:                     if self.direction_change_streak < max_hold:
021315:                         if recent_bounce_continue_needed:
021316:                             recent_bounce_continue = self._retrack_recent_return_bounce_continue(
021317:                                 frame, frame_gray, predicted_point=predicted_point
021318:                             )
021319:                             if recent_bounce_continue is not None:
021320:                                 self._register_ground_bounce_from_context(
021321:                                     ground_bounce_context, frame, source_label="recent bounce continue"
021322:                                 )
021323:                                 prev_pos = self.ball_center
021324:                                 new_pos = recent_bounce_continue['pos']
021325:                                 self.ball_center = new_pos
021326:                                 self.ball_hsv = recent_bounce_continue['hsv']
021327:                                 self.ball_size = recent_bounce_continue['area']
021328:                                 self._update_recovered_motion(prev_pos, new_pos)
021329:                                 self._activate_regular_hsv()
021330:                                 self._recent_return_bounce_recover_frame = self.frame_count
021331:                                 self.direction_change_streak = 0
021332:                                 self.stuck_frame_count = 0
021333:                                 print(f"Frame {self.frame_count}: [RECENT RETURN BOUNCE CONTINUE] Ball at {new_pos}")
021334:                                 return self.ball_center
021335:                         print(f"Frame {self.frame_count}: Direction change candidate (holding {self.direction_change_streak}/{max_hold}) angle_jump={angle_jump:.1f}")
021336:                         self.stuck_frame_count += 1
021337:                         return self.ball_center
021338:                     if lower_racket_contact_turn_candidate:
021339:                         if self.ground_bounce_count > 0:
021340:                             print(f"Frame {self.frame_count}: Resetting bounce count after lower-racket contact turn")
021341:                         self.ground_bounce_count = 0
021342:                         self.last_ground_bounce_frame = -1000000
021343:                         contact_origin = self.ball_center
021344:                         contact_label = (
021345:                             'mid-court racket rebound'
021346:                             if mid_court_racket_rebound_candidate
021347:                             else 'racket contact'
021348:                         )
021349:                         rally_ref_cap = 340.0 if near_camera_large_turn_override else 140.0
021350:                         contact_grace_frames = (
021351:                             5 if immediate_lower_racket_contact_turn_candidate else 3
021352:                         )
021353:                         self._rally_contact_grace_frames = max(
021354:                             getattr(self, '_rally_contact_grace_frames', 0),
021355:                             contact_grace_frames,
021356:                         )
021357:                         if near_camera_large_turn_override:
021358:                             self._rally_contact_large_ref_frames = max(
021359:                                 getattr(self, '_rally_contact_large_ref_frames', 0),
021360:                                 3,
021361:                             )
021362:                         self._rally_contact_ref_size = max(40.0, min(float(bulb_size), rally_ref_cap))
021363:                         self._rally_contact_origin = contact_origin
021364:                         self._rally_contact_expected = (cx, cy)
021365:                         self._rally_contact_progress = math.hypot(
021366:                             cx - contact_origin[0],
021367:                             cy - contact_origin[1],
021368:                         )
021369:                         impact_event = {
021370:                             'kind': 'racket_contact',
021371:                             'point': contact_origin,
021372:                             'label': contact_label,
021373:                         }
021374:                         print(f"Frame {self.frame_count}: Allowing {contact_label}")
021375:                         self._record_racket_contact(impact_event['point'], label=impact_event['label'])
021376:                         self._add_impact_marker(
021377:                             impact_event['point'],
021378:                             kind=impact_event['kind'],
021379:                             ttl=7,
021380:                             label=impact_event['label'],
021381:                         )
021382:                         self.direction_change_streak = 0
021383:                 else:
021384:                     if change_detected and upper_racket_contact_turn_candidate:
021385:                         if self.ground_bounce_count > 0:
021386:                             print(f"Frame {self.frame_count}: Resetting bounce count after upper-racket contact turn")
021387:                         self.ground_bounce_count = 0
021388:                         self.last_ground_bounce_frame = -1000000
021389:                         contact_origin = self.ball_center
021390:                         self._rally_contact_grace_frames = max(
021391:                             getattr(self, '_rally_contact_grace_frames', 0),
021392:                             3,
021393:                         )
021394:                         self._rally_contact_ref_size = max(12.0, min(float(bulb_size), 60.0))
021395:                         self._rally_contact_origin = contact_origin
021396:                         self._rally_contact_expected = (cx, cy)
021397:                         self._rally_contact_progress = math.hypot(
021398:                             cx - contact_origin[0],
021399:                             cy - contact_origin[1],
021400:                         )
021401:                         impact_event = {
021402:                             'kind': 'racket_contact',
021403:                             'point': contact_origin,
021404:                             'label': 'upper racket contact',
021405:                         }
021406:                         print(f"Frame {self.frame_count}: Allowing upper-racket contact turn")
021407:                     if (change_detected and not ground_bounce_candidate and not predicted_turn_candidate
021408:                             and not predicted_continuation_candidate and not upper_slow_arc_candidate
021409:                             and serve_bounce_out_event is None
021410:                             and not recent_bounce_reversal_candidate and not recent_bounce_regular_candidate):
021411:                         strong_x_reversal = (self.last_motion['dx'] * dx) < -12 if self.last_motion is not None else False
021412:                         contact_like_shot_change = (
021413:                             lower_contact_launch_candidate or
021414:                             large_lower_launch_candidate
021415:                         )
021416:                         direction_only_shot_change = (
021417:                             strong_x_reversal or
021418:                             angle_jump >= 120 or
021419:                             speed_ratio > 1.8
021420:                         )
021421:                         if contact_like_shot_change or direction_only_shot_change:
021422:                             if self.ground_bounce_count > 0:
021423:                                 if contact_like_shot_change:
021424:                                     if (
021425:                                         large_lower_launch_candidate and
021426:                                         self.ball_center is not None and
021427:                                         self.ball_center[1] >= int(frame_height * 0.50)
021428:                                     ):
021429:                                         self._late_contact_prior_bounce_count = max(
021430:                                             int(getattr(self, '_late_contact_prior_bounce_count', 0)),
021431:                                             int(self.ground_bounce_count),
021432:                                         )
021433:                                         self._late_contact_prior_bounce_until_frame = max(
021434:                                             int(getattr(self, '_late_contact_prior_bounce_until_frame', -1000000)),
021435:                                             self.frame_count + 90,
021436:                                         )
021437:                                         print(
021438:                                             f"Frame {self.frame_count}: Preserving prior bounce count "
021439:                                             f"{self._late_contact_prior_bounce_count} through late lower launch"
021440:                                         )
021441:                                     print(f"Frame {self.frame_count}: Resetting bounce count after non-bounce shot change")
021442:                                     self.ground_bounce_count = 0
021443:                                     self.last_ground_bounce_frame = -1000000
021444:                                 else:
021445:                                     self._direction_only_bounce_preserve_frame = self.frame_count
021446:                                     print(
021447:                                         f"Frame {self.frame_count}: Preserving bounce count through "
021448:                                         f"direction-only shot change"
021449:                                     )
021450:                     if change_detected and serve_bounce_in_event is not None:
021451:                         self._commit_serve_bounce_in_event(serve_bounce_in_event, frame)
021452:                     elif change_detected and serve_bounce_out_event is not None:
021453:                         bounce_point = serve_bounce_out_event['point']
021454:                         reason = serve_bounce_out_event['reason']
021455:                         self.ground_bounce_count += 1
021456:                         self.last_ground_bounce_frame = self.frame_count
021457:                         self._pending_rally_end_reason = reason
021458:                         self._pending_rally_end_frame = self.frame_count
021459:                         self._awaiting_serve_bounce = False
021460:                         self._add_impact_marker(
021461:                             bounce_point,
021462:                             kind="serve_bounce",
021463:                             color=(0, 0, 255),
021464:                             label=reason,
021465:                         )
021466:                         print(f"Frame {self.frame_count}: Ground bounce #{self.ground_bounce_count} detected (serve box jump)")
021467:                         print(f"Frame {self.frame_count}: [BOUNCE OUT] {reason} at {bounce_point}")
021468:                         print(f"Frame {self.frame_count}: Allowing serve-bounce-out continuation")
021469:                     elif change_detected and predicted_turn_candidate:
021470:                         print(f"Frame {self.frame_count}: Allowing upper-flight turn near predicted path")
021471:                     elif change_detected and predicted_continuation_candidate:
021472:                         if upper_far_player_escape_override:
021473:                             print(f"Frame {self.frame_count}: Allowing upper-far player ball correction")
021474:                         else:
021475:                             print(f"Frame {self.frame_count}: Allowing predicted-path continuation after speed drop")
021476:                     elif change_detected and upper_slow_arc_candidate:
021477:                         print(f"Frame {self.frame_count}: Allowing upper slow-arc continuation")
021478:                     elif change_detected and recent_bounce_regular_candidate:
021479:                         self._register_ground_bounce_from_context(
021480:                             ground_bounce_context, frame, source_label="recent bounce regular"
021481:                         )
021482:                         self._recent_return_bounce_recover_frame = self.frame_count
021483:                         print(f"Frame {self.frame_count}: Allowing recent bounce regular-candidate continuation")
021484:                     elif change_detected and recent_bounce_reversal_candidate:
```

