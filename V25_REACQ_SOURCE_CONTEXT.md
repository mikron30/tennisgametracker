# Exact V20/V25 player-reacquisition source context

Generated from `interactive_ball_analyzer.py` on this branch for V25 diagnosis.

## Lines 19486-19762
```python
019486:                             selected_area_for_guard,
019487:                             guard_motion_mean,
019488:                             guard_motion_max,
019489:                             best_source,
019490:                             frame.shape,
019491:                         )
019492:                         if not allow_confirmed_top_return:
019493:                             self._record_rejected_contour_debug(
019494:                                 best_contour,
019495:                                 x1,
019496:                                 y1,
019497:                                 cx,
019498:                                 cy,
019499:                                 selected_area_for_guard,
019500:                                 f"player-reacq jump {actual_distance:.1f}px > {max_player_reacq_jump:.1f}px",
019501:                                 source=best_source,
019502:                             )
019503:                             self.stuck_frame_count = max(
019504:                                 int(getattr(self, 'stuck_frame_count', 0)) + 1,
019505:                                 5,
019506:                             )
019507:                             print(
019508:                                 f"Frame {self.frame_count}: [PLAYER-REACQ JUMP REJECT] "
019509:                                 f"holding {self.ball_center} instead of ({cx},{cy}) "
019510:                                 f"jump={actual_distance:.1f}px limit={max_player_reacq_jump:.1f}px"
019511:                             )
019512:                             return self.ball_center
019513:                         print(
019514:                             f"Frame {self.frame_count}: [TOP-RETURN JUMP ALLOW] "
019515:                             f"accepting strongly confirmed re-entry at ({cx},{cy}) "
019516:                             f"jump={actual_distance:.1f}px motion="
019517:                             f"{guard_motion_mean:.1f}/{guard_motion_max:.1f}"
019518:                         )
019519:                 
019520:                 frame_height, frame_width = frame.shape[:2]
019521:                 at_edge = (y_prev < edge_threshold or y_prev > frame_height - edge_threshold or
019522:                           x_prev < edge_threshold or x_prev > frame_width - edge_threshold)
019523:                 if (at_edge and actual_distance > jump_threshold and
019524:                         not top_return_search_context and
019525:                         not back_return_search_context):
019526:                     self._record_rejected_contour_debug(
019527:                         best_contour,
019528:                         x1,
019529:                         y1,
019530:                         cx,
019531:                         cy,
019532:                         cv2.contourArea(best_contour),
019533:                         f"edge jump {actual_distance:.1f}px from {self.ball_center}",
019534:                         source=best_source,
019535:                     )
019536:                     print(f"  DEBUG: [REJECTED] Candidate at ({cx},{cy}), distance={actual_distance:.1f}px")
019537:                     print(f"  DEBUG: Ball was at edge (y={y_prev}), closest match is {actual_distance:.1f}px away")
019538:                     print(f"  DEBUG: This is likely a FALSE POSITIVE - ball probably went off-screen")
019539:                     _verbose_debug_print(f"  DEBUG: KEEPING marker at last edge position: {self.ball_center}")
019540:                     print(f"  DEBUG: Will wait for ball to return...")
019541:                     return self.ball_center
019542: 
019543:                 selected_motion = (
019544:                     selected_motion_for_guard
019545:                     if selected_motion_for_guard is not None else
019546:                     self._candidate_motion_metrics(frame_gray, cx, cy)
019547:                 )
019548:                 motion_mean = selected_motion['mean'] if selected_motion is not None else 0.0
019549:                 motion_max = selected_motion['max'] if selected_motion is not None else 0.0
019550:                 player_reacq_guard_active = int(
019551:                     getattr(self, '_player_reacq_protect_until_frame', -1)
019552:                 ) >= self.frame_count
019553:                 selected_predicted_distance = (
019554:                     math.hypot(cx - predicted_point[0], cy - predicted_point[1])
019555:                     if predicted_point is not None else None
019556:                 )
019557: 
019558:                 # Do not let a distant, weakly moving background blob end the
019559:                 # protected player-contact reacquisition window. A real fast
019560:                 # ball may travel far, so large jumps are still allowed when
019561:                 # inter-frame motion is strong or the candidate agrees with
019562:                 # the predicted trajectory.
019563:                 player_reacq_strong_motion = (
019564:                     motion_mean >= 12.0 or motion_max >= 80.0
019565:                 )
019566:                 player_reacq_prediction_close = (
019567:                     selected_predicted_distance is not None and
019568:                     selected_predicted_distance <= 60.0
019569:                 )
019570:                 player_reacq_close_to_anchor = actual_distance <= 80.0
019571:                 player_reacq_weak_large_jump = (
019572:                     player_reacq_guard_active and
019573:                     actual_distance >= 150.0 and
019574:                     not player_reacq_strong_motion and
019575:                     not player_reacq_prediction_close
019576:                 )
019577:                 if player_reacq_weak_large_jump:
019578:                     self._record_rejected_contour_debug(
019579:                         best_contour,
019580:                         x1,
019581:                         y1,
019582:                         cx,
019583:                         cy,
019584:                         selected_area_for_guard,
019585:                         (
019586:                             f"player-reacq weak large jump {actual_distance:.1f}px "
019587:                             f"motion={motion_mean:.1f}/{motion_max:.1f}"
019588:                         ),
019589:                         source=best_source,
019590:                     )
019591:                     self.stuck_frame_count = max(
019592:                         int(getattr(self, 'stuck_frame_count', 0)) + 1,
019593:                         1,
019594:                     )
019595:                     print(
019596:                         f"Frame {self.frame_count}: [PLAYER-REACQ WEAK-JUMP REJECT] "
019597:                         f"holding {self.ball_center} instead of ({cx},{cy}) "
019598:                         f"jump={actual_distance:.1f}px motion="
019599:                         f"{motion_mean:.1f}/{motion_max:.1f} "
019600:                         f"pred_dist={selected_predicted_distance}"
019601:                     )
019602:                     visible_ball = self._find_night_visible_ball_candidate(frame, frame_gray)
019603:                     if visible_ball is not None:
019604:                         visible_pos = tuple(visible_ball['pos'])
019605:                         visible_jump = math.hypot(
019606:                             float(visible_pos[0]) - float(self.ball_center[0]),
019607:                             float(visible_pos[1]) - float(self.ball_center[1]),
019608:                         )
019609:                         visible_mean = float(visible_ball.get('motion_mean', 0.0) or 0.0)
019610:                         visible_max = float(visible_ball.get('motion_max', 0.0) or 0.0)
019611:                         visible_pred_dist = (
019612:                             math.hypot(
019613:                                 float(visible_pos[0]) - float(predicted_point[0]),
019614:                                 float(visible_pos[1]) - float(predicted_point[1]),
019615:                             )
019616:                             if predicted_point is not None else None
019617:                         )
019618:                         visible_safe = (
019619:                             visible_jump <= 80.0 or
019620:                             visible_mean >= 12.0 or
019621:                             visible_max >= 80.0 or
019622:                             (visible_pred_dist is not None and visible_pred_dist <= 60.0)
019623:                         )
019624:                         if visible_safe:
019625:                             print(
019626:                                 f"Frame {self.frame_count}: [PLAYER-REACQ WEAK-JUMP VISIBLE BALL] "
019627:                                 f"using {visible_pos} instead; jump={visible_jump:.1f}px "
019628:                                 f"motion={visible_mean:.1f}/{visible_max:.1f}"
019629:                             )
019630:                             return self._commit_night_visible_ball_recovery(visible_ball, frame)
019631:                         print(
019632:                             f"Frame {self.frame_count}: [PLAYER-REACQ WEAK-JUMP FALLBACK REJECT] "
019633:                             f"ignoring visible candidate {visible_pos} jump={visible_jump:.1f}px "
019634:                             f"motion={visible_mean:.1f}/{visible_max:.1f}"
019635:                         )
019636:                     return self.ball_center
019637: 
019638:                 # Clearing the protection now needs either strong image motion
019639:                 # or geometric continuity. This preserves slow nearby tracks
019640:                 # while preventing a 300+ px weak-motion blob from becoming a
019641:                 # new anchor simply because it is outside the player box.
019642:                 if (
019643:                         player_reacq_guard_active and
019644:                         (
019645:                             player_reacq_strong_motion or
019646:                             player_reacq_prediction_close or
019647:                             player_reacq_close_to_anchor
019648:                         ) and
019649:                         self._player_point_zone((cx, cy)) is None
019650:                 ):
019651:                     print(
019652:                         f"Frame {self.frame_count}: [PLAYER-REACQ MOTION CONFIRMED] "
019653:                         f"clearing guard at ({cx},{cy}) motion={motion_mean:.1f}/{motion_max:.1f} "
019654:                         f"jump={actual_distance:.1f}px pred_dist={selected_predicted_distance}"
019655:                     )
019656:                     self._player_reacq_protect_until_frame = -1
019657: 
019658:                 relaxed_contact_min_y = self._contact_reacquire_min_y(
019659:                     contact_reacquire_bounds,
019660:                     selected_predicted_distance,
019661:                     motion_mean,
019662:                     motion_max,
019663:                 ) if contact_reacquire_bounds is not None else None
019664:                 selected_area = selected_area_for_guard
019665:                 player_reacq_static = self._player_reacq_static_candidate(
019666:                     {
019667:                         'pos': (cx, cy),
019668:                         'area': selected_area,
019669:                         'motion_mean': motion_mean,
019670:                         'motion_max': motion_max,
019671:                     },
019672:                     predicted_point=predicted_point,
019673:                 )
019674:                 # The regular contour search can still run after motion
019675:                 # reacquisition returns no candidate.  Apply the same
019676:                 # adjacent-court guard to that path; otherwise a tiny static
019677:                 # blob outside the singles sideline can reset stuck_frame_count
019678:                 # and become the point-ending position.
019679:                 if not player_reacq_static and int(
019680:                         getattr(self, '_player_reacq_protect_until_frame', -1)
019681:                     ) >= self.frame_count:
019682:                     try:
019683:                         outside_court, outside_side, left_x, right_x = self._point_outside_singles_sidelines(
019684:                             (cx, cy), frame
019685:                         )
019686:                     except Exception:
019687:                         outside_court, outside_side, left_x, right_x = False, None, None, None
019688:                     if outside_court:
019689:                         sideline_depth = (
019690:                             (float(left_x) - cx) if outside_side == 'left'
019691:                             else (cx - float(right_x))
019692:                         )
019693:                         if sideline_depth > 35.0 or (motion_mean < 6.0 and motion_max < 25.0):
019694:                             player_reacq_static = True
019695:                 if not player_reacq_static and int(
019696:                         getattr(self, '_player_reacq_protect_until_frame', -1)
019697:                     ) >= self.frame_count:
019698:                     tracker = getattr(self, 'player_tracker', None)
019699:                     try:
019700:                         outside_main_court = (
019701:                             tracker is not None and
019702:                             hasattr(tracker, '_center_is_in_court') and
019703:                             not tracker._center_is_in_court((cx, cy), frame_width=frame.shape[1])
019704:                         )
019705:                     except Exception:
019706:                         outside_main_court = False
019707:                     if outside_main_court:
019708:                         player_reacq_static = True
019709:                 if player_reacq_static:
019710:                     self._record_rejected_contour_debug(
019711:                         best_contour,
019712:                         x1,
019713:                         y1,
019714:                         cx,
019715:                         cy,
019716:                         selected_area,
019717:                         "player-reacq distant static blob",
019718:                         source=best_source,
019719:                     )
019720:                     self.stuck_frame_count = max(
019721:                         int(getattr(self, 'stuck_frame_count', 0)) + 1, 1
019722:                     )
019723:                     print(
019724:                         f"Frame {self.frame_count}: [PLAYER-REACQ STATIC REJECT] "
019725:                         f"holding {self.ball_center} instead of ({cx},{cy}) "
019726:                         f"motion={motion_mean:.1f}/{motion_max:.1f}"
019727:                     )
019728:                     visible_ball = self._find_night_visible_ball_candidate(frame, frame_gray)
019729:                     if visible_ball is not None:
019730:                         print(
019731:                             f"Frame {self.frame_count}: [PLAYER-REACQ VISIBLE BALL] "
019732:                             f"using {visible_ball['pos']} area={visible_ball['area']:.1f}px "
019733:                             f"instead of static candidate ({cx},{cy})"
019734:                         )
019735:                         return self._commit_night_visible_ball_recovery(visible_ball, frame)
019736:                     return self.ball_center
019737:                 if top_return_search_context:
019738:                     if (
019739:                         getattr(self, '_top_return_mode', 'edge') in ('upper_side', 'upper_racket') and
019740:                         best_source not in ('primary', 'regular', 'alt')
019741:                     ):
019742:                         top_return_ok = False
019743:                         top_return_reason = f"top-return source {best_source} not allowed for upper-side reentry"
019744:                     else:
019745:                         top_return_ok, top_return_reason = self._top_return_reentry_ok(
019746:                             (cx, cy), selected_area, motion_mean, motion_max, frame.shape
019747:                         )
019748:                     if not top_return_ok:
019749:                         motion_reentry = self._find_top_return_motion_reentry(frame, frame_gray)
019750:                         if motion_reentry is not None:
019751:                             return self._commit_top_return_reentry(
019752:                                 motion_reentry, "TOP-RETURN MOTION REENTRY"
019753:                             )
019754:                         self._record_rejected_contour_debug(
019755:                             best_contour,
019756:                             x1,
019757:                             y1,
019758:                             cx,
019759:                             cy,
019760:                             selected_area,
019761:                             top_return_reason,
019762:                             source=best_source,
```

