# Exact V20/V25 player-reacquisition source context

Generated from `interactive_ball_analyzer.py` on this branch for V25 diagnosis.

## Lines 19486-19815
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
019638:                 # V25: strong image motion by itself must not clear the
019639:                 # player-contact reacquisition guard for an extreme jump that
019640:                 # strongly contradicts the predicted ball trajectory. The
019641:                 # reviewed V20 f175 failure jumped 388px from the recovered
019642:                 # ball while landing about 560px from prediction; moving-player
019643:                 # image motion made that false fragment look valid. Hold the
019644:                 # last trusted ball for one more frame instead. Specialized
019645:                 # top/back-return paths keep their own long-jump validation.
019646:                 player_reacq_extreme_jump_limit = max(
019647:                     240.0, float(frame_width) * 0.065
019648:                 )
019649:                 player_reacq_extreme_pred_limit = max(
019650:                     150.0, float(frame_width) * 0.045
019651:                 )
019652:                 player_reacq_extreme_prediction_conflict = (
019653:                     player_reacq_guard_active and
019654:                     actual_distance >= player_reacq_extreme_jump_limit and
019655:                     selected_predicted_distance is not None and
019656:                     selected_predicted_distance >= player_reacq_extreme_pred_limit and
019657:                     not top_return_search_context and
019658:                     not back_return_search_context
019659:                 )
019660:                 if player_reacq_extreme_prediction_conflict:
019661:                     self._record_rejected_contour_debug(
019662:                         best_contour,
019663:                         x1,
019664:                         y1,
019665:                         cx,
019666:                         cy,
019667:                         selected_area_for_guard,
019668:                         (
019669:                             f"player-reacq extreme prediction conflict "
019670:                             f"jump={actual_distance:.1f}px "
019671:                             f"pred_dist={selected_predicted_distance:.1f}px"
019672:                         ),
019673:                         source=best_source,
019674:                     )
019675:                     self.stuck_frame_count = max(
019676:                         int(getattr(self, 'stuck_frame_count', 0)) + 1,
019677:                         1,
019678:                     )
019679:                     print(
019680:                         f"Frame {self.frame_count}: "
019681:                         f"[PLAYER-REACQ EXTREME-PREDICTION REJECT] "
019682:                         f"holding {self.ball_center} instead of ({cx},{cy}) "
019683:                         f"jump={actual_distance:.1f}px "
019684:                         f"limit={player_reacq_extreme_jump_limit:.1f}px "
019685:                         f"pred_dist={selected_predicted_distance:.1f}px "
019686:                         f"pred_limit={player_reacq_extreme_pred_limit:.1f}px "
019687:                         f"motion={motion_mean:.1f}/{motion_max:.1f}"
019688:                     )
019689:                     return self.ball_center
019690: 
019691:                 # Clearing the protection now needs either strong image motion
019692:                 # or geometric continuity. This preserves slow nearby tracks
019693:                 # while preventing a 300+ px weak-motion blob from becoming a
019694:                 # new anchor simply because it is outside the player box.
019695:                 if (
019696:                         player_reacq_guard_active and
019697:                         (
019698:                             player_reacq_strong_motion or
019699:                             player_reacq_prediction_close or
019700:                             player_reacq_close_to_anchor
019701:                         ) and
019702:                         self._player_point_zone((cx, cy)) is None
019703:                 ):
019704:                     print(
019705:                         f"Frame {self.frame_count}: [PLAYER-REACQ MOTION CONFIRMED] "
019706:                         f"clearing guard at ({cx},{cy}) motion={motion_mean:.1f}/{motion_max:.1f} "
019707:                         f"jump={actual_distance:.1f}px pred_dist={selected_predicted_distance}"
019708:                     )
019709:                     self._player_reacq_protect_until_frame = -1
019710: 
019711:                 relaxed_contact_min_y = self._contact_reacquire_min_y(
019712:                     contact_reacquire_bounds,
019713:                     selected_predicted_distance,
019714:                     motion_mean,
019715:                     motion_max,
019716:                 ) if contact_reacquire_bounds is not None else None
019717:                 selected_area = selected_area_for_guard
019718:                 player_reacq_static = self._player_reacq_static_candidate(
019719:                     {
019720:                         'pos': (cx, cy),
019721:                         'area': selected_area,
019722:                         'motion_mean': motion_mean,
019723:                         'motion_max': motion_max,
019724:                     },
019725:                     predicted_point=predicted_point,
019726:                 )
019727:                 # The regular contour search can still run after motion
019728:                 # reacquisition returns no candidate.  Apply the same
019729:                 # adjacent-court guard to that path; otherwise a tiny static
019730:                 # blob outside the singles sideline can reset stuck_frame_count
019731:                 # and become the point-ending position.
019732:                 if not player_reacq_static and int(
019733:                         getattr(self, '_player_reacq_protect_until_frame', -1)
019734:                     ) >= self.frame_count:
019735:                     try:
019736:                         outside_court, outside_side, left_x, right_x = self._point_outside_singles_sidelines(
019737:                             (cx, cy), frame
019738:                         )
019739:                     except Exception:
019740:                         outside_court, outside_side, left_x, right_x = False, None, None, None
019741:                     if outside_court:
019742:                         sideline_depth = (
019743:                             (float(left_x) - cx) if outside_side == 'left'
019744:                             else (cx - float(right_x))
019745:                         )
019746:                         if sideline_depth > 35.0 or (motion_mean < 6.0 and motion_max < 25.0):
019747:                             player_reacq_static = True
019748:                 if not player_reacq_static and int(
019749:                         getattr(self, '_player_reacq_protect_until_frame', -1)
019750:                     ) >= self.frame_count:
019751:                     tracker = getattr(self, 'player_tracker', None)
019752:                     try:
019753:                         outside_main_court = (
019754:                             tracker is not None and
019755:                             hasattr(tracker, '_center_is_in_court') and
019756:                             not tracker._center_is_in_court((cx, cy), frame_width=frame.shape[1])
019757:                         )
019758:                     except Exception:
019759:                         outside_main_court = False
019760:                     if outside_main_court:
019761:                         player_reacq_static = True
019762:                 if player_reacq_static:
019763:                     self._record_rejected_contour_debug(
019764:                         best_contour,
019765:                         x1,
019766:                         y1,
019767:                         cx,
019768:                         cy,
019769:                         selected_area,
019770:                         "player-reacq distant static blob",
019771:                         source=best_source,
019772:                     )
019773:                     self.stuck_frame_count = max(
019774:                         int(getattr(self, 'stuck_frame_count', 0)) + 1, 1
019775:                     )
019776:                     print(
019777:                         f"Frame {self.frame_count}: [PLAYER-REACQ STATIC REJECT] "
019778:                         f"holding {self.ball_center} instead of ({cx},{cy}) "
019779:                         f"motion={motion_mean:.1f}/{motion_max:.1f}"
019780:                     )
019781:                     visible_ball = self._find_night_visible_ball_candidate(frame, frame_gray)
019782:                     if visible_ball is not None:
019783:                         print(
019784:                             f"Frame {self.frame_count}: [PLAYER-REACQ VISIBLE BALL] "
019785:                             f"using {visible_ball['pos']} area={visible_ball['area']:.1f}px "
019786:                             f"instead of static candidate ({cx},{cy})"
019787:                         )
019788:                         return self._commit_night_visible_ball_recovery(visible_ball, frame)
019789:                     return self.ball_center
019790:                 if top_return_search_context:
019791:                     if (
019792:                         getattr(self, '_top_return_mode', 'edge') in ('upper_side', 'upper_racket') and
019793:                         best_source not in ('primary', 'regular', 'alt')
019794:                     ):
019795:                         top_return_ok = False
019796:                         top_return_reason = f"top-return source {best_source} not allowed for upper-side reentry"
019797:                     else:
019798:                         top_return_ok, top_return_reason = self._top_return_reentry_ok(
019799:                             (cx, cy), selected_area, motion_mean, motion_max, frame.shape
019800:                         )
019801:                     if not top_return_ok:
019802:                         motion_reentry = self._find_top_return_motion_reentry(frame, frame_gray)
019803:                         if motion_reentry is not None:
019804:                             return self._commit_top_return_reentry(
019805:                                 motion_reentry, "TOP-RETURN MOTION REENTRY"
019806:                             )
019807:                         self._record_rejected_contour_debug(
019808:                             best_contour,
019809:                             x1,
019810:                             y1,
019811:                             cx,
019812:                             cy,
019813:                             selected_area,
019814:                             top_return_reason,
019815:                             source=best_source,
```

