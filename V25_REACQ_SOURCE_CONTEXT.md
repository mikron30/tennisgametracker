# Exact V20/V25 player-reacquisition source context

Generated from `interactive_ball_analyzer.py` on this branch for V25 diagnosis.

## Lines 19486-19915
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
019695:                 # V28: a moving player/racket fragment can satisfy the generic
019696:                 # strong-motion escape even when it is a large, prediction-conflicting
019697:                 # reacquisition jump.  The reviewed early rally does this at f176:
019698:                 # (709,1558), area 249, S=42, jump 301px, pred error 153px, while
019699:                 # the real yellow ball is visibly elsewhere.  Do not weaken the
019700:                 # normal strong-motion path.  Only override when the selected
019701:                 # candidate is simultaneously large-jump, prediction-conflicting,
019702:                 # large-area and low-saturation, and the existing visible-ball
019703:                 # detector independently finds a more saturated moving ball well
019704:                 # away from it in the same frame.
019705:                 try:
019706:                     player_reacq_selected_h = int(hsv_values[0])
019707:                     player_reacq_selected_s = int(hsv_values[1])
019708:                     player_reacq_selected_v = int(hsv_values[2])
019709:                     player_reacq_selected_area = float(bulb_size or 0.0)
019710:                 except (TypeError, ValueError, IndexError):
019711:                     player_reacq_selected_h = -1
019712:                     player_reacq_selected_s = -1
019713:                     player_reacq_selected_v = -1
019714:                     player_reacq_selected_area = 0.0
019715: 
019716:                 player_reacq_visible_conflict_probe = (
019717:                     player_reacq_guard_active and
019718:                     frame is not None and
019719:                     actual_distance >= max(240.0, float(frame.shape[1]) * 0.06) and
019720:                     selected_predicted_distance is not None and
019721:                     selected_predicted_distance >= max(110.0, float(frame.shape[1]) * 0.028) and
019722:                     player_reacq_selected_area >= 120.0 and
019723:                     0 <= player_reacq_selected_s <= 60
019724:                 )
019725:                 if player_reacq_visible_conflict_probe:
019726:                     visible_ball = None
019727:                     try:
019728:                         visible_ball = self._find_night_visible_ball_candidate(
019729:                             frame, frame_gray
019730:                         )
019731:                     except Exception:
019732:                         visible_ball = None
019733: 
019734:                     if visible_ball is not None:
019735:                         try:
019736:                             visible_pos = tuple(visible_ball.get('pos', ()))
019737:                             visible_hsv = visible_ball.get('hsv')
019738:                             visible_s = (
019739:                                 int(visible_hsv[1])
019740:                                 if visible_hsv is not None and len(visible_hsv) >= 3
019741:                                 else -1
019742:                             )
019743:                             visible_motion_mean = float(
019744:                                 visible_ball.get('motion_mean', 0.0) or 0.0
019745:                             )
019746:                             visible_motion_max = float(
019747:                                 visible_ball.get('motion_max', 0.0) or 0.0
019748:                             )
019749:                             visible_conflict_distance = (
019750:                                 math.hypot(
019751:                                     float(visible_pos[0]) - float(cx),
019752:                                     float(visible_pos[1]) - float(cy),
019753:                                 )
019754:                                 if len(visible_pos) >= 2 else 0.0
019755:                             )
019756:                         except (TypeError, ValueError, IndexError):
019757:                             visible_pos = ()
019758:                             visible_s = -1
019759:                             visible_motion_mean = 0.0
019760:                             visible_motion_max = 0.0
019761:                             visible_conflict_distance = 0.0
019762: 
019763:                         visible_conflict_limit = max(100.0, float(frame.shape[1]) * 0.025)
019764:                         visible_independent = (
019765:                             visible_conflict_distance >= visible_conflict_limit and
019766:                             visible_s >= player_reacq_selected_s + 15 and
019767:                             (
019768:                                 visible_motion_mean >= 5.0 or
019769:                                 visible_motion_max >= 35.0
019770:                             )
019771:                         )
019772:                         if visible_independent:
019773:                             print(
019774:                                 f"Frame {self.frame_count}: "
019775:                                 f"[PLAYER-REACQ STRONG-MOTION VISIBLE-CONFLICT REANCHOR] "
019776:                                 f"rejected=({cx},{cy}) "
019777:                                 f"hsv=({player_reacq_selected_h},"
019778:                                 f"{player_reacq_selected_s},"
019779:                                 f"{player_reacq_selected_v}) "
019780:                                 f"area={player_reacq_selected_area:.1f}px "
019781:                                 f"jump={actual_distance:.1f}px "
019782:                                 f"pred_dist={selected_predicted_distance:.1f}px "
019783:                                 f"visible={visible_pos} visible_s={visible_s} "
019784:                                 f"motion={visible_motion_mean:.1f}/{visible_motion_max:.1f} "
019785:                                 f"conflict={visible_conflict_distance:.1f}px"
019786:                             )
019787:                             recovery_candidate = dict(visible_ball)
019788:                             recovery_candidate['recovery_label'] = (
019789:                                 'PLAYER-REACQ STRONG-MOTION VISIBLE REANCHOR'
019790:                             )
019791:                             return self._commit_night_visible_ball_recovery(
019792:                                 recovery_candidate, frame
019793:                             )
019794: 
019795:                 if (
019796:                         player_reacq_guard_active and
019797:                         (
019798:                             player_reacq_strong_motion or
019799:                             player_reacq_prediction_close or
019800:                             player_reacq_close_to_anchor
019801:                         ) and
019802:                         self._player_point_zone((cx, cy)) is None
019803:                 ):
019804:                     print(
019805:                         f"Frame {self.frame_count}: [PLAYER-REACQ MOTION CONFIRMED] "
019806:                         f"clearing guard at ({cx},{cy}) motion={motion_mean:.1f}/{motion_max:.1f} "
019807:                         f"jump={actual_distance:.1f}px pred_dist={selected_predicted_distance}"
019808:                     )
019809:                     self._player_reacq_protect_until_frame = -1
019810: 
019811:                 relaxed_contact_min_y = self._contact_reacquire_min_y(
019812:                     contact_reacquire_bounds,
019813:                     selected_predicted_distance,
019814:                     motion_mean,
019815:                     motion_max,
019816:                 ) if contact_reacquire_bounds is not None else None
019817:                 selected_area = selected_area_for_guard
019818:                 player_reacq_static = self._player_reacq_static_candidate(
019819:                     {
019820:                         'pos': (cx, cy),
019821:                         'area': selected_area,
019822:                         'motion_mean': motion_mean,
019823:                         'motion_max': motion_max,
019824:                     },
019825:                     predicted_point=predicted_point,
019826:                 )
019827:                 # The regular contour search can still run after motion
019828:                 # reacquisition returns no candidate.  Apply the same
019829:                 # adjacent-court guard to that path; otherwise a tiny static
019830:                 # blob outside the singles sideline can reset stuck_frame_count
019831:                 # and become the point-ending position.
019832:                 if not player_reacq_static and int(
019833:                         getattr(self, '_player_reacq_protect_until_frame', -1)
019834:                     ) >= self.frame_count:
019835:                     try:
019836:                         outside_court, outside_side, left_x, right_x = self._point_outside_singles_sidelines(
019837:                             (cx, cy), frame
019838:                         )
019839:                     except Exception:
019840:                         outside_court, outside_side, left_x, right_x = False, None, None, None
019841:                     if outside_court:
019842:                         sideline_depth = (
019843:                             (float(left_x) - cx) if outside_side == 'left'
019844:                             else (cx - float(right_x))
019845:                         )
019846:                         if sideline_depth > 35.0 or (motion_mean < 6.0 and motion_max < 25.0):
019847:                             player_reacq_static = True
019848:                 if not player_reacq_static and int(
019849:                         getattr(self, '_player_reacq_protect_until_frame', -1)
019850:                     ) >= self.frame_count:
019851:                     tracker = getattr(self, 'player_tracker', None)
019852:                     try:
019853:                         outside_main_court = (
019854:                             tracker is not None and
019855:                             hasattr(tracker, '_center_is_in_court') and
019856:                             not tracker._center_is_in_court((cx, cy), frame_width=frame.shape[1])
019857:                         )
019858:                     except Exception:
019859:                         outside_main_court = False
019860:                     if outside_main_court:
019861:                         player_reacq_static = True
019862:                 if player_reacq_static:
019863:                     self._record_rejected_contour_debug(
019864:                         best_contour,
019865:                         x1,
019866:                         y1,
019867:                         cx,
019868:                         cy,
019869:                         selected_area,
019870:                         "player-reacq distant static blob",
019871:                         source=best_source,
019872:                     )
019873:                     self.stuck_frame_count = max(
019874:                         int(getattr(self, 'stuck_frame_count', 0)) + 1, 1
019875:                     )
019876:                     print(
019877:                         f"Frame {self.frame_count}: [PLAYER-REACQ STATIC REJECT] "
019878:                         f"holding {self.ball_center} instead of ({cx},{cy}) "
019879:                         f"motion={motion_mean:.1f}/{motion_max:.1f}"
019880:                     )
019881:                     visible_ball = self._find_night_visible_ball_candidate(frame, frame_gray)
019882:                     if visible_ball is not None:
019883:                         print(
019884:                             f"Frame {self.frame_count}: [PLAYER-REACQ VISIBLE BALL] "
019885:                             f"using {visible_ball['pos']} area={visible_ball['area']:.1f}px "
019886:                             f"instead of static candidate ({cx},{cy})"
019887:                         )
019888:                         return self._commit_night_visible_ball_recovery(visible_ball, frame)
019889:                     return self.ball_center
019890:                 if top_return_search_context:
019891:                     if (
019892:                         getattr(self, '_top_return_mode', 'edge') in ('upper_side', 'upper_racket') and
019893:                         best_source not in ('primary', 'regular', 'alt')
019894:                     ):
019895:                         top_return_ok = False
019896:                         top_return_reason = f"top-return source {best_source} not allowed for upper-side reentry"
019897:                     else:
019898:                         top_return_ok, top_return_reason = self._top_return_reentry_ok(
019899:                             (cx, cy), selected_area, motion_mean, motion_max, frame.shape
019900:                         )
019901:                     if not top_return_ok:
019902:                         motion_reentry = self._find_top_return_motion_reentry(frame, frame_gray)
019903:                         if motion_reentry is not None:
019904:                             return self._commit_top_return_reentry(
019905:                                 motion_reentry, "TOP-RETURN MOTION REENTRY"
019906:                             )
019907:                         self._record_rejected_contour_debug(
019908:                             best_contour,
019909:                             x1,
019910:                             y1,
019911:                             cx,
019912:                             cy,
019913:                             selected_area,
019914:                             top_return_reason,
019915:                             source=best_source,
```

