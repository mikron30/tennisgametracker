# Exact V20/V25 player-reacquisition source context

Generated from `interactive_ball_analyzer.py` on this branch for V25 diagnosis.

## Lines 19486-19955
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
019705:                 # V29: this V28 probe executes before hsv_values/bulb_size are
019706:                 # guaranteed to be bound on every tracking path.  Prefer those
019707:                 # already-bound values when present, otherwise derive evidence
019708:                 # directly from the selected candidate location/contour.
019709:                 player_reacq_candidate_hsv = locals().get('hsv_values')
019710:                 player_reacq_candidate_area = locals().get('bulb_size')
019711:                 try:
019712:                     if (
019713:                             player_reacq_candidate_hsv is not None and
019714:                             len(player_reacq_candidate_hsv) >= 3
019715:                     ):
019716:                         player_reacq_selected_h = int(player_reacq_candidate_hsv[0])
019717:                         player_reacq_selected_s = int(player_reacq_candidate_hsv[1])
019718:                         player_reacq_selected_v = int(player_reacq_candidate_hsv[2])
019719:                     else:
019720:                         x0 = max(0, int(cx) - 2)
019721:                         x1 = min(frame.shape[1], int(cx) + 3)
019722:                         y0 = max(0, int(cy) - 2)
019723:                         y1 = min(frame.shape[0], int(cy) + 3)
019724:                         candidate_patch = frame[y0:y1, x0:x1]
019725:                         if not candidate_patch.size:
019726:                             raise ValueError('empty candidate patch')
019727:                         candidate_patch_hsv = cv2.cvtColor(
019728:                             candidate_patch, cv2.COLOR_BGR2HSV
019729:                         )
019730:                         player_reacq_selected_h = int(
019731:                             np.median(candidate_patch_hsv[:, :, 0])
019732:                         )
019733:                         player_reacq_selected_s = int(
019734:                             np.median(candidate_patch_hsv[:, :, 1])
019735:                         )
019736:                         player_reacq_selected_v = int(
019737:                             np.median(candidate_patch_hsv[:, :, 2])
019738:                         )
019739: 
019740:                     if player_reacq_candidate_area is not None:
019741:                         player_reacq_selected_area = float(
019742:                             player_reacq_candidate_area or 0.0
019743:                         )
019744:                     else:
019745:                         candidate_contour = locals().get('best_contour')
019746:                         player_reacq_selected_area = (
019747:                             float(cv2.contourArea(candidate_contour))
019748:                             if candidate_contour is not None else 0.0
019749:                         )
019750:                 except (TypeError, ValueError, IndexError, cv2.error):
019751:                     player_reacq_selected_h = -1
019752:                     player_reacq_selected_s = -1
019753:                     player_reacq_selected_v = -1
019754:                     player_reacq_selected_area = 0.0
019755: 
019756:                 player_reacq_visible_conflict_probe = (
019757:                     player_reacq_guard_active and
019758:                     frame is not None and
019759:                     actual_distance >= max(240.0, float(frame.shape[1]) * 0.06) and
019760:                     selected_predicted_distance is not None and
019761:                     selected_predicted_distance >= max(110.0, float(frame.shape[1]) * 0.028) and
019762:                     player_reacq_selected_area >= 120.0 and
019763:                     0 <= player_reacq_selected_s <= 60
019764:                 )
019765:                 if player_reacq_visible_conflict_probe:
019766:                     visible_ball = None
019767:                     try:
019768:                         visible_ball = self._find_night_visible_ball_candidate(
019769:                             frame, frame_gray
019770:                         )
019771:                     except Exception:
019772:                         visible_ball = None
019773: 
019774:                     if visible_ball is not None:
019775:                         try:
019776:                             visible_pos = tuple(visible_ball.get('pos', ()))
019777:                             visible_hsv = visible_ball.get('hsv')
019778:                             visible_s = (
019779:                                 int(visible_hsv[1])
019780:                                 if visible_hsv is not None and len(visible_hsv) >= 3
019781:                                 else -1
019782:                             )
019783:                             visible_motion_mean = float(
019784:                                 visible_ball.get('motion_mean', 0.0) or 0.0
019785:                             )
019786:                             visible_motion_max = float(
019787:                                 visible_ball.get('motion_max', 0.0) or 0.0
019788:                             )
019789:                             visible_conflict_distance = (
019790:                                 math.hypot(
019791:                                     float(visible_pos[0]) - float(cx),
019792:                                     float(visible_pos[1]) - float(cy),
019793:                                 )
019794:                                 if len(visible_pos) >= 2 else 0.0
019795:                             )
019796:                         except (TypeError, ValueError, IndexError):
019797:                             visible_pos = ()
019798:                             visible_s = -1
019799:                             visible_motion_mean = 0.0
019800:                             visible_motion_max = 0.0
019801:                             visible_conflict_distance = 0.0
019802: 
019803:                         visible_conflict_limit = max(100.0, float(frame.shape[1]) * 0.025)
019804:                         visible_independent = (
019805:                             visible_conflict_distance >= visible_conflict_limit and
019806:                             visible_s >= player_reacq_selected_s + 15 and
019807:                             (
019808:                                 visible_motion_mean >= 5.0 or
019809:                                 visible_motion_max >= 35.0
019810:                             )
019811:                         )
019812:                         if visible_independent:
019813:                             print(
019814:                                 f"Frame {self.frame_count}: "
019815:                                 f"[PLAYER-REACQ STRONG-MOTION VISIBLE-CONFLICT REANCHOR] "
019816:                                 f"rejected=({cx},{cy}) "
019817:                                 f"hsv=({player_reacq_selected_h},"
019818:                                 f"{player_reacq_selected_s},"
019819:                                 f"{player_reacq_selected_v}) "
019820:                                 f"area={player_reacq_selected_area:.1f}px "
019821:                                 f"jump={actual_distance:.1f}px "
019822:                                 f"pred_dist={selected_predicted_distance:.1f}px "
019823:                                 f"visible={visible_pos} visible_s={visible_s} "
019824:                                 f"motion={visible_motion_mean:.1f}/{visible_motion_max:.1f} "
019825:                                 f"conflict={visible_conflict_distance:.1f}px"
019826:                             )
019827:                             recovery_candidate = dict(visible_ball)
019828:                             recovery_candidate['recovery_label'] = (
019829:                                 'PLAYER-REACQ STRONG-MOTION VISIBLE REANCHOR'
019830:                             )
019831:                             return self._commit_night_visible_ball_recovery(
019832:                                 recovery_candidate, frame
019833:                             )
019834: 
019835:                 if (
019836:                         player_reacq_guard_active and
019837:                         (
019838:                             player_reacq_strong_motion or
019839:                             player_reacq_prediction_close or
019840:                             player_reacq_close_to_anchor
019841:                         ) and
019842:                         self._player_point_zone((cx, cy)) is None
019843:                 ):
019844:                     print(
019845:                         f"Frame {self.frame_count}: [PLAYER-REACQ MOTION CONFIRMED] "
019846:                         f"clearing guard at ({cx},{cy}) motion={motion_mean:.1f}/{motion_max:.1f} "
019847:                         f"jump={actual_distance:.1f}px pred_dist={selected_predicted_distance}"
019848:                     )
019849:                     self._player_reacq_protect_until_frame = -1
019850: 
019851:                 relaxed_contact_min_y = self._contact_reacquire_min_y(
019852:                     contact_reacquire_bounds,
019853:                     selected_predicted_distance,
019854:                     motion_mean,
019855:                     motion_max,
019856:                 ) if contact_reacquire_bounds is not None else None
019857:                 selected_area = selected_area_for_guard
019858:                 player_reacq_static = self._player_reacq_static_candidate(
019859:                     {
019860:                         'pos': (cx, cy),
019861:                         'area': selected_area,
019862:                         'motion_mean': motion_mean,
019863:                         'motion_max': motion_max,
019864:                     },
019865:                     predicted_point=predicted_point,
019866:                 )
019867:                 # The regular contour search can still run after motion
019868:                 # reacquisition returns no candidate.  Apply the same
019869:                 # adjacent-court guard to that path; otherwise a tiny static
019870:                 # blob outside the singles sideline can reset stuck_frame_count
019871:                 # and become the point-ending position.
019872:                 if not player_reacq_static and int(
019873:                         getattr(self, '_player_reacq_protect_until_frame', -1)
019874:                     ) >= self.frame_count:
019875:                     try:
019876:                         outside_court, outside_side, left_x, right_x = self._point_outside_singles_sidelines(
019877:                             (cx, cy), frame
019878:                         )
019879:                     except Exception:
019880:                         outside_court, outside_side, left_x, right_x = False, None, None, None
019881:                     if outside_court:
019882:                         sideline_depth = (
019883:                             (float(left_x) - cx) if outside_side == 'left'
019884:                             else (cx - float(right_x))
019885:                         )
019886:                         if sideline_depth > 35.0 or (motion_mean < 6.0 and motion_max < 25.0):
019887:                             player_reacq_static = True
019888:                 if not player_reacq_static and int(
019889:                         getattr(self, '_player_reacq_protect_until_frame', -1)
019890:                     ) >= self.frame_count:
019891:                     tracker = getattr(self, 'player_tracker', None)
019892:                     try:
019893:                         outside_main_court = (
019894:                             tracker is not None and
019895:                             hasattr(tracker, '_center_is_in_court') and
019896:                             not tracker._center_is_in_court((cx, cy), frame_width=frame.shape[1])
019897:                         )
019898:                     except Exception:
019899:                         outside_main_court = False
019900:                     if outside_main_court:
019901:                         player_reacq_static = True
019902:                 if player_reacq_static:
019903:                     self._record_rejected_contour_debug(
019904:                         best_contour,
019905:                         x1,
019906:                         y1,
019907:                         cx,
019908:                         cy,
019909:                         selected_area,
019910:                         "player-reacq distant static blob",
019911:                         source=best_source,
019912:                     )
019913:                     self.stuck_frame_count = max(
019914:                         int(getattr(self, 'stuck_frame_count', 0)) + 1, 1
019915:                     )
019916:                     print(
019917:                         f"Frame {self.frame_count}: [PLAYER-REACQ STATIC REJECT] "
019918:                         f"holding {self.ball_center} instead of ({cx},{cy}) "
019919:                         f"motion={motion_mean:.1f}/{motion_max:.1f}"
019920:                     )
019921:                     visible_ball = self._find_night_visible_ball_candidate(frame, frame_gray)
019922:                     if visible_ball is not None:
019923:                         print(
019924:                             f"Frame {self.frame_count}: [PLAYER-REACQ VISIBLE BALL] "
019925:                             f"using {visible_ball['pos']} area={visible_ball['area']:.1f}px "
019926:                             f"instead of static candidate ({cx},{cy})"
019927:                         )
019928:                         return self._commit_night_visible_ball_recovery(visible_ball, frame)
019929:                     return self.ball_center
019930:                 if top_return_search_context:
019931:                     if (
019932:                         getattr(self, '_top_return_mode', 'edge') in ('upper_side', 'upper_racket') and
019933:                         best_source not in ('primary', 'regular', 'alt')
019934:                     ):
019935:                         top_return_ok = False
019936:                         top_return_reason = f"top-return source {best_source} not allowed for upper-side reentry"
019937:                     else:
019938:                         top_return_ok, top_return_reason = self._top_return_reentry_ok(
019939:                             (cx, cy), selected_area, motion_mean, motion_max, frame.shape
019940:                         )
019941:                     if not top_return_ok:
019942:                         motion_reentry = self._find_top_return_motion_reentry(frame, frame_gray)
019943:                         if motion_reentry is not None:
019944:                             return self._commit_top_return_reentry(
019945:                                 motion_reentry, "TOP-RETURN MOTION REENTRY"
019946:                             )
019947:                         self._record_rejected_contour_debug(
019948:                             best_contour,
019949:                             x1,
019950:                             y1,
019951:                             cx,
019952:                             cy,
019953:                             selected_area,
019954:                             top_return_reason,
019955:                             source=best_source,
```

