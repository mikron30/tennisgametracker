# Exact V20/V25 player-reacquisition source context

Generated from `interactive_ball_analyzer.py` on this branch for V25 diagnosis.

## Lines 19486-19736
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
019720:                         # V30: the final per-candidate HSV is not bound yet on the
019721:                         # strong-motion reacquisition path.  Use the exact candidate
019722:                         # center pixel instead of a 5x5 median, which can be dominated
019723:                         # by the surrounding court/player pixels.  This matches the
019724:                         # same selected location that is about to be committed.
019725:                         px = int(cx)
019726:                         py = int(cy)
019727:                         if not (
019728:                                 0 <= px < frame.shape[1] and
019729:                                 0 <= py < frame.shape[0]
019730:                         ):
019731:                             raise ValueError('candidate center outside frame')
019732:                         candidate_pixel = frame[py:py + 1, px:px + 1]
019733:                         if not candidate_pixel.size:
019734:                             raise ValueError('empty candidate center pixel')
019735:                         candidate_pixel_hsv = cv2.cvtColor(
019736:                             candidate_pixel, cv2.COLOR_BGR2HSV
```

## Lines 19813-20033
```python
019813:                                     float(visible_pos[0]) - float(cx),
019814:                                     float(visible_pos[1]) - float(cy),
019815:                                 )
019816:                                 if len(visible_pos) >= 2 else 0.0
019817:                             )
019818:                         except (TypeError, ValueError, IndexError):
019819:                             visible_pos = ()
019820:                             visible_s = -1
019821:                             visible_motion_mean = 0.0
019822:                             visible_motion_max = 0.0
019823:                             visible_conflict_distance = 0.0
019824: 
019825:                         visible_conflict_limit = max(100.0, float(frame.shape[1]) * 0.025)
019826:                         visible_independent = (
019827:                             visible_conflict_distance >= visible_conflict_limit and
019828:                             visible_s >= player_reacq_selected_s + 15 and
019829:                             (
019830:                                 visible_motion_mean >= 5.0 or
019831:                                 visible_motion_max >= 35.0
019832:                             )
019833:                         )
019834:                         if visible_independent:
019835:                             print(
019836:                                 f"Frame {self.frame_count}: "
019837:                                 f"[PLAYER-REACQ STRONG-MOTION VISIBLE-CONFLICT REANCHOR] "
019838:                                 f"rejected=({cx},{cy}) "
019839:                                 f"hsv=({player_reacq_selected_h},"
019840:                                 f"{player_reacq_selected_s},"
019841:                                 f"{player_reacq_selected_v}) "
019842:                                 f"area={player_reacq_selected_area:.1f}px "
019843:                                 f"jump={actual_distance:.1f}px "
019844:                                 f"pred_dist={selected_predicted_distance:.1f}px "
019845:                                 f"visible={visible_pos} visible_s={visible_s} "
019846:                                 f"motion={visible_motion_mean:.1f}/{visible_motion_max:.1f} "
019847:                                 f"conflict={visible_conflict_distance:.1f}px"
019848:                             )
019849:                             recovery_candidate = dict(visible_ball)
019850:                             recovery_candidate['recovery_label'] = (
019851:                                 'PLAYER-REACQ STRONG-MOTION VISIBLE REANCHOR'
019852:                             )
019853:                             return self._commit_night_visible_ball_recovery(
019854:                                 recovery_candidate, frame
019855:                             )
019856: 
019857:                 # V31: V30 proves the selected f176 candidate is a large,
019858:                 # low-saturation, prediction-conflicting fragment, but the broad
019859:                 # visible-ball helper can fail to return an independent candidate
019860:                 # in that exact frame.  Do not then let a sharp player/racket edge
019861:                 # clear the protection merely because motion_max is high.  A real
019862:                 # ball recovery in this sequence has much stronger mean patch
019863:                 # motion; the false fragment is characteristically spiky
019864:                 # (low mean, high max).  In this narrow combination, hold the last
019865:                 # trusted anchor for one additional frame so the genuine ball can
019866:                 # reappear without poisoning direction/bounce state.
019867:                 player_reacq_spiky_visible_conflict_hold = (
019868:                     player_reacq_visible_conflict_probe and
019869:                     actual_distance >= max(280.0, float(frame.shape[1]) * 0.07) and
019870:                     selected_predicted_distance is not None and
019871:                     selected_predicted_distance >= 130.0 and
019872:                     player_reacq_selected_area >= 180.0 and
019873:                     0 <= player_reacq_selected_s <= 55 and
019874:                     motion_mean < 30.0 and
019875:                     motion_max >= 80.0 and
019876:                     not top_return_search_context and
019877:                     not back_return_search_context
019878:                 )
019879:                 if player_reacq_spiky_visible_conflict_hold:
019880:                     self._record_rejected_contour_debug(
019881:                         best_contour,
019882:                         x1,
019883:                         y1,
019884:                         cx,
019885:                         cy,
019886:                         selected_area_for_guard,
019887:                         (
019888:                             f'player-reacq spiky visible-conflict hold '
019889:                             f'jump={actual_distance:.1f}px '
019890:                             f'pred_dist={selected_predicted_distance:.1f}px '
019891:                             f'motion={motion_mean:.1f}/{motion_max:.1f}'
019892:                         ),
019893:                         source=best_source,
019894:                     )
019895:                     self.stuck_frame_count = max(
019896:                         int(getattr(self, 'stuck_frame_count', 0)) + 1,
019897:                         1,
019898:                     )
019899:                     print(
019900:                         f'Frame {self.frame_count}: '
019901:                         f'[PLAYER-REACQ VISIBLE-CONFLICT HOLD] '
019902:                         f'holding {self.ball_center} instead of ({cx},{cy}) '
019903:                         f'hsv=({player_reacq_selected_h},'
019904:                         f'{player_reacq_selected_s},'
019905:                         f'{player_reacq_selected_v}) '
019906:                         f'area={player_reacq_selected_area:.1f}px '
019907:                         f'jump={actual_distance:.1f}px '
019908:                         f'pred_dist={selected_predicted_distance:.1f}px '
019909:                         f'motion={motion_mean:.1f}/{motion_max:.1f}'
019910:                     )
019911:                     return self.ball_center
019912: 
019913:                 if (
019914:                         player_reacq_guard_active and
019915:                         (
019916:                             player_reacq_strong_motion or
019917:                             player_reacq_prediction_close or
019918:                             player_reacq_close_to_anchor
019919:                         ) and
019920:                         self._player_point_zone((cx, cy)) is None
019921:                 ):
019922:                     print(
019923:                         f"Frame {self.frame_count}: [PLAYER-REACQ MOTION CONFIRMED] "
019924:                         f"clearing guard at ({cx},{cy}) motion={motion_mean:.1f}/{motion_max:.1f} "
019925:                         f"jump={actual_distance:.1f}px pred_dist={selected_predicted_distance}"
019926:                     )
019927:                     self._player_reacq_protect_until_frame = -1
019928: 
019929:                 relaxed_contact_min_y = self._contact_reacquire_min_y(
019930:                     contact_reacquire_bounds,
019931:                     selected_predicted_distance,
019932:                     motion_mean,
019933:                     motion_max,
019934:                 ) if contact_reacquire_bounds is not None else None
019935:                 selected_area = selected_area_for_guard
019936:                 player_reacq_static = self._player_reacq_static_candidate(
019937:                     {
019938:                         'pos': (cx, cy),
019939:                         'area': selected_area,
019940:                         'motion_mean': motion_mean,
019941:                         'motion_max': motion_max,
019942:                     },
019943:                     predicted_point=predicted_point,
019944:                 )
019945:                 # The regular contour search can still run after motion
019946:                 # reacquisition returns no candidate.  Apply the same
019947:                 # adjacent-court guard to that path; otherwise a tiny static
019948:                 # blob outside the singles sideline can reset stuck_frame_count
019949:                 # and become the point-ending position.
019950:                 if not player_reacq_static and int(
019951:                         getattr(self, '_player_reacq_protect_until_frame', -1)
019952:                     ) >= self.frame_count:
019953:                     try:
019954:                         outside_court, outside_side, left_x, right_x = self._point_outside_singles_sidelines(
019955:                             (cx, cy), frame
019956:                         )
019957:                     except Exception:
019958:                         outside_court, outside_side, left_x, right_x = False, None, None, None
019959:                     if outside_court:
019960:                         sideline_depth = (
019961:                             (float(left_x) - cx) if outside_side == 'left'
019962:                             else (cx - float(right_x))
019963:                         )
019964:                         if sideline_depth > 35.0 or (motion_mean < 6.0 and motion_max < 25.0):
019965:                             player_reacq_static = True
019966:                 if not player_reacq_static and int(
019967:                         getattr(self, '_player_reacq_protect_until_frame', -1)
019968:                     ) >= self.frame_count:
019969:                     tracker = getattr(self, 'player_tracker', None)
019970:                     try:
019971:                         outside_main_court = (
019972:                             tracker is not None and
019973:                             hasattr(tracker, '_center_is_in_court') and
019974:                             not tracker._center_is_in_court((cx, cy), frame_width=frame.shape[1])
019975:                         )
019976:                     except Exception:
019977:                         outside_main_court = False
019978:                     if outside_main_court:
019979:                         player_reacq_static = True
019980:                 if player_reacq_static:
019981:                     self._record_rejected_contour_debug(
019982:                         best_contour,
019983:                         x1,
019984:                         y1,
019985:                         cx,
019986:                         cy,
019987:                         selected_area,
019988:                         "player-reacq distant static blob",
019989:                         source=best_source,
019990:                     )
019991:                     self.stuck_frame_count = max(
019992:                         int(getattr(self, 'stuck_frame_count', 0)) + 1, 1
019993:                     )
019994:                     print(
019995:                         f"Frame {self.frame_count}: [PLAYER-REACQ STATIC REJECT] "
019996:                         f"holding {self.ball_center} instead of ({cx},{cy}) "
019997:                         f"motion={motion_mean:.1f}/{motion_max:.1f}"
019998:                     )
019999:                     visible_ball = self._find_night_visible_ball_candidate(frame, frame_gray)
020000:                     if visible_ball is not None:
020001:                         print(
020002:                             f"Frame {self.frame_count}: [PLAYER-REACQ VISIBLE BALL] "
020003:                             f"using {visible_ball['pos']} area={visible_ball['area']:.1f}px "
020004:                             f"instead of static candidate ({cx},{cy})"
020005:                         )
020006:                         return self._commit_night_visible_ball_recovery(visible_ball, frame)
020007:                     return self.ball_center
020008:                 if top_return_search_context:
020009:                     if (
020010:                         getattr(self, '_top_return_mode', 'edge') in ('upper_side', 'upper_racket') and
020011:                         best_source not in ('primary', 'regular', 'alt')
020012:                     ):
020013:                         top_return_ok = False
020014:                         top_return_reason = f"top-return source {best_source} not allowed for upper-side reentry"
020015:                     else:
020016:                         top_return_ok, top_return_reason = self._top_return_reentry_ok(
020017:                             (cx, cy), selected_area, motion_mean, motion_max, frame.shape
020018:                         )
020019:                     if not top_return_ok:
020020:                         motion_reentry = self._find_top_return_motion_reentry(frame, frame_gray)
020021:                         if motion_reentry is not None:
020022:                             return self._commit_top_return_reentry(
020023:                                 motion_reentry, "TOP-RETURN MOTION REENTRY"
020024:                             )
020025:                         self._record_rejected_contour_debug(
020026:                             best_contour,
020027:                             x1,
020028:                             y1,
020029:                             cx,
020030:                             cy,
020031:                             selected_area,
020032:                             top_return_reason,
020033:                             source=best_source,
```

