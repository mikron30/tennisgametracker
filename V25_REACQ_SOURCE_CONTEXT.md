# Exact V20/V25 player-reacquisition source context

Generated from `interactive_ball_analyzer.py` on this branch for V25 diagnosis.

## Lines 19495-19745
```python
019495:                             selected_area_for_guard,
019496:                             guard_motion_mean,
019497:                             guard_motion_max,
019498:                             best_source,
019499:                             frame.shape,
019500:                         )
019501:                         if not allow_confirmed_top_return:
019502:                             self._record_rejected_contour_debug(
019503:                                 best_contour,
019504:                                 x1,
019505:                                 y1,
019506:                                 cx,
019507:                                 cy,
019508:                                 selected_area_for_guard,
019509:                                 f"player-reacq jump {actual_distance:.1f}px > {max_player_reacq_jump:.1f}px",
019510:                                 source=best_source,
019511:                             )
019512:                             self.stuck_frame_count = max(
019513:                                 int(getattr(self, 'stuck_frame_count', 0)) + 1,
019514:                                 5,
019515:                             )
019516:                             print(
019517:                                 f"Frame {self.frame_count}: [PLAYER-REACQ JUMP REJECT] "
019518:                                 f"holding {self.ball_center} instead of ({cx},{cy}) "
019519:                                 f"jump={actual_distance:.1f}px limit={max_player_reacq_jump:.1f}px"
019520:                             )
019521:                             return self.ball_center
019522:                         print(
019523:                             f"Frame {self.frame_count}: [TOP-RETURN JUMP ALLOW] "
019524:                             f"accepting strongly confirmed re-entry at ({cx},{cy}) "
019525:                             f"jump={actual_distance:.1f}px motion="
019526:                             f"{guard_motion_mean:.1f}/{guard_motion_max:.1f}"
019527:                         )
019528:                 
019529:                 frame_height, frame_width = frame.shape[:2]
019530:                 at_edge = (y_prev < edge_threshold or y_prev > frame_height - edge_threshold or
019531:                           x_prev < edge_threshold or x_prev > frame_width - edge_threshold)
019532:                 if (at_edge and actual_distance > jump_threshold and
019533:                         not top_return_search_context and
019534:                         not back_return_search_context):
019535:                     self._record_rejected_contour_debug(
019536:                         best_contour,
019537:                         x1,
019538:                         y1,
019539:                         cx,
019540:                         cy,
019541:                         cv2.contourArea(best_contour),
019542:                         f"edge jump {actual_distance:.1f}px from {self.ball_center}",
019543:                         source=best_source,
019544:                     )
019545:                     print(f"  DEBUG: [REJECTED] Candidate at ({cx},{cy}), distance={actual_distance:.1f}px")
019546:                     print(f"  DEBUG: Ball was at edge (y={y_prev}), closest match is {actual_distance:.1f}px away")
019547:                     print(f"  DEBUG: This is likely a FALSE POSITIVE - ball probably went off-screen")
019548:                     _verbose_debug_print(f"  DEBUG: KEEPING marker at last edge position: {self.ball_center}")
019549:                     print(f"  DEBUG: Will wait for ball to return...")
019550:                     return self.ball_center
019551: 
019552:                 selected_motion = (
019553:                     selected_motion_for_guard
019554:                     if selected_motion_for_guard is not None else
019555:                     self._candidate_motion_metrics(frame_gray, cx, cy)
019556:                 )
019557:                 motion_mean = selected_motion['mean'] if selected_motion is not None else 0.0
019558:                 motion_max = selected_motion['max'] if selected_motion is not None else 0.0
019559:                 player_reacq_guard_active = int(
019560:                     getattr(self, '_player_reacq_protect_until_frame', -1)
019561:                 ) >= self.frame_count
019562:                 selected_predicted_distance = (
019563:                     math.hypot(cx - predicted_point[0], cy - predicted_point[1])
019564:                     if predicted_point is not None else None
019565:                 )
019566: 
019567:                 # Do not let a distant, weakly moving background blob end the
019568:                 # protected player-contact reacquisition window. A real fast
019569:                 # ball may travel far, so large jumps are still allowed when
019570:                 # inter-frame motion is strong or the candidate agrees with
019571:                 # the predicted trajectory.
019572:                 player_reacq_strong_motion = (
019573:                     motion_mean >= 12.0 or motion_max >= 80.0
019574:                 )
019575:                 player_reacq_prediction_close = (
019576:                     selected_predicted_distance is not None and
019577:                     selected_predicted_distance <= 60.0
019578:                 )
019579:                 player_reacq_close_to_anchor = actual_distance <= 80.0
019580:                 player_reacq_weak_large_jump = (
019581:                     player_reacq_guard_active and
019582:                     actual_distance >= 150.0 and
019583:                     not player_reacq_strong_motion and
019584:                     not player_reacq_prediction_close
019585:                 )
019586:                 if player_reacq_weak_large_jump:
019587:                     self._record_rejected_contour_debug(
019588:                         best_contour,
019589:                         x1,
019590:                         y1,
019591:                         cx,
019592:                         cy,
019593:                         selected_area_for_guard,
019594:                         (
019595:                             f"player-reacq weak large jump {actual_distance:.1f}px "
019596:                             f"motion={motion_mean:.1f}/{motion_max:.1f}"
019597:                         ),
019598:                         source=best_source,
019599:                     )
019600:                     self.stuck_frame_count = max(
019601:                         int(getattr(self, 'stuck_frame_count', 0)) + 1,
019602:                         1,
019603:                     )
019604:                     print(
019605:                         f"Frame {self.frame_count}: [PLAYER-REACQ WEAK-JUMP REJECT] "
019606:                         f"holding {self.ball_center} instead of ({cx},{cy}) "
019607:                         f"jump={actual_distance:.1f}px motion="
019608:                         f"{motion_mean:.1f}/{motion_max:.1f} "
019609:                         f"pred_dist={selected_predicted_distance}"
019610:                     )
019611:                     visible_ball = self._find_night_visible_ball_candidate(frame, frame_gray)
019612:                     if visible_ball is not None:
019613:                         visible_pos = tuple(visible_ball['pos'])
019614:                         visible_jump = math.hypot(
019615:                             float(visible_pos[0]) - float(self.ball_center[0]),
019616:                             float(visible_pos[1]) - float(self.ball_center[1]),
019617:                         )
019618:                         visible_mean = float(visible_ball.get('motion_mean', 0.0) or 0.0)
019619:                         visible_max = float(visible_ball.get('motion_max', 0.0) or 0.0)
019620:                         visible_pred_dist = (
019621:                             math.hypot(
019622:                                 float(visible_pos[0]) - float(predicted_point[0]),
019623:                                 float(visible_pos[1]) - float(predicted_point[1]),
019624:                             )
019625:                             if predicted_point is not None else None
019626:                         )
019627:                         visible_safe = (
019628:                             visible_jump <= 80.0 or
019629:                             visible_mean >= 12.0 or
019630:                             visible_max >= 80.0 or
019631:                             (visible_pred_dist is not None and visible_pred_dist <= 60.0)
019632:                         )
019633:                         if visible_safe:
019634:                             print(
019635:                                 f"Frame {self.frame_count}: [PLAYER-REACQ WEAK-JUMP VISIBLE BALL] "
019636:                                 f"using {visible_pos} instead; jump={visible_jump:.1f}px "
019637:                                 f"motion={visible_mean:.1f}/{visible_max:.1f}"
019638:                             )
019639:                             return self._commit_night_visible_ball_recovery(visible_ball, frame)
019640:                         print(
019641:                             f"Frame {self.frame_count}: [PLAYER-REACQ WEAK-JUMP FALLBACK REJECT] "
019642:                             f"ignoring visible candidate {visible_pos} jump={visible_jump:.1f}px "
019643:                             f"motion={visible_mean:.1f}/{visible_max:.1f}"
019644:                         )
019645:                     return self.ball_center
019646: 
019647:                 # V25: strong image motion by itself must not clear the
019648:                 # player-contact reacquisition guard for an extreme jump that
019649:                 # strongly contradicts the predicted ball trajectory. The
019650:                 # reviewed V20 f175 failure jumped 388px from the recovered
019651:                 # ball while landing about 560px from prediction; moving-player
019652:                 # image motion made that false fragment look valid. Hold the
019653:                 # last trusted ball for one more frame instead. Specialized
019654:                 # top/back-return paths keep their own long-jump validation.
019655:                 player_reacq_extreme_jump_limit = max(
019656:                     240.0, float(frame_width) * 0.065
019657:                 )
019658:                 player_reacq_extreme_pred_limit = max(
019659:                     150.0, float(frame_width) * 0.045
019660:                 )
019661:                 player_reacq_extreme_prediction_conflict = (
019662:                     player_reacq_guard_active and
019663:                     actual_distance >= player_reacq_extreme_jump_limit and
019664:                     selected_predicted_distance is not None and
019665:                     selected_predicted_distance >= player_reacq_extreme_pred_limit and
019666:                     not top_return_search_context and
019667:                     not back_return_search_context
019668:                 )
019669:                 if player_reacq_extreme_prediction_conflict:
019670:                     self._record_rejected_contour_debug(
019671:                         best_contour,
019672:                         x1,
019673:                         y1,
019674:                         cx,
019675:                         cy,
019676:                         selected_area_for_guard,
019677:                         (
019678:                             f"player-reacq extreme prediction conflict "
019679:                             f"jump={actual_distance:.1f}px "
019680:                             f"pred_dist={selected_predicted_distance:.1f}px"
019681:                         ),
019682:                         source=best_source,
019683:                     )
019684:                     self.stuck_frame_count = max(
019685:                         int(getattr(self, 'stuck_frame_count', 0)) + 1,
019686:                         1,
019687:                     )
019688:                     print(
019689:                         f"Frame {self.frame_count}: "
019690:                         f"[PLAYER-REACQ EXTREME-PREDICTION REJECT] "
019691:                         f"holding {self.ball_center} instead of ({cx},{cy}) "
019692:                         f"jump={actual_distance:.1f}px "
019693:                         f"limit={player_reacq_extreme_jump_limit:.1f}px "
019694:                         f"pred_dist={selected_predicted_distance:.1f}px "
019695:                         f"pred_limit={player_reacq_extreme_pred_limit:.1f}px "
019696:                         f"motion={motion_mean:.1f}/{motion_max:.1f}"
019697:                     )
019698:                     return self.ball_center
019699: 
019700:                 # Clearing the protection now needs either strong image motion
019701:                 # or geometric continuity. This preserves slow nearby tracks
019702:                 # while preventing a 300+ px weak-motion blob from becoming a
019703:                 # new anchor simply because it is outside the player box.
019704:                 # V28: a moving player/racket fragment can satisfy the generic
019705:                 # strong-motion escape even when it is a large, prediction-conflicting
019706:                 # reacquisition jump.  The reviewed early rally does this at f176:
019707:                 # (709,1558), area 249, S=42, jump 301px, pred error 153px, while
019708:                 # the real yellow ball is visibly elsewhere.  Do not weaken the
019709:                 # normal strong-motion path.  Only override when the selected
019710:                 # candidate is simultaneously large-jump, prediction-conflicting,
019711:                 # large-area and low-saturation, and the existing visible-ball
019712:                 # detector independently finds a more saturated moving ball well
019713:                 # away from it in the same frame.
019714:                 # V29: this V28 probe executes before hsv_values/bulb_size are
019715:                 # guaranteed to be bound on every tracking path.  Prefer those
019716:                 # already-bound values when present, otherwise derive evidence
019717:                 # directly from the selected candidate location/contour.
019718:                 player_reacq_candidate_hsv = locals().get('hsv_values')
019719:                 player_reacq_candidate_area = locals().get('bulb_size')
019720:                 try:
019721:                     if (
019722:                             player_reacq_candidate_hsv is not None and
019723:                             len(player_reacq_candidate_hsv) >= 3
019724:                     ):
019725:                         player_reacq_selected_h = int(player_reacq_candidate_hsv[0])
019726:                         player_reacq_selected_s = int(player_reacq_candidate_hsv[1])
019727:                         player_reacq_selected_v = int(player_reacq_candidate_hsv[2])
019728:                     else:
019729:                         # V30: the final per-candidate HSV is not bound yet on the
019730:                         # strong-motion reacquisition path.  Use the exact candidate
019731:                         # center pixel instead of a 5x5 median, which can be dominated
019732:                         # by the surrounding court/player pixels.  This matches the
019733:                         # same selected location that is about to be committed.
019734:                         px = int(cx)
019735:                         py = int(cy)
019736:                         if not (
019737:                                 0 <= px < frame.shape[1] and
019738:                                 0 <= py < frame.shape[0]
019739:                         ):
019740:                             raise ValueError('candidate center outside frame')
019741:                         candidate_pixel = frame[py:py + 1, px:px + 1]
019742:                         if not candidate_pixel.size:
019743:                             raise ValueError('empty candidate center pixel')
019744:                         candidate_pixel_hsv = cv2.cvtColor(
019745:                             candidate_pixel, cv2.COLOR_BGR2HSV
```

## Lines 19822-20042
```python
019822:                                     float(visible_pos[0]) - float(cx),
019823:                                     float(visible_pos[1]) - float(cy),
019824:                                 )
019825:                                 if len(visible_pos) >= 2 else 0.0
019826:                             )
019827:                         except (TypeError, ValueError, IndexError):
019828:                             visible_pos = ()
019829:                             visible_s = -1
019830:                             visible_motion_mean = 0.0
019831:                             visible_motion_max = 0.0
019832:                             visible_conflict_distance = 0.0
019833: 
019834:                         visible_conflict_limit = max(100.0, float(frame.shape[1]) * 0.025)
019835:                         visible_independent = (
019836:                             visible_conflict_distance >= visible_conflict_limit and
019837:                             visible_s >= player_reacq_selected_s + 15 and
019838:                             (
019839:                                 visible_motion_mean >= 5.0 or
019840:                                 visible_motion_max >= 35.0
019841:                             )
019842:                         )
019843:                         if visible_independent:
019844:                             print(
019845:                                 f"Frame {self.frame_count}: "
019846:                                 f"[PLAYER-REACQ STRONG-MOTION VISIBLE-CONFLICT REANCHOR] "
019847:                                 f"rejected=({cx},{cy}) "
019848:                                 f"hsv=({player_reacq_selected_h},"
019849:                                 f"{player_reacq_selected_s},"
019850:                                 f"{player_reacq_selected_v}) "
019851:                                 f"area={player_reacq_selected_area:.1f}px "
019852:                                 f"jump={actual_distance:.1f}px "
019853:                                 f"pred_dist={selected_predicted_distance:.1f}px "
019854:                                 f"visible={visible_pos} visible_s={visible_s} "
019855:                                 f"motion={visible_motion_mean:.1f}/{visible_motion_max:.1f} "
019856:                                 f"conflict={visible_conflict_distance:.1f}px"
019857:                             )
019858:                             recovery_candidate = dict(visible_ball)
019859:                             recovery_candidate['recovery_label'] = (
019860:                                 'PLAYER-REACQ STRONG-MOTION VISIBLE REANCHOR'
019861:                             )
019862:                             return self._commit_night_visible_ball_recovery(
019863:                                 recovery_candidate, frame
019864:                             )
019865: 
019866:                 # V31: V30 proves the selected f176 candidate is a large,
019867:                 # low-saturation, prediction-conflicting fragment, but the broad
019868:                 # visible-ball helper can fail to return an independent candidate
019869:                 # in that exact frame.  Do not then let a sharp player/racket edge
019870:                 # clear the protection merely because motion_max is high.  A real
019871:                 # ball recovery in this sequence has much stronger mean patch
019872:                 # motion; the false fragment is characteristically spiky
019873:                 # (low mean, high max).  In this narrow combination, hold the last
019874:                 # trusted anchor for one additional frame so the genuine ball can
019875:                 # reappear without poisoning direction/bounce state.
019876:                 player_reacq_spiky_visible_conflict_hold = (
019877:                     player_reacq_visible_conflict_probe and
019878:                     actual_distance >= max(280.0, float(frame.shape[1]) * 0.07) and
019879:                     selected_predicted_distance is not None and
019880:                     selected_predicted_distance >= 130.0 and
019881:                     player_reacq_selected_area >= 180.0 and
019882:                     0 <= player_reacq_selected_s <= 55 and
019883:                     motion_mean < 30.0 and
019884:                     motion_max >= 80.0 and
019885:                     not top_return_search_context and
019886:                     not back_return_search_context
019887:                 )
019888:                 if player_reacq_spiky_visible_conflict_hold:
019889:                     self._record_rejected_contour_debug(
019890:                         best_contour,
019891:                         x1,
019892:                         y1,
019893:                         cx,
019894:                         cy,
019895:                         selected_area_for_guard,
019896:                         (
019897:                             f'player-reacq spiky visible-conflict hold '
019898:                             f'jump={actual_distance:.1f}px '
019899:                             f'pred_dist={selected_predicted_distance:.1f}px '
019900:                             f'motion={motion_mean:.1f}/{motion_max:.1f}'
019901:                         ),
019902:                         source=best_source,
019903:                     )
019904:                     self.stuck_frame_count = max(
019905:                         int(getattr(self, 'stuck_frame_count', 0)) + 1,
019906:                         1,
019907:                     )
019908:                     print(
019909:                         f'Frame {self.frame_count}: '
019910:                         f'[PLAYER-REACQ VISIBLE-CONFLICT HOLD] '
019911:                         f'holding {self.ball_center} instead of ({cx},{cy}) '
019912:                         f'hsv=({player_reacq_selected_h},'
019913:                         f'{player_reacq_selected_s},'
019914:                         f'{player_reacq_selected_v}) '
019915:                         f'area={player_reacq_selected_area:.1f}px '
019916:                         f'jump={actual_distance:.1f}px '
019917:                         f'pred_dist={selected_predicted_distance:.1f}px '
019918:                         f'motion={motion_mean:.1f}/{motion_max:.1f}'
019919:                     )
019920:                     return self.ball_center
019921: 
019922:                 if (
019923:                         player_reacq_guard_active and
019924:                         (
019925:                             player_reacq_strong_motion or
019926:                             player_reacq_prediction_close or
019927:                             player_reacq_close_to_anchor
019928:                         ) and
019929:                         self._player_point_zone((cx, cy)) is None
019930:                 ):
019931:                     print(
019932:                         f"Frame {self.frame_count}: [PLAYER-REACQ MOTION CONFIRMED] "
019933:                         f"clearing guard at ({cx},{cy}) motion={motion_mean:.1f}/{motion_max:.1f} "
019934:                         f"jump={actual_distance:.1f}px pred_dist={selected_predicted_distance}"
019935:                     )
019936:                     self._player_reacq_protect_until_frame = -1
019937: 
019938:                 relaxed_contact_min_y = self._contact_reacquire_min_y(
019939:                     contact_reacquire_bounds,
019940:                     selected_predicted_distance,
019941:                     motion_mean,
019942:                     motion_max,
019943:                 ) if contact_reacquire_bounds is not None else None
019944:                 selected_area = selected_area_for_guard
019945:                 player_reacq_static = self._player_reacq_static_candidate(
019946:                     {
019947:                         'pos': (cx, cy),
019948:                         'area': selected_area,
019949:                         'motion_mean': motion_mean,
019950:                         'motion_max': motion_max,
019951:                     },
019952:                     predicted_point=predicted_point,
019953:                 )
019954:                 # The regular contour search can still run after motion
019955:                 # reacquisition returns no candidate.  Apply the same
019956:                 # adjacent-court guard to that path; otherwise a tiny static
019957:                 # blob outside the singles sideline can reset stuck_frame_count
019958:                 # and become the point-ending position.
019959:                 if not player_reacq_static and int(
019960:                         getattr(self, '_player_reacq_protect_until_frame', -1)
019961:                     ) >= self.frame_count:
019962:                     try:
019963:                         outside_court, outside_side, left_x, right_x = self._point_outside_singles_sidelines(
019964:                             (cx, cy), frame
019965:                         )
019966:                     except Exception:
019967:                         outside_court, outside_side, left_x, right_x = False, None, None, None
019968:                     if outside_court:
019969:                         sideline_depth = (
019970:                             (float(left_x) - cx) if outside_side == 'left'
019971:                             else (cx - float(right_x))
019972:                         )
019973:                         if sideline_depth > 35.0 or (motion_mean < 6.0 and motion_max < 25.0):
019974:                             player_reacq_static = True
019975:                 if not player_reacq_static and int(
019976:                         getattr(self, '_player_reacq_protect_until_frame', -1)
019977:                     ) >= self.frame_count:
019978:                     tracker = getattr(self, 'player_tracker', None)
019979:                     try:
019980:                         outside_main_court = (
019981:                             tracker is not None and
019982:                             hasattr(tracker, '_center_is_in_court') and
019983:                             not tracker._center_is_in_court((cx, cy), frame_width=frame.shape[1])
019984:                         )
019985:                     except Exception:
019986:                         outside_main_court = False
019987:                     if outside_main_court:
019988:                         player_reacq_static = True
019989:                 if player_reacq_static:
019990:                     self._record_rejected_contour_debug(
019991:                         best_contour,
019992:                         x1,
019993:                         y1,
019994:                         cx,
019995:                         cy,
019996:                         selected_area,
019997:                         "player-reacq distant static blob",
019998:                         source=best_source,
019999:                     )
020000:                     self.stuck_frame_count = max(
020001:                         int(getattr(self, 'stuck_frame_count', 0)) + 1, 1
020002:                     )
020003:                     print(
020004:                         f"Frame {self.frame_count}: [PLAYER-REACQ STATIC REJECT] "
020005:                         f"holding {self.ball_center} instead of ({cx},{cy}) "
020006:                         f"motion={motion_mean:.1f}/{motion_max:.1f}"
020007:                     )
020008:                     visible_ball = self._find_night_visible_ball_candidate(frame, frame_gray)
020009:                     if visible_ball is not None:
020010:                         print(
020011:                             f"Frame {self.frame_count}: [PLAYER-REACQ VISIBLE BALL] "
020012:                             f"using {visible_ball['pos']} area={visible_ball['area']:.1f}px "
020013:                             f"instead of static candidate ({cx},{cy})"
020014:                         )
020015:                         return self._commit_night_visible_ball_recovery(visible_ball, frame)
020016:                     return self.ball_center
020017:                 if top_return_search_context:
020018:                     if (
020019:                         getattr(self, '_top_return_mode', 'edge') in ('upper_side', 'upper_racket') and
020020:                         best_source not in ('primary', 'regular', 'alt')
020021:                     ):
020022:                         top_return_ok = False
020023:                         top_return_reason = f"top-return source {best_source} not allowed for upper-side reentry"
020024:                     else:
020025:                         top_return_ok, top_return_reason = self._top_return_reentry_ok(
020026:                             (cx, cy), selected_area, motion_mean, motion_max, frame.shape
020027:                         )
020028:                     if not top_return_ok:
020029:                         motion_reentry = self._find_top_return_motion_reentry(frame, frame_gray)
020030:                         if motion_reentry is not None:
020031:                             return self._commit_top_return_reentry(
020032:                                 motion_reentry, "TOP-RETURN MOTION REENTRY"
020033:                             )
020034:                         self._record_rejected_contour_debug(
020035:                             best_contour,
020036:                             x1,
020037:                             y1,
020038:                             cx,
020039:                             cy,
020040:                             selected_area,
020041:                             top_return_reason,
020042:                             source=best_source,
```

