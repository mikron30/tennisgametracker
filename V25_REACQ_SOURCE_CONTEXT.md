# Exact V20/V25 player-reacquisition source context

Generated from `interactive_ball_analyzer.py` on this branch for V25 diagnosis.

## Lines 19579-19829
```python
019579:                             selected_area_for_guard,
019580:                             guard_motion_mean,
019581:                             guard_motion_max,
019582:                             best_source,
019583:                             frame.shape,
019584:                         )
019585:                         if not allow_confirmed_top_return:
019586:                             self._record_rejected_contour_debug(
019587:                                 best_contour,
019588:                                 x1,
019589:                                 y1,
019590:                                 cx,
019591:                                 cy,
019592:                                 selected_area_for_guard,
019593:                                 f"player-reacq jump {actual_distance:.1f}px > {max_player_reacq_jump:.1f}px",
019594:                                 source=best_source,
019595:                             )
019596:                             self.stuck_frame_count = max(
019597:                                 int(getattr(self, 'stuck_frame_count', 0)) + 1,
019598:                                 5,
019599:                             )
019600:                             print(
019601:                                 f"Frame {self.frame_count}: [PLAYER-REACQ JUMP REJECT] "
019602:                                 f"holding {self.ball_center} instead of ({cx},{cy}) "
019603:                                 f"jump={actual_distance:.1f}px limit={max_player_reacq_jump:.1f}px"
019604:                             )
019605:                             return self.ball_center
019606:                         print(
019607:                             f"Frame {self.frame_count}: [TOP-RETURN JUMP ALLOW] "
019608:                             f"accepting strongly confirmed re-entry at ({cx},{cy}) "
019609:                             f"jump={actual_distance:.1f}px motion="
019610:                             f"{guard_motion_mean:.1f}/{guard_motion_max:.1f}"
019611:                         )
019612:                 
019613:                 frame_height, frame_width = frame.shape[:2]
019614:                 at_edge = (y_prev < edge_threshold or y_prev > frame_height - edge_threshold or
019615:                           x_prev < edge_threshold or x_prev > frame_width - edge_threshold)
019616:                 if (at_edge and actual_distance > jump_threshold and
019617:                         not top_return_search_context and
019618:                         not back_return_search_context):
019619:                     self._record_rejected_contour_debug(
019620:                         best_contour,
019621:                         x1,
019622:                         y1,
019623:                         cx,
019624:                         cy,
019625:                         cv2.contourArea(best_contour),
019626:                         f"edge jump {actual_distance:.1f}px from {self.ball_center}",
019627:                         source=best_source,
019628:                     )
019629:                     print(f"  DEBUG: [REJECTED] Candidate at ({cx},{cy}), distance={actual_distance:.1f}px")
019630:                     print(f"  DEBUG: Ball was at edge (y={y_prev}), closest match is {actual_distance:.1f}px away")
019631:                     print(f"  DEBUG: This is likely a FALSE POSITIVE - ball probably went off-screen")
019632:                     _verbose_debug_print(f"  DEBUG: KEEPING marker at last edge position: {self.ball_center}")
019633:                     print(f"  DEBUG: Will wait for ball to return...")
019634:                     return self.ball_center
019635: 
019636:                 selected_motion = (
019637:                     selected_motion_for_guard
019638:                     if selected_motion_for_guard is not None else
019639:                     self._candidate_motion_metrics(frame_gray, cx, cy)
019640:                 )
019641:                 motion_mean = selected_motion['mean'] if selected_motion is not None else 0.0
019642:                 motion_max = selected_motion['max'] if selected_motion is not None else 0.0
019643:                 player_reacq_guard_active = int(
019644:                     getattr(self, '_player_reacq_protect_until_frame', -1)
019645:                 ) >= self.frame_count
019646:                 selected_predicted_distance = (
019647:                     math.hypot(cx - predicted_point[0], cy - predicted_point[1])
019648:                     if predicted_point is not None else None
019649:                 )
019650: 
019651:                 # Do not let a distant, weakly moving background blob end the
019652:                 # protected player-contact reacquisition window. A real fast
019653:                 # ball may travel far, so large jumps are still allowed when
019654:                 # inter-frame motion is strong or the candidate agrees with
019655:                 # the predicted trajectory.
019656:                 player_reacq_strong_motion = (
019657:                     motion_mean >= 12.0 or motion_max >= 80.0
019658:                 )
019659:                 player_reacq_prediction_close = (
019660:                     selected_predicted_distance is not None and
019661:                     selected_predicted_distance <= 60.0
019662:                 )
019663:                 player_reacq_close_to_anchor = actual_distance <= 80.0
019664:                 player_reacq_weak_large_jump = (
019665:                     player_reacq_guard_active and
019666:                     actual_distance >= 150.0 and
019667:                     not player_reacq_strong_motion and
019668:                     not player_reacq_prediction_close
019669:                 )
019670:                 if player_reacq_weak_large_jump:
019671:                     self._record_rejected_contour_debug(
019672:                         best_contour,
019673:                         x1,
019674:                         y1,
019675:                         cx,
019676:                         cy,
019677:                         selected_area_for_guard,
019678:                         (
019679:                             f"player-reacq weak large jump {actual_distance:.1f}px "
019680:                             f"motion={motion_mean:.1f}/{motion_max:.1f}"
019681:                         ),
019682:                         source=best_source,
019683:                     )
019684:                     self.stuck_frame_count = max(
019685:                         int(getattr(self, 'stuck_frame_count', 0)) + 1,
019686:                         1,
019687:                     )
019688:                     print(
019689:                         f"Frame {self.frame_count}: [PLAYER-REACQ WEAK-JUMP REJECT] "
019690:                         f"holding {self.ball_center} instead of ({cx},{cy}) "
019691:                         f"jump={actual_distance:.1f}px motion="
019692:                         f"{motion_mean:.1f}/{motion_max:.1f} "
019693:                         f"pred_dist={selected_predicted_distance}"
019694:                     )
019695:                     visible_ball = self._find_night_visible_ball_candidate(frame, frame_gray)
019696:                     if visible_ball is not None:
019697:                         visible_pos = tuple(visible_ball['pos'])
019698:                         visible_jump = math.hypot(
019699:                             float(visible_pos[0]) - float(self.ball_center[0]),
019700:                             float(visible_pos[1]) - float(self.ball_center[1]),
019701:                         )
019702:                         visible_mean = float(visible_ball.get('motion_mean', 0.0) or 0.0)
019703:                         visible_max = float(visible_ball.get('motion_max', 0.0) or 0.0)
019704:                         visible_pred_dist = (
019705:                             math.hypot(
019706:                                 float(visible_pos[0]) - float(predicted_point[0]),
019707:                                 float(visible_pos[1]) - float(predicted_point[1]),
019708:                             )
019709:                             if predicted_point is not None else None
019710:                         )
019711:                         visible_safe = (
019712:                             visible_jump <= 80.0 or
019713:                             visible_mean >= 12.0 or
019714:                             visible_max >= 80.0 or
019715:                             (visible_pred_dist is not None and visible_pred_dist <= 60.0)
019716:                         )
019717:                         if visible_safe:
019718:                             print(
019719:                                 f"Frame {self.frame_count}: [PLAYER-REACQ WEAK-JUMP VISIBLE BALL] "
019720:                                 f"using {visible_pos} instead; jump={visible_jump:.1f}px "
019721:                                 f"motion={visible_mean:.1f}/{visible_max:.1f}"
019722:                             )
019723:                             return self._commit_night_visible_ball_recovery(visible_ball, frame)
019724:                         print(
019725:                             f"Frame {self.frame_count}: [PLAYER-REACQ WEAK-JUMP FALLBACK REJECT] "
019726:                             f"ignoring visible candidate {visible_pos} jump={visible_jump:.1f}px "
019727:                             f"motion={visible_mean:.1f}/{visible_max:.1f}"
019728:                         )
019729:                     return self.ball_center
019730: 
019731:                 # V25: strong image motion by itself must not clear the
019732:                 # player-contact reacquisition guard for an extreme jump that
019733:                 # strongly contradicts the predicted ball trajectory. The
019734:                 # reviewed V20 f175 failure jumped 388px from the recovered
019735:                 # ball while landing about 560px from prediction; moving-player
019736:                 # image motion made that false fragment look valid. Hold the
019737:                 # last trusted ball for one more frame instead. Specialized
019738:                 # top/back-return paths keep their own long-jump validation.
019739:                 player_reacq_extreme_jump_limit = max(
019740:                     240.0, float(frame_width) * 0.065
019741:                 )
019742:                 player_reacq_extreme_pred_limit = max(
019743:                     150.0, float(frame_width) * 0.045
019744:                 )
019745:                 player_reacq_extreme_prediction_conflict = (
019746:                     player_reacq_guard_active and
019747:                     actual_distance >= player_reacq_extreme_jump_limit and
019748:                     selected_predicted_distance is not None and
019749:                     selected_predicted_distance >= player_reacq_extreme_pred_limit and
019750:                     not top_return_search_context and
019751:                     not back_return_search_context
019752:                 )
019753:                 if player_reacq_extreme_prediction_conflict:
019754:                     self._record_rejected_contour_debug(
019755:                         best_contour,
019756:                         x1,
019757:                         y1,
019758:                         cx,
019759:                         cy,
019760:                         selected_area_for_guard,
019761:                         (
019762:                             f"player-reacq extreme prediction conflict "
019763:                             f"jump={actual_distance:.1f}px "
019764:                             f"pred_dist={selected_predicted_distance:.1f}px"
019765:                         ),
019766:                         source=best_source,
019767:                     )
019768:                     self.stuck_frame_count = max(
019769:                         int(getattr(self, 'stuck_frame_count', 0)) + 1,
019770:                         1,
019771:                     )
019772:                     print(
019773:                         f"Frame {self.frame_count}: "
019774:                         f"[PLAYER-REACQ EXTREME-PREDICTION REJECT] "
019775:                         f"holding {self.ball_center} instead of ({cx},{cy}) "
019776:                         f"jump={actual_distance:.1f}px "
019777:                         f"limit={player_reacq_extreme_jump_limit:.1f}px "
019778:                         f"pred_dist={selected_predicted_distance:.1f}px "
019779:                         f"pred_limit={player_reacq_extreme_pred_limit:.1f}px "
019780:                         f"motion={motion_mean:.1f}/{motion_max:.1f}"
019781:                     )
019782:                     return self.ball_center
019783: 
019784:                 # Clearing the protection now needs either strong image motion
019785:                 # or geometric continuity. This preserves slow nearby tracks
019786:                 # while preventing a 300+ px weak-motion blob from becoming a
019787:                 # new anchor simply because it is outside the player box.
019788:                 # V28: a moving player/racket fragment can satisfy the generic
019789:                 # strong-motion escape even when it is a large, prediction-conflicting
019790:                 # reacquisition jump.  The reviewed early rally does this at f176:
019791:                 # (709,1558), area 249, S=42, jump 301px, pred error 153px, while
019792:                 # the real yellow ball is visibly elsewhere.  Do not weaken the
019793:                 # normal strong-motion path.  Only override when the selected
019794:                 # candidate is simultaneously large-jump, prediction-conflicting,
019795:                 # large-area and low-saturation, and the existing visible-ball
019796:                 # detector independently finds a more saturated moving ball well
019797:                 # away from it in the same frame.
019798:                 # V29: this V28 probe executes before hsv_values/bulb_size are
019799:                 # guaranteed to be bound on every tracking path.  Prefer those
019800:                 # already-bound values when present, otherwise derive evidence
019801:                 # directly from the selected candidate location/contour.
019802:                 player_reacq_candidate_hsv = locals().get('hsv_values')
019803:                 player_reacq_candidate_area = locals().get('bulb_size')
019804:                 try:
019805:                     if (
019806:                             player_reacq_candidate_hsv is not None and
019807:                             len(player_reacq_candidate_hsv) >= 3
019808:                     ):
019809:                         player_reacq_selected_h = int(player_reacq_candidate_hsv[0])
019810:                         player_reacq_selected_s = int(player_reacq_candidate_hsv[1])
019811:                         player_reacq_selected_v = int(player_reacq_candidate_hsv[2])
019812:                     else:
019813:                         # V30: the final per-candidate HSV is not bound yet on the
019814:                         # strong-motion reacquisition path.  Use the exact candidate
019815:                         # center pixel instead of a 5x5 median, which can be dominated
019816:                         # by the surrounding court/player pixels.  This matches the
019817:                         # same selected location that is about to be committed.
019818:                         px = int(cx)
019819:                         py = int(cy)
019820:                         if not (
019821:                                 0 <= px < frame.shape[1] and
019822:                                 0 <= py < frame.shape[0]
019823:                         ):
019824:                             raise ValueError('candidate center outside frame')
019825:                         candidate_pixel = frame[py:py + 1, px:px + 1]
019826:                         if not candidate_pixel.size:
019827:                             raise ValueError('empty candidate center pixel')
019828:                         candidate_pixel_hsv = cv2.cvtColor(
019829:                             candidate_pixel, cv2.COLOR_BGR2HSV
```

## Lines 19906-20126
```python
019906:                                     float(visible_pos[0]) - float(cx),
019907:                                     float(visible_pos[1]) - float(cy),
019908:                                 )
019909:                                 if len(visible_pos) >= 2 else 0.0
019910:                             )
019911:                         except (TypeError, ValueError, IndexError):
019912:                             visible_pos = ()
019913:                             visible_s = -1
019914:                             visible_motion_mean = 0.0
019915:                             visible_motion_max = 0.0
019916:                             visible_conflict_distance = 0.0
019917: 
019918:                         visible_conflict_limit = max(100.0, float(frame.shape[1]) * 0.025)
019919:                         visible_independent = (
019920:                             visible_conflict_distance >= visible_conflict_limit and
019921:                             visible_s >= player_reacq_selected_s + 15 and
019922:                             (
019923:                                 visible_motion_mean >= 5.0 or
019924:                                 visible_motion_max >= 35.0
019925:                             )
019926:                         )
019927:                         if visible_independent:
019928:                             print(
019929:                                 f"Frame {self.frame_count}: "
019930:                                 f"[PLAYER-REACQ STRONG-MOTION VISIBLE-CONFLICT REANCHOR] "
019931:                                 f"rejected=({cx},{cy}) "
019932:                                 f"hsv=({player_reacq_selected_h},"
019933:                                 f"{player_reacq_selected_s},"
019934:                                 f"{player_reacq_selected_v}) "
019935:                                 f"area={player_reacq_selected_area:.1f}px "
019936:                                 f"jump={actual_distance:.1f}px "
019937:                                 f"pred_dist={selected_predicted_distance:.1f}px "
019938:                                 f"visible={visible_pos} visible_s={visible_s} "
019939:                                 f"motion={visible_motion_mean:.1f}/{visible_motion_max:.1f} "
019940:                                 f"conflict={visible_conflict_distance:.1f}px"
019941:                             )
019942:                             recovery_candidate = dict(visible_ball)
019943:                             recovery_candidate['recovery_label'] = (
019944:                                 'PLAYER-REACQ STRONG-MOTION VISIBLE REANCHOR'
019945:                             )
019946:                             return self._commit_night_visible_ball_recovery(
019947:                                 recovery_candidate, frame
019948:                             )
019949: 
019950:                 # V31: V30 proves the selected f176 candidate is a large,
019951:                 # low-saturation, prediction-conflicting fragment, but the broad
019952:                 # visible-ball helper can fail to return an independent candidate
019953:                 # in that exact frame.  Do not then let a sharp player/racket edge
019954:                 # clear the protection merely because motion_max is high.  A real
019955:                 # ball recovery in this sequence has much stronger mean patch
019956:                 # motion; the false fragment is characteristically spiky
019957:                 # (low mean, high max).  In this narrow combination, hold the last
019958:                 # trusted anchor for one additional frame so the genuine ball can
019959:                 # reappear without poisoning direction/bounce state.
019960:                 player_reacq_spiky_visible_conflict_hold = (
019961:                     player_reacq_visible_conflict_probe and
019962:                     actual_distance >= max(280.0, float(frame.shape[1]) * 0.07) and
019963:                     selected_predicted_distance is not None and
019964:                     selected_predicted_distance >= 130.0 and
019965:                     player_reacq_selected_area >= 180.0 and
019966:                     0 <= player_reacq_selected_s <= 55 and
019967:                     motion_mean < 30.0 and
019968:                     motion_max >= 80.0 and
019969:                     not top_return_search_context and
019970:                     not back_return_search_context
019971:                 )
019972:                 if player_reacq_spiky_visible_conflict_hold:
019973:                     self._record_rejected_contour_debug(
019974:                         best_contour,
019975:                         x1,
019976:                         y1,
019977:                         cx,
019978:                         cy,
019979:                         selected_area_for_guard,
019980:                         (
019981:                             f'player-reacq spiky visible-conflict hold '
019982:                             f'jump={actual_distance:.1f}px '
019983:                             f'pred_dist={selected_predicted_distance:.1f}px '
019984:                             f'motion={motion_mean:.1f}/{motion_max:.1f}'
019985:                         ),
019986:                         source=best_source,
019987:                     )
019988:                     self.stuck_frame_count = max(
019989:                         int(getattr(self, 'stuck_frame_count', 0)) + 1,
019990:                         1,
019991:                     )
019992:                     print(
019993:                         f'Frame {self.frame_count}: '
019994:                         f'[PLAYER-REACQ VISIBLE-CONFLICT HOLD] '
019995:                         f'holding {self.ball_center} instead of ({cx},{cy}) '
019996:                         f'hsv=({player_reacq_selected_h},'
019997:                         f'{player_reacq_selected_s},'
019998:                         f'{player_reacq_selected_v}) '
019999:                         f'area={player_reacq_selected_area:.1f}px '
020000:                         f'jump={actual_distance:.1f}px '
020001:                         f'pred_dist={selected_predicted_distance:.1f}px '
020002:                         f'motion={motion_mean:.1f}/{motion_max:.1f}'
020003:                     )
020004:                     return self.ball_center
020005: 
020006:                 if (
020007:                         player_reacq_guard_active and
020008:                         (
020009:                             player_reacq_strong_motion or
020010:                             player_reacq_prediction_close or
020011:                             player_reacq_close_to_anchor
020012:                         ) and
020013:                         self._player_point_zone((cx, cy)) is None
020014:                 ):
020015:                     print(
020016:                         f"Frame {self.frame_count}: [PLAYER-REACQ MOTION CONFIRMED] "
020017:                         f"clearing guard at ({cx},{cy}) motion={motion_mean:.1f}/{motion_max:.1f} "
020018:                         f"jump={actual_distance:.1f}px pred_dist={selected_predicted_distance}"
020019:                     )
020020:                     self._player_reacq_protect_until_frame = -1
020021: 
020022:                 relaxed_contact_min_y = self._contact_reacquire_min_y(
020023:                     contact_reacquire_bounds,
020024:                     selected_predicted_distance,
020025:                     motion_mean,
020026:                     motion_max,
020027:                 ) if contact_reacquire_bounds is not None else None
020028:                 selected_area = selected_area_for_guard
020029:                 player_reacq_static = self._player_reacq_static_candidate(
020030:                     {
020031:                         'pos': (cx, cy),
020032:                         'area': selected_area,
020033:                         'motion_mean': motion_mean,
020034:                         'motion_max': motion_max,
020035:                     },
020036:                     predicted_point=predicted_point,
020037:                 )
020038:                 # The regular contour search can still run after motion
020039:                 # reacquisition returns no candidate.  Apply the same
020040:                 # adjacent-court guard to that path; otherwise a tiny static
020041:                 # blob outside the singles sideline can reset stuck_frame_count
020042:                 # and become the point-ending position.
020043:                 if not player_reacq_static and int(
020044:                         getattr(self, '_player_reacq_protect_until_frame', -1)
020045:                     ) >= self.frame_count:
020046:                     try:
020047:                         outside_court, outside_side, left_x, right_x = self._point_outside_singles_sidelines(
020048:                             (cx, cy), frame
020049:                         )
020050:                     except Exception:
020051:                         outside_court, outside_side, left_x, right_x = False, None, None, None
020052:                     if outside_court:
020053:                         sideline_depth = (
020054:                             (float(left_x) - cx) if outside_side == 'left'
020055:                             else (cx - float(right_x))
020056:                         )
020057:                         if sideline_depth > 35.0 or (motion_mean < 6.0 and motion_max < 25.0):
020058:                             player_reacq_static = True
020059:                 if not player_reacq_static and int(
020060:                         getattr(self, '_player_reacq_protect_until_frame', -1)
020061:                     ) >= self.frame_count:
020062:                     tracker = getattr(self, 'player_tracker', None)
020063:                     try:
020064:                         outside_main_court = (
020065:                             tracker is not None and
020066:                             hasattr(tracker, '_center_is_in_court') and
020067:                             not tracker._center_is_in_court((cx, cy), frame_width=frame.shape[1])
020068:                         )
020069:                     except Exception:
020070:                         outside_main_court = False
020071:                     if outside_main_court:
020072:                         player_reacq_static = True
020073:                 if player_reacq_static:
020074:                     self._record_rejected_contour_debug(
020075:                         best_contour,
020076:                         x1,
020077:                         y1,
020078:                         cx,
020079:                         cy,
020080:                         selected_area,
020081:                         "player-reacq distant static blob",
020082:                         source=best_source,
020083:                     )
020084:                     self.stuck_frame_count = max(
020085:                         int(getattr(self, 'stuck_frame_count', 0)) + 1, 1
020086:                     )
020087:                     print(
020088:                         f"Frame {self.frame_count}: [PLAYER-REACQ STATIC REJECT] "
020089:                         f"holding {self.ball_center} instead of ({cx},{cy}) "
020090:                         f"motion={motion_mean:.1f}/{motion_max:.1f}"
020091:                     )
020092:                     visible_ball = self._find_night_visible_ball_candidate(frame, frame_gray)
020093:                     if visible_ball is not None:
020094:                         print(
020095:                             f"Frame {self.frame_count}: [PLAYER-REACQ VISIBLE BALL] "
020096:                             f"using {visible_ball['pos']} area={visible_ball['area']:.1f}px "
020097:                             f"instead of static candidate ({cx},{cy})"
020098:                         )
020099:                         return self._commit_night_visible_ball_recovery(visible_ball, frame)
020100:                     return self.ball_center
020101:                 if top_return_search_context:
020102:                     if (
020103:                         getattr(self, '_top_return_mode', 'edge') in ('upper_side', 'upper_racket') and
020104:                         best_source not in ('primary', 'regular', 'alt')
020105:                     ):
020106:                         top_return_ok = False
020107:                         top_return_reason = f"top-return source {best_source} not allowed for upper-side reentry"
020108:                     else:
020109:                         top_return_ok, top_return_reason = self._top_return_reentry_ok(
020110:                             (cx, cy), selected_area, motion_mean, motion_max, frame.shape
020111:                         )
020112:                     if not top_return_ok:
020113:                         motion_reentry = self._find_top_return_motion_reentry(frame, frame_gray)
020114:                         if motion_reentry is not None:
020115:                             return self._commit_top_return_reentry(
020116:                                 motion_reentry, "TOP-RETURN MOTION REENTRY"
020117:                             )
020118:                         self._record_rejected_contour_debug(
020119:                             best_contour,
020120:                             x1,
020121:                             y1,
020122:                             cx,
020123:                             cy,
020124:                             selected_area,
020125:                             top_return_reason,
020126:                             source=best_source,
```

