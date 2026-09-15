# V33 post-track state context

## 'def _try_local_ai_recovery' — 1 hit(s)
```python
002458:             else:
002459:                 print(
002460:                     f"[LOCAL_AI_TRAJECTORY] f{self.frame_count}: no safe candidate "
002461:                     f"prev={previous_position} predicted={predicted} radius={roi_radius:.1f}px"
002462:                 )
002463:             return None
002464: 
002465:         point = (int(selected_candidate["x"]), int(selected_candidate["y"]))
002466:         step_dx = float(point[0]) - float(previous_position[0])
002467:         step_dy = float(point[1]) - float(previous_position[1])
002468:         step = math.hypot(step_dx, step_dy)
002469:         selected_candidate["trajectory_rescue"] = True
002470:         selected_candidate["trajectory_prediction"] = predicted
002471:         selected_candidate["trajectory_prediction_error"] = float(selected_row["prediction_error"])
002472:         selected_candidate["trajectory_cost"] = float(selected_row["trajectory_cost"])
002473:         if post_serve_corridor_active:
002474:             selected_candidate["trajectory_post_serve_corridor"] = True
002475:             selected_candidate["trajectory_corridor_along"] = float(selected_row["corridor_along"])
002476:             selected_candidate["trajectory_corridor_cross"] = float(selected_row["corridor_cross"])
002477:         selected_candidate["trajectory_motion"] = {
002478:             "dx": step_dx,
002479:             "dy": step_dy,
002480:             "distance": step,
002481:             "direction_deg": math.degrees(math.atan2(step_dy, step_dx)) if step > 0.0 else None,
002482:         }
002483:         mode_label = " corridor" if post_serve_corridor_active else ""
002484:         print(
002485:             f"[LOCAL_AI_TRAJECTORY_ACCEPT] f{self.frame_count}:{mode_label} "
002486:             f"{point} ai={float(selected_candidate.get('ai_score', 0.0)):.6f} "
002487:             f"pred={predicted} err={selected_row['prediction_error']:.1f}px "
002488:             f"motion={selected_row['motion_mean']:.1f}/{selected_row['motion_max']:.1f} "
002489:             f"zone={selected_row['zone'] or 'clear'} cost={selected_row['trajectory_cost']:.2f}"
002490:         )
002491:         return selected_candidate
002492: 
002493:     def _try_local_ai_recovery(
002494:             self, previous_position, tracked_position, previous_stuck,
002495:             pre_track_snapshot=None, frame=None):
002496:         """Repair a suspect selection from the recent raw-frame buffer if safe."""
002497:         confirmed_static_artifact = self._finalize_pending_provisional_static_candidate()
002498:         if (
002499:                 confirmed_static_artifact and previous_position is not None and
002500:                 tracked_position is not None and pre_track_snapshot is not None and
002501:                 math.hypot(
002502:                     float(tracked_position[0]) - float(getattr(self, "_last_static_guard_position", tracked_position)[0]),
002503:                     float(tracked_position[1]) - float(getattr(self, "_last_static_guard_position", tracked_position)[1]),
002504:                 ) <= 22.0):
002505:             self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
002506:             # Re-run the normal candidate selection now that the confirmed
002507:             # static blob is ignored.  This keeps the real next-frame toss
002508:             # (a prior frame here) instead of deliberately dropping a visible ball.
002509:             retracked = self.track_ball_in_frame(frame) if frame is not None else None
002510:             if retracked is not None and math.hypot(
002511:                     float(retracked[0]) - float(self._last_static_guard_position[0]),
002512:                     float(retracked[1]) - float(self._last_static_guard_position[1]),
002513:             ) > 22.0:
002514:                 print(
002515:                     f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
002516:                     f"retracked {retracked} after excluding static selection"
002517:                 )
002518:                 return retracked
002519:             self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
002520:             self.stuck_frame_count = int(previous_stuck) + 1
002521:             print(
002522:                 f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
002523:                 f"held current static selection {tracked_position}; resuming from {previous_position}"
002524:             )
002525:             return previous_position
002526:         # Keep a weak player/background reacquisition from becoming the
002527:         # next tracking anchor before tight local AI can arbitrate it.
002528:         if (
002529:                 pre_track_snapshot is not None and
002530:                 previous_position is not None and
002531:                 tracked_position is not None and
002532:                 int(previous_stuck or 0) >= 3):
002533:             reacq_jump = math.hypot(
002534:                 float(tracked_position[0]) - float(previous_position[0]),
002535:                 float(tracked_position[1]) - float(previous_position[1]),
002536:             )
002537:             reacq_motion_mean = float(
002538:                 getattr(self, '_last_tracked_candidate_motion_mean', 0.0) or 0.0
002539:             )
002540:             reacq_motion_max = float(
002541:                 getattr(self, '_last_tracked_candidate_motion_max', 0.0) or 0.0
002542:             )
002543:             reacq_zone = self._player_point_zone(tracked_position)
002544:             weak_player_candidate = (
002545:                 reacq_zone is not None and
002546:                 reacq_motion_mean < 8.0 and
002547:                 reacq_motion_max < 50.0
002548:             )
002549:             weak_long_jump = (
002550:                 reacq_jump >= 150.0 and
002551:                 reacq_motion_mean < 12.0 and
002552:                 reacq_motion_max < 80.0
002553:             )
002554:             if weak_player_candidate or weak_long_jump:
002555:                 rejected_position = tuple(tracked_position)
002556:                 self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
002557:                 self.stuck_frame_count = int(previous_stuck or 0) + 1
002558:                 tracked_position = tuple(previous_position)
002559:                 if getattr(self, '_last_motion_reacq_frame', -1000000) == self.frame_count:
002560:                     self._last_motion_reacq_frame = -1000000
002561:                     self._last_motion_reacq_pos = None
002562:                 if weak_player_candidate:
002563:                     print(
002564:                         f'[PLAYER_REACQ_WEAK_PLAYER_REJECTED] f{self.frame_count}: '
002565:                         f'rejected {rejected_position} zone={reacq_zone} '
002566:                         f'motion={reacq_motion_mean:.1f}/{reacq_motion_max:.1f}; '
002567:                         f'restored {previous_position} stuck={self.stuck_frame_count}'
002568:                     )
002569:                 else:
002570:                     print(
002571:                         f'[PLAYER_REACQ_WEAK_JUMP_REJECTED] f{self.frame_count}: '
002572:                         f'rejected {rejected_position} jump={reacq_jump:.1f}px '
002573:                         f'motion={reacq_motion_mean:.1f}/{reacq_motion_max:.1f}; '
002574:                         f'restored {previous_position} stuck={self.stuck_frame_count}'
002575:                     )
002576: 
002577:         tight_hold_recovery = self._try_active_tight_local_ai_hold(
002578:             frame,
002579:             tracked_position,
002580:             previous_gray=(pre_track_snapshot or {}).get('_prev_frame_gray'),
002581:         )
002582:         if tight_hold_recovery is not None:
002583:             return tight_hold_recovery
002584:         reason = self._local_ai_recovery_reason(
002585:             previous_position, tracked_position, previous_stuck
002586:         )
002587:         if reason is None:
002588:             self._maybe_clear_post_serve_pre_net_recovery(
002589:                 previous_position, tracked_position, source='normal-tracker'
002590:             )
002591:             return tracked_position
002592:         post_serve_recovery_active = (
002593:             bool(getattr(self, '_post_serve_pre_net_recovery_active', False)) or
002594:             int(self.frame_count) <= max(
002595:                 int(getattr(self, "_post_serve_launch_lock_until_frame", -1)),
002596:                 int(getattr(self, "_post_serve_recovery_hold_until_frame", -1000000)),
002597:             )
002598:         )
002599:         trajectory_recovered = self._try_local_ai_trajectory_rescue(
002600:             frame,
002601:             previous_position,
002602:             tracked_position,
002603:             previous_stuck,
002604:             pre_track_snapshot,
002605:             reason,
002606:         )
002607:         if trajectory_recovered is not None:
002608:             recovered = trajectory_recovered
002609:             reason = f"{reason}+trajectory"
002610:         else:
002611:             recovery_reason = reason
002612:             if post_serve_recovery_active and reason.startswith("player-region:"):
002613:                 # Keep using the post-serve low-saturation candidate pool on
002614:                 # the frames after the first rejected jump.  The normal tracker
002615:                 # may keep proposing a player contour until the ball emerges.
002616:                 recovery_reason = f"post-serve-launch-occlusion:{reason}"
002617:                 print(
002618:                     f"[POST_SERVE_RECOVERY_RETRY] f{self.frame_count}: "
002619:                     f"{reason} -> {recovery_reason}"
002620:                 )
002621:             recovered = self.local_ai_recovery.recover(
002622:                 self.frame_count,
002623:                 self._local_ai_frame_buffer,
002624:                 predicted_position=previous_position,
002625:                 player_zone=self._player_point_zone,
002626:                 reason=recovery_reason,
002627:                 force=False,
002628:             )
002629:         if recovered is None:
002630:             if getattr(self.local_ai_recovery, "last_rejection", None) == "all-player-body-path":
002631:                 self._local_ai_all_body_rejections = int(
002632:                     getattr(self, "_local_ai_all_body_rejections", 0)
002633:                 ) + 1
002634:                 point_age = int(self.frame_count) - int(
002635:                     getattr(self, "point_start_frame_internal", self.frame_count) or self.frame_count
002636:                 )
002637:                 if (
002638:                         self._local_ai_all_body_rejections >= 1 and
002639:                         not post_serve_recovery_active and
002640:                         # A previously accepted non-body recovery is positive
002641:                         # evidence that this is a real serve/rally.  A later
002642:                         # player-body-only attempt can happen at contact or
002643:                         # occlusion and must not erase that valid point.
002644:                         int(getattr(self, "_local_ai_recovery_count", 0)) == 0 and
002645:                         int(getattr(self, "_point_hit_count", 0)) == 0 and
002646:                         point_age <= 150):
002647:                     self._discard_provisional_serve_from_ai = True
002648:             else:
002649:                 self._local_ai_all_body_rejections = 0
002650:             print(f"[LOCAL_AI_RECOVERY] f{self.frame_count}: no safe path ({reason})")
002651:             # POST_SERVE_STATE_ROLLBACK_V3
002652:             # The reason detector runs after normal HSV tracking has already
002653:             # mutated ball_center/motion/history.  If the post-serve guard says
002654:             # that mutation is unsafe and AI cannot replace it, restore the
002655:             # complete pre-track snapshot immediately.  Returning the previous
002656:             # position makes the main loop log/record the held sane point rather
002657:             # than the rejected player pixel.
002658:             if (
002659:                     post_serve_recovery_active and
002660:                     pre_track_snapshot is not None and
002661:                     previous_position is not None and
002662:                     (
002663:                         reason.startswith("post-serve-launch-") or
002664:                         reason.startswith("player-region:")
002665:                     )):
002666:                 rejected_position = (
002667:                     tuple(tracked_position) if tracked_position is not None else None
002668:                 )
002669:                 self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
002670:                 self.stuck_frame_count = max(
002671:                     int(previous_stuck or 0) + 1,
002672:                     int(getattr(self, "stuck_frame_count", 0) or 0),
002673:                 )
002674:                 if getattr(self, "_last_motion_reacq_frame", -1000000) == self.frame_count:
002675:                     self._last_motion_reacq_frame = -1000000
002676:                     self._last_motion_reacq_pos = None
002677:                 print(
002678:                     f"[POST_SERVE_LAUNCH_ROLLBACK] f{self.frame_count}: "
002679:                     f"rejected={rejected_position} reason={reason}; "
002680:                     f"restored={previous_position} stuck={self.stuck_frame_count} "
002681:                     f"retry_through=f{int(getattr(self, '_post_serve_recovery_hold_until_frame', -1))}"
002682:                 )
002683:                 return tuple(previous_position)
002684:             # If HSV just teleported from a physically coherent ball path into a
002685:             # player region and local AI could not verify a replacement, never let
002686:             # that player pixel become the next-frame anchor. Restore the complete
002687:             # pre-track state (position, size, motion and velocity history) and let
002688:             # the rapid AI retry continue from the last sane trajectory.
002689:             if (
002690:                     reason.startswith("player-region:") and
002691:                     pre_track_snapshot is not None and
002692:                     previous_position is not None and
002693:                     tracked_position is not None):
002694:                 player_jump = math.hypot(
002695:                     float(tracked_position[0]) - float(previous_position[0]),
002696:                     float(tracked_position[1]) - float(previous_position[1]),
002697:                 )
002698:                 prior_motion = pre_track_snapshot.get("last_motion") or {}
002699:                 prior_speed = float(prior_motion.get("distance", 0.0) or 0.0)
002700:                 prior_velocities = pre_track_snapshot.get("ball_velocity_history") or []
002701:                 if prior_velocities:
002702:                     prior_speed = max(
002703:                         prior_speed,
002704:                         max(float(value or 0.0) for value in prior_velocities[-3:]),
002705:                     )
002706:                 jump_limit = max(150.0, min(520.0, prior_speed * 3.2 + 60.0))
002707:                 if player_jump > jump_limit:
002708:                     rejected_position = tuple(tracked_position)
002709:                     self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
002710:                     self.stuck_frame_count = int(previous_stuck) + 1
002711:                     if getattr(self, "_last_motion_reacq_frame", -1000000) == self.frame_count:
002712:                         self._last_motion_reacq_frame = -1000000
002713:                         self._last_motion_reacq_pos = None
002714:                     print(
002715:                         f"[LOCAL_AI_PLAYER_JUMP_ROLLBACK] f{self.frame_count}: "
002716:                         f"rejected {rejected_position} jump={player_jump:.1f}px > "
002717:                         f"{jump_limit:.1f}px; restored {previous_position} "
002718:                         f"stuck={self.stuck_frame_count}"
002719:                     )
002720:                     return previous_position
002721:             # Do not let a local-AI rejection still mutate the track.  In the
002722:             # serve-start state a compact candidate with no inter-frame motion
002723:             # is a likely static highlight.  Restore the state from before
002724:             # HSV committed it, then use the next decoded frame to determine
002725:             # whether that blob persisted in place.
002726:             if (
002727:                     reason.startswith(("provisional-serve-jump:", "static-candidate-jump:")) and
002728:                     pre_track_snapshot is not None and
002729:                     tracked_position is not None and
002730:                     float(getattr(self, "_last_tracked_candidate_motion_mean", 0.0) or 0.0) < 5.0 and
002731:                     float(getattr(self, "_last_tracked_candidate_motion_max", 0.0) or 0.0) < 25.0):
002732:                 current_image = None
002733:                 if self._local_ai_frame_buffer:
002734:                     current_image = self._local_ai_frame_buffer[-1].get("image")
002735:                 self._pending_provisional_static_candidate = {
002736:                     "frame": int(self.frame_count),
002737:                     "pos": tuple(tracked_position),
002738:                     "blob": self._static_blob_near(current_image, tracked_position),
002739:                 }
002740:                 self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
002741:                 self.stuck_frame_count = int(previous_stuck) + 1
002742:                 print(
002743:                     f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
002744:                     f"deferred zero-motion rejected jump {tracked_position}; "
002745:                     "waiting for next-frame persistence"
002746:                 )
002747:                 return previous_position
002748:             return tracked_position
002749:         self._local_ai_all_body_rejections = 0
002750:         repaired_position = (int(recovered["x"]), int(recovered["y"]))
002751:         strict_repaired_reason = self._post_serve_pre_net_recovery_reason(
002752:             previous_position, repaired_position
002753:         )
002754:         if strict_repaired_reason is not None:
002755:             rejected_position = tuple(repaired_position)
002756:             if pre_track_snapshot is not None:
002757:                 self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
002758:             self.stuck_frame_count = max(
002759:                 int(previous_stuck or 0) + 1,
002760:                 int(getattr(self, 'stuck_frame_count', 0) or 0),
002761:             )
002762:             print(
002763:                 f"[POST_SERVE_PRE_NET_AI_REJECT] f{self.frame_count}: "
002764:                 f"rejected={rejected_position} reason={strict_repaired_reason}; "
002765:                 f"restored={previous_position} stuck={self.stuck_frame_count}"
002766:             )
002767:             return tuple(previous_position) if previous_position is not None else None
002768: 
002769:         self._maybe_clear_post_serve_pre_net_recovery(
002770:             previous_position, repaired_position, source='local-ai'
002771:         )
002772:         self.ball_center = repaired_position
002773:         self.ball_size = float(recovered.get("area", self.ball_size or 0.0))
002774:         if recovered.get("trajectory_rescue"):
002775:             trajectory_motion = dict(recovered.get("trajectory_motion") or {})
002776:             prior_motion = (pre_track_snapshot or {}).get("last_motion")
002777:             self.prev_motion = dict(prior_motion) if isinstance(prior_motion, dict) else prior_motion
002778:             self.last_motion = trajectory_motion
002779:             if float(trajectory_motion.get("distance", 0.0) or 0.0) >= 3.0:
002780:                 self.last_nonzero_motion = dict(trajectory_motion)
002781:             prior_history = list((pre_track_snapshot or {}).get("ball_velocity_history") or [])
002782:             prior_history.append(float(trajectory_motion.get("distance", 0.0) or 0.0))
002783:             self.ball_velocity_history = prior_history[-5:]
002784:             self._held_direction_candidate = None
002785:         self.last_seen_frame = self.frame_count
002786:         self.stuck_frame_count = 0
002787:         self._pending_rally_end_reason = None
002788:         self._pending_rally_end_frame = -1
002789:         self._local_ai_recovery_count += 1
002790:         if reason != "post-recovery-follow":
002791:             # A successful repair commonly happens at contact/occlusion.  The
002792:             # next few frames are the outgoing flight, where the normal tracker
002793:             # has the least reliable prediction.  Establish a bounded handoff
002794:             # window; each *verified* local-AI frame can extend the immediate
002795:             # follow-up by four frames, but never beyond this deadline.
002796:             self._local_ai_handoff_deadline_frame = self.frame_count + 14
002797: 
002798:         handoff_deadline = int(getattr(self, "_local_ai_handoff_deadline_frame", -1))
002799:         if handoff_deadline >= self.frame_count:
002800:             self._local_ai_follow_until_frame = min(
002801:                 handoff_deadline,
002802:                 max(
002803:                     int(getattr(self, "_local_ai_follow_until_frame", -1)),
002804:                     self.frame_count + 4,
002805:                 ),
002806:             )
002807:         print(
002808:             f"[LOCAL_AI_RECOVERY] f{self.frame_count}: accepted {repaired_position} "
002809:             f"score={float(recovered.get('ai_score', 0.0)):.6f} "
002810:             f"replayed={self.local_ai_recovery.lookback_frames}f reason={reason}"
002811:         )
002812:         return repaired_position
002813: 
002814:     def _player_serve_context(self, serve_position):
002815:         tracker = getattr(self, "player_tracker", None)
002816:         if tracker is None:
002817:             return {}
002818:         try:
002819:             server_idx = self._current_server_index()
002820:             context = tracker.serve_context(
002821:                 serve_position,
002822:                 player_name=self.player_names[server_idx] if server_idx is not None else "",
002823:             )
002824:             tracker.adopt_serve_context(context, frame_index=int(getattr(self, 'frame_count', -1)))
002825:             return context
002826:         except Exception:
002827:             return {}
002828: 
002829:     def _refresh_player_serve_context(self):
002830:         """Refresh server/receiver geometry while the serve is still starting."""
002831:         tracker = getattr(self, "player_tracker", None)
002832:         context = getattr(self, "_point_history_current", None)
002833:         if tracker is None or context is None:
002834:             return
002835:         serve_context = context.get('serve_context') or {}
002836:         serve_x = serve_context.get('serve_x')
002837:         serve_y = serve_context.get('serve_y')
002838:         if serve_x is None or serve_y is None:
002839:             return
002840:         try:
002841:             previous_receiver = serve_context.get('receiver_position')
002842:             refreshed = self._player_serve_context((serve_x, serve_y))
002843:             if not refreshed:
002844:                 return
002845:             # Do not replace a valid serve-ball coordinate; update the player
002846:             # detections as they become available during the opening frames.
002847:             for key, value in refreshed.items():
002848:                 if value is not None and value != '':
002849:                     serve_context[key] = value
002850:             context['serve_context'] = serve_context
002851:             current_receiver = serve_context.get('receiver_position')
002852:             if current_receiver and current_receiver != previous_receiver:
002853:                 print(
002854:                     f"[SERVE_PLAYERS] f{self.frame_count}: "
002855:                     f"server={serve_context.get('player') or 'unknown'} "
002856:                     f"side={serve_context.get('player_side')} "
002857:                     f"position={serve_context.get('player_position')} "
002858:                     f"receiver_side={serve_context.get('receiver_side')} "
002859:                     f"receiver_position={current_receiver} "
002860:                     f"area={serve_context.get('receiver_area')}"
002861:                 )
002862:         except Exception:
002863:             return
002864: 
002865:     def _draw_player_tracking(self, frame, scale=1.0):
002866:         tracker = getattr(self, "player_tracker", None)
002867:         if tracker is None:
002868:             return frame
002869:         if not getattr(self, "tracking", False) and getattr(self, "_point_history_current", None) is None:
002870:             return frame
002871:         result = frame
002872:         try:
002873:             for side, track in tracker.tracks.items():
002874:                 if track.bbox is None:
002875:                     continue
002876:                 x, y, w, h = track.bbox
002877:                 x1 = int(round(x * scale))
002878:                 y1 = int(round(y * scale))
002879:                 x2 = int(round((x + w) * scale))
002880:                 y2 = int(round((y + h) * scale))
002881:                 color = (255, 180, 40) if side == "near" else (40, 190, 255)
002882:                 thickness = 2 if track.visible else 1
002883:                 cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
002884:                 if track.head is not None:
002885:                     hx, hy = int(track.head[0] * scale), int(track.head[1] * scale)
002886:                     cv2.circle(result, (hx, hy), max(4, int(8 * scale)), color, 1)
002887:                 if track.shoes is not None:
002888:                     sx, sy = int(track.shoes[0] * scale), int(track.shoes[1] * scale)
002889:                     cv2.circle(result, (sx, sy), max(3, int(6 * scale)), color, 1)
002890:                 if track.racket and track.racket.get("line"):
002891:                     lx1, ly1, lx2, ly2 = track.racket["line"]
002892:                     cv2.line(
002893:                         result,
002894:                         (int(lx1 * scale), int(ly1 * scale)),
002895:                         (int(lx2 * scale), int(ly2 * scale)),
002896:                         (0, 165, 255),
002897:                         max(1, int(round(2 * scale))),
002898:                     )
002899:                 label = f"{side} {track.confidence:.2f}"
002900:                 cv2.putText(
002901:                     result,
002902:                     label,
002903:                     (x1, max(15, y1 - 5)),
002904:                     cv2.FONT_HERSHEY_SIMPLEX,
002905:                     max(0.35, 0.5 * scale),
002906:                     color,
002907:                     max(1, int(round(1.5 * scale))),
002908:                     cv2.LINE_AA,
002909:                 )
002910:         except Exception:
002911:             return result
002912:         return result
002913: 
002914:     def log_motion_metrics(self, prev_pos, dx, dy, distance, direction_deg):
002915:         """Log per-frame motion and raise a focus-loss flag when movement spikes."""
002916:         direction_text = f"{direction_deg:+.1f} deg" if direction_deg is not None else "N/A"
002917:         print(f"Frame {self.frame_count}: Movement {distance:.1f}px (dx={dx}, dy={dy}) Direction: {direction_text}")
002918: 
002919:         self.prev_motion = self.last_motion
002920:         self.last_motion = {
002921:             'distance': distance,
002922:             'dx': dx,
002923:             'dy': dy,
002924:             'direction_deg': direction_deg
002925:         }
002926:         if distance >= 3.0:
002927:             self.last_nonzero_motion = dict(self.last_motion)
002928:         angle_diff = None
002929:         forward_vertical_reversal = False
002930:         direction_change_detected = False
002931:         if direction_deg is not None and distance > 5:
002932:             if self.near_edge or getattr(self, 'edge_wait', False):
002933:                 self._record_motion_vector(prev_pos, self.ball_center, direction_deg)
002934:                 self.last_direction = direction_deg
002935:                 return False
002936:             if self.last_direction is not None:
002937:                 threshold_deg = max(0.0, float(getattr(self, 'direction_change_min_degrees', 20.0)))
002938:                 prev_dx = float(self.prev_motion.get('dx', 0.0)) if self.prev_motion is not None else 0.0
002939:                 prev_dy = float(self.prev_motion.get('dy', 0.0)) if self.prev_motion is not None else 0.0
002940:                 delta = abs(direction_deg - self.last_direction) % 360
002941:                 angle_diff = min(delta, 360 - delta)
002942:                 same_forward_x = abs(prev_dx) <= 3.0 or abs(dx) <= 3.0 or (prev_dx * dx) >= -6.0
002943:                 forward_vertical_reversal = (
002944:                     prev_dy >= 4.0 and
002945:                     dy <= -4.0 and
002946:                     same_forward_x
002947:                 )
002948:                 direction_change_detected = forward_vertical_reversal or angle_diff >= threshold_deg
002949:                 if direction_change_detected:
002950:                     change_point = prev_pos if prev_pos is not None else self.ball_center
002951:                     if (getattr(self, 'ground_bounce_count', 0) > 0 and
002952:                             (self.frame_count - getattr(self, 'last_ground_bounce_frame', -1000000)) > 2 and
002953:                             distance > 12 and
002954:                             (angle_diff >= 110.0 or (prev_dx * dx) < -12.0)):
002955:                         if getattr(self, '_direction_only_bounce_preserve_frame', -1000000) == self.frame_count:
002956:                             print(
002957:                                 f"Frame {self.frame_count}: Preserving bounce count through "
002958:                                 f"same-frame direction-only change"
002959:                             )
002960:                         else:
002961:                             print(f"Frame {self.frame_count}: Resetting bounce count after new shot direction change")
002962:                             self.ground_bounce_count = 0
002963:                             self.last_ground_bounce_frame = -1000000
002964:                     self.direction_change_points.append((change_point[0], change_point[1]))
002965:                     self._record_direction_change_event(
002966:                         change_point,
002967:                         prev_dx,
002968:                         prev_dy,
002969:                         dx,
002970:                         dy,
002971:                         float(self.prev_motion.get('distance', 0.0)) if self.prev_motion is not None else 0.0,
002972:                         distance,
002973:                         angle_diff,
002974:                     )
002975:                     self._last_direction_change_frame = self.frame_count
002976:                     self._last_direction_change_point = (int(change_point[0]), int(change_point[1]))
002977:                     self._last_direction_change_angle = float(angle_diff)
002978:                     self._last_direction_change_vertical_reversal = bool(forward_vertical_reversal)
002979:                     if getattr(self, '_last_impact_marker_frame', -1000000) != self.frame_count:
002980:                         self._add_impact_marker(
002981:                             change_point,
002982:                             kind="direction_change",
002983:                             ttl=7,
002984:                             label="direction change",
002985:                         )
002986:                     extra_tag = " vertical-reversal" if forward_vertical_reversal else ""
002987:                     print(
002988:                         f"Frame {self.frame_count}: Direction change detected "
002989:                         f"({angle_diff:.1f} deg, threshold={threshold_deg:.1f}{extra_tag})"
002990:                     )
002991:             self._record_motion_vector(
002992:                 prev_pos,
002993:                 self.ball_center,
002994:                 direction_deg,
002995:                 angle_diff=angle_diff,
002996:                 changed=direction_change_detected,
002997:                 vertical_reversal=forward_vertical_reversal,
002998:             )
002999:             self.last_direction = direction_deg
003000:         self.motion_history.append({
003001:             'frame': self.frame_count,
003002:             'distance': distance,
003003:             'direction_deg': direction_deg,
003004:             'pos': tuple(self.ball_center) if self.ball_center is not None else None,
003005:             'prev_pos': tuple(prev_pos) if prev_pos is not None else None,
003006:         })
003007:         if len(self.motion_history) > 200:
003008:             self.motion_history.pop(0)
003009: 
003010:         # Compare to recent median distance (starting from start_frame)
003011:         if self.focus_loss_frame is None and self.frame_count >= self.start_frame:
003012:             baseline_distances = [
```

## 'repaired_position = (int(recovered["x"]), int(recovered["y"]))' — 1 hit(s)
```python
002715:                         f"[LOCAL_AI_PLAYER_JUMP_ROLLBACK] f{self.frame_count}: "
002716:                         f"rejected {rejected_position} jump={player_jump:.1f}px > "
002717:                         f"{jump_limit:.1f}px; restored {previous_position} "
002718:                         f"stuck={self.stuck_frame_count}"
002719:                     )
002720:                     return previous_position
002721:             # Do not let a local-AI rejection still mutate the track.  In the
002722:             # serve-start state a compact candidate with no inter-frame motion
002723:             # is a likely static highlight.  Restore the state from before
002724:             # HSV committed it, then use the next decoded frame to determine
002725:             # whether that blob persisted in place.
002726:             if (
002727:                     reason.startswith(("provisional-serve-jump:", "static-candidate-jump:")) and
002728:                     pre_track_snapshot is not None and
002729:                     tracked_position is not None and
002730:                     float(getattr(self, "_last_tracked_candidate_motion_mean", 0.0) or 0.0) < 5.0 and
002731:                     float(getattr(self, "_last_tracked_candidate_motion_max", 0.0) or 0.0) < 25.0):
002732:                 current_image = None
002733:                 if self._local_ai_frame_buffer:
002734:                     current_image = self._local_ai_frame_buffer[-1].get("image")
002735:                 self._pending_provisional_static_candidate = {
002736:                     "frame": int(self.frame_count),
002737:                     "pos": tuple(tracked_position),
002738:                     "blob": self._static_blob_near(current_image, tracked_position),
002739:                 }
002740:                 self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
002741:                 self.stuck_frame_count = int(previous_stuck) + 1
002742:                 print(
002743:                     f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
002744:                     f"deferred zero-motion rejected jump {tracked_position}; "
002745:                     "waiting for next-frame persistence"
002746:                 )
002747:                 return previous_position
002748:             return tracked_position
002749:         self._local_ai_all_body_rejections = 0
002750:         repaired_position = (int(recovered["x"]), int(recovered["y"]))
002751:         strict_repaired_reason = self._post_serve_pre_net_recovery_reason(
002752:             previous_position, repaired_position
002753:         )
002754:         if strict_repaired_reason is not None:
002755:             rejected_position = tuple(repaired_position)
002756:             if pre_track_snapshot is not None:
002757:                 self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
002758:             self.stuck_frame_count = max(
002759:                 int(previous_stuck or 0) + 1,
002760:                 int(getattr(self, 'stuck_frame_count', 0) or 0),
002761:             )
002762:             print(
002763:                 f"[POST_SERVE_PRE_NET_AI_REJECT] f{self.frame_count}: "
002764:                 f"rejected={rejected_position} reason={strict_repaired_reason}; "
002765:                 f"restored={previous_position} stuck={self.stuck_frame_count}"
002766:             )
002767:             return tuple(previous_position) if previous_position is not None else None
002768: 
002769:         self._maybe_clear_post_serve_pre_net_recovery(
002770:             previous_position, repaired_position, source='local-ai'
002771:         )
002772:         self.ball_center = repaired_position
002773:         self.ball_size = float(recovered.get("area", self.ball_size or 0.0))
002774:         if recovered.get("trajectory_rescue"):
002775:             trajectory_motion = dict(recovered.get("trajectory_motion") or {})
002776:             prior_motion = (pre_track_snapshot or {}).get("last_motion")
002777:             self.prev_motion = dict(prior_motion) if isinstance(prior_motion, dict) else prior_motion
002778:             self.last_motion = trajectory_motion
002779:             if float(trajectory_motion.get("distance", 0.0) or 0.0) >= 3.0:
002780:                 self.last_nonzero_motion = dict(trajectory_motion)
002781:             prior_history = list((pre_track_snapshot or {}).get("ball_velocity_history") or [])
002782:             prior_history.append(float(trajectory_motion.get("distance", 0.0) or 0.0))
002783:             self.ball_velocity_history = prior_history[-5:]
002784:             self._held_direction_candidate = None
002785:         self.last_seen_frame = self.frame_count
002786:         self.stuck_frame_count = 0
002787:         self._pending_rally_end_reason = None
002788:         self._pending_rally_end_frame = -1
002789:         self._local_ai_recovery_count += 1
002790:         if reason != "post-recovery-follow":
002791:             # A successful repair commonly happens at contact/occlusion.  The
002792:             # next few frames are the outgoing flight, where the normal tracker
002793:             # has the least reliable prediction.  Establish a bounded handoff
002794:             # window; each *verified* local-AI frame can extend the immediate
002795:             # follow-up by four frames, but never beyond this deadline.
002796:             self._local_ai_handoff_deadline_frame = self.frame_count + 14
002797: 
002798:         handoff_deadline = int(getattr(self, "_local_ai_handoff_deadline_frame", -1))
002799:         if handoff_deadline >= self.frame_count:
002800:             self._local_ai_follow_until_frame = min(
002801:                 handoff_deadline,
002802:                 max(
002803:                     int(getattr(self, "_local_ai_follow_until_frame", -1)),
002804:                     self.frame_count + 4,
002805:                 ),
002806:             )
002807:         print(
002808:             f"[LOCAL_AI_RECOVERY] f{self.frame_count}: accepted {repaired_position} "
002809:             f"score={float(recovered.get('ai_score', 0.0)):.6f} "
002810:             f"replayed={self.local_ai_recovery.lookback_frames}f reason={reason}"
002811:         )
002812:         return repaired_position
002813: 
002814:     def _player_serve_context(self, serve_position):
002815:         tracker = getattr(self, "player_tracker", None)
002816:         if tracker is None:
002817:             return {}
002818:         try:
002819:             server_idx = self._current_server_index()
002820:             context = tracker.serve_context(
002821:                 serve_position,
002822:                 player_name=self.player_names[server_idx] if server_idx is not None else "",
002823:             )
002824:             tracker.adopt_serve_context(context, frame_index=int(getattr(self, 'frame_count', -1)))
002825:             return context
002826:         except Exception:
002827:             return {}
002828: 
002829:     def _refresh_player_serve_context(self):
002830:         """Refresh server/receiver geometry while the serve is still starting."""
002831:         tracker = getattr(self, "player_tracker", None)
002832:         context = getattr(self, "_point_history_current", None)
002833:         if tracker is None or context is None:
002834:             return
002835:         serve_context = context.get('serve_context') or {}
002836:         serve_x = serve_context.get('serve_x')
002837:         serve_y = serve_context.get('serve_y')
002838:         if serve_x is None or serve_y is None:
002839:             return
002840:         try:
002841:             previous_receiver = serve_context.get('receiver_position')
002842:             refreshed = self._player_serve_context((serve_x, serve_y))
002843:             if not refreshed:
002844:                 return
002845:             # Do not replace a valid serve-ball coordinate; update the player
002846:             # detections as they become available during the opening frames.
002847:             for key, value in refreshed.items():
002848:                 if value is not None and value != '':
002849:                     serve_context[key] = value
002850:             context['serve_context'] = serve_context
002851:             current_receiver = serve_context.get('receiver_position')
002852:             if current_receiver and current_receiver != previous_receiver:
002853:                 print(
002854:                     f"[SERVE_PLAYERS] f{self.frame_count}: "
002855:                     f"server={serve_context.get('player') or 'unknown'} "
002856:                     f"side={serve_context.get('player_side')} "
002857:                     f"position={serve_context.get('player_position')} "
002858:                     f"receiver_side={serve_context.get('receiver_side')} "
002859:                     f"receiver_position={current_receiver} "
002860:                     f"area={serve_context.get('receiver_area')}"
002861:                 )
002862:         except Exception:
002863:             return
002864: 
002865:     def _draw_player_tracking(self, frame, scale=1.0):
002866:         tracker = getattr(self, "player_tracker", None)
002867:         if tracker is None:
002868:             return frame
002869:         if not getattr(self, "tracking", False) and getattr(self, "_point_history_current", None) is None:
002870:             return frame
002871:         result = frame
002872:         try:
002873:             for side, track in tracker.tracks.items():
002874:                 if track.bbox is None:
002875:                     continue
002876:                 x, y, w, h = track.bbox
002877:                 x1 = int(round(x * scale))
002878:                 y1 = int(round(y * scale))
002879:                 x2 = int(round((x + w) * scale))
002880:                 y2 = int(round((y + h) * scale))
002881:                 color = (255, 180, 40) if side == "near" else (40, 190, 255)
002882:                 thickness = 2 if track.visible else 1
002883:                 cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
002884:                 if track.head is not None:
002885:                     hx, hy = int(track.head[0] * scale), int(track.head[1] * scale)
002886:                     cv2.circle(result, (hx, hy), max(4, int(8 * scale)), color, 1)
002887:                 if track.shoes is not None:
002888:                     sx, sy = int(track.shoes[0] * scale), int(track.shoes[1] * scale)
002889:                     cv2.circle(result, (sx, sy), max(3, int(6 * scale)), color, 1)
002890:                 if track.racket and track.racket.get("line"):
002891:                     lx1, ly1, lx2, ly2 = track.racket["line"]
002892:                     cv2.line(
002893:                         result,
002894:                         (int(lx1 * scale), int(ly1 * scale)),
002895:                         (int(lx2 * scale), int(ly2 * scale)),
002896:                         (0, 165, 255),
002897:                         max(1, int(round(2 * scale))),
002898:                     )
002899:                 label = f"{side} {track.confidence:.2f}"
002900:                 cv2.putText(
002901:                     result,
002902:                     label,
002903:                     (x1, max(15, y1 - 5)),
002904:                     cv2.FONT_HERSHEY_SIMPLEX,
002905:                     max(0.35, 0.5 * scale),
002906:                     color,
002907:                     max(1, int(round(1.5 * scale))),
002908:                     cv2.LINE_AA,
002909:                 )
002910:         except Exception:
002911:             return result
002912:         return result
002913: 
002914:     def log_motion_metrics(self, prev_pos, dx, dy, distance, direction_deg):
002915:         """Log per-frame motion and raise a focus-loss flag when movement spikes."""
002916:         direction_text = f"{direction_deg:+.1f} deg" if direction_deg is not None else "N/A"
002917:         print(f"Frame {self.frame_count}: Movement {distance:.1f}px (dx={dx}, dy={dy}) Direction: {direction_text}")
002918: 
002919:         self.prev_motion = self.last_motion
002920:         self.last_motion = {
002921:             'distance': distance,
002922:             'dx': dx,
002923:             'dy': dy,
002924:             'direction_deg': direction_deg
002925:         }
002926:         if distance >= 3.0:
002927:             self.last_nonzero_motion = dict(self.last_motion)
002928:         angle_diff = None
002929:         forward_vertical_reversal = False
```

## 'def _snapshot_tracking_state_for_provisional_guard' — 1 hit(s)
```python
000987:                         float(getattr(self, "_last_tracked_candidate_motion_max", 0.0) or 0.0) < 25.0):
000988:                     return f"static-candidate-jump:{jump:.0f}px"
000989:             zone = self._player_point_zone(tracked_position)
000990:             # Shoes and small head fragments appear often during ordinary
000991:             # play.  They are already penalized by the normal tracker; do not
000992:             # pay the recovery cost unless the failure is a body/racket
000993:             # takeover, where a ball can genuinely disappear behind a player.
000994:             if zone in ("player_body", "racket_fragment"):
000995:                 return f"player-region:{zone}"
000996:             if previous_position is not None:
000997:                 if jump > 600.0:
000998:                     return f"untrusted-jump:{jump:.0f}px"
000999:         if tracked_position is None and int(previous_stuck) >= 3:
001000:             return "missing-after-stuck"
001001:         if (
001002:             tracked_position is not None and previous_position is not None and
001003:             tuple(tracked_position) == tuple(previous_position) and int(previous_stuck) >= 3
001004:         ):
001005:             return "held-position"
001006:         return None
001007: 
001008:     _PROVISIONAL_BOUNCE_FIELDS = (
001009:         'ground_bounce_count', 'last_ground_bounce_frame',
001010:         '_recent_racket_rebound_bounce_frame', '_ground_bounce_debug_history',
001011:         '_pending_rally_end_reason', '_pending_rally_end_frame',
001012:         'recent_bounce_markers', '_last_impact_marker_frame',
001013:         '_last_impact_marker_pos', '_last_impact_marker_kind',
001014:         'direction_change_events', '_awaiting_serve_bounce',
001015:         '_serve_phase_active', '_serve_phase_closed_frame',
001016:         '_serve_landed_in_current_attempt', '_serve_in_recorded_attempt',
001017:         '_last_serve_bounce_frame', '_last_serve_bounce_point',
001018:         '_last_serve_bounce_net_contact_like', '_last_serve_bounce_was_in',
001019:         'serve_stats',
001020:     )
001021: 
001022:     def _snapshot_tracking_state_for_provisional_guard(self):
001023:         """Save only live tracking fields that a rejected contour can poison."""
001024:         def clone(value):
001025:             if isinstance(value, np.ndarray):
001026:                 return value.copy()
001027:             if isinstance(value, dict):
001028:                 return dict(value)
001029:             if isinstance(value, list):
001030:                 return list(value)
001031:             return value
001032: 
001033:         fields = (
001034:             "ball_center", "ball_size", "ball_hsv", "last_motion", "prev_motion",
001035:             "last_nonzero_motion", "last_delta", "last_direction",
001036:             "ball_velocity_history", "last_seen_frame", "stuck_frame_count",
001037:             "near_edge", "focus_loss_active", "focus_loss_frame",
001038:             "_focus_loss_guard_until_frame", "_recent_max_ball_size",
001039:             "_prev_frame_gray", "_last_tracked_candidate_motion_frame",
001040:             "_last_tracked_candidate_motion_mean", "_last_tracked_candidate_motion_max",
001041:         )
001042:         snapshot = {
001043:             field: clone(getattr(self, field))
001044:             for field in fields
001045:             if hasattr(self, field)
001046:         }
001047:         # HSV can register a bounce before its candidate is accepted.  Position
001048:         # rollback must also undo that candidate's bounce, queued terminal and
001049:         # serve-in accounting, or repeated rejected launches become two bounces.
001050:         snapshot['_provisional_bounce_state'] = copy.deepcopy({
001051:             field: getattr(self, field)
001052:             for field in self._PROVISIONAL_BOUNCE_FIELDS
001053:             if hasattr(self, field)
001054:         })
001055:         return snapshot
001056: 
001057:     def _restore_tracking_state_for_provisional_guard(self, snapshot):
001058:         if not snapshot:
001059:             return
001060:         for field, value in snapshot.items():
001061:             if field != '_provisional_bounce_state':
001062:                 setattr(self, field, copy.deepcopy(value))
001063:         if '_provisional_bounce_state' in snapshot:
001064:             bounce_state = snapshot['_provisional_bounce_state']
001065:             previous_count = int(getattr(self, 'ground_bounce_count', 0))
001066:             for field in self._PROVISIONAL_BOUNCE_FIELDS:
001067:                 if field in bounce_state:
001068:                     setattr(self, field, copy.deepcopy(bounce_state[field]))
001069:                 elif hasattr(self, field):
001070:                     delattr(self, field)
001071:             restored_count = int(getattr(self, 'ground_bounce_count', 0))
001072:             if previous_count != restored_count:
001073:                 print(
001074:                     f'[REJECTED_CANDIDATE_BOUNCE_ROLLBACK] f{self.frame_count}: '
001075:                     f'bounces={previous_count}->{restored_count}; '
001076:                     'restored pre-candidate bounce evidence'
001077:                 )
001078: 
001079:     def _static_blob_near(self, image, position, radius=14):
001080:         """Find a compact HSV blob near ``position`` in one raw frame.
001081: 
001082:         It is deliberately a local confirmation tool, not a second tracker:
001083:         this only decides whether a rejected *provisional serve* blob stayed
001084:         in the same screen location for an adjacent frame.
001085:         """
001086:         if image is None or position is None:
001087:             return None
001088:         try:
001089:             hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
001090:         except cv2.error:
001091:             return None
001092: 
001093:         x, y = int(position[0]), int(position[1])
001094:         height, width = hsv.shape[:2]
001095:         x1, y1 = max(0, x - radius), max(0, y - radius)
001096:         x2, y2 = min(width, x + radius + 1), min(height, y + radius + 1)
001097:         if x2 <= x1 or y2 <= y1:
001098:             return None
001099: 
001100:         specs = []
001101:         if self.hsv_lower is not None and self.hsv_upper is not None:
001102:             specs.append((self.hsv_lower, self.hsv_upper))
001103:         regular = getattr(self, "hsv_regular", None)
001104:         if regular is not None:
001105:             specs.append((regular["lower"], regular["upper"]))
001106: 
001107:         best = None
001108:         for lower, upper in specs:
001109:             mask = cv2.inRange(hsv[y1:y2, x1:x2], lower, upper)
001110:             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
001111:             for contour in contours:
001112:                 area = float(cv2.contourArea(contour))
001113:                 if area < 0.5 or area > 80.0:
001114:                     continue
001115:                 moments = cv2.moments(contour)
001116:                 if moments["m00"] == 0:
001117:                     continue
001118:                 cx = x1 + float(moments["m10"] / moments["m00"])
001119:                 cy = y1 + float(moments["m01"] / moments["m00"])
001120:                 distance = math.hypot(cx - x, cy - y)
001121:                 if distance > radius:
001122:                     continue
001123:                 candidate = {"pos": (cx, cy), "area": area, "distance": distance}
001124:                 if best is None or candidate["distance"] < best["distance"]:
001125:                     best = candidate
001126:         return best
001127: 
001128:     def _finalize_pending_provisional_static_candidate(self):
001129:         """Use the next raw frame to confirm/reject a provisional static blob."""
001130:         pending = getattr(self, "_pending_provisional_static_candidate", None)
001131:         if not pending:
001132:             return False
001133:         pending_frame = int(pending.get("frame", -1))
001134:         if self.frame_count <= pending_frame:
001135:             return False
001136:         # The guard is one-frame only.  An interrupted/seeked decode must not
001137:         # turn this into a global false-point rule.
001138:         self._pending_provisional_static_candidate = None
001139:         if self.frame_count != pending_frame + 1:
001140:             return False
001141: 
001142:         raw_frames = list(getattr(self, "_local_ai_frame_buffer", []))
001143:         current = raw_frames[-1].get("image") if raw_frames else None
001144:         now_blob = self._static_blob_near(current, pending["pos"])
001145:         if now_blob is None:
001146:             print(
001147:                 f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
001148:                 f"candidate {pending['pos']} did not persist; no ignore"
001149:             )
001150:             return False
001151: 
001152:         prior_blob = pending.get("blob")
001153:         prior_pos = prior_blob.get("pos") if prior_blob else pending["pos"]
001154:         displacement = math.hypot(
001155:             now_blob["pos"][0] - prior_pos[0], now_blob["pos"][1] - prior_pos[1]
001156:         )
001157:         if displacement <= 10.0:
001158:             # A three-sample rule in practice: the candidate was already
001159:             # measured as static against the prior frame, and it now survives
001160:             # at the same position in the following frame.  Restrict the
001161:             # ignore to a few frames so a later live ball remains eligible.
001162:             self._learn_ignored_tracking_position(
001163:                 pending["pos"], radius=18, ttl=4,
001164:                 reason="provisional three-frame static artifact",
001165:             )
001166:             self._last_static_guard_position = tuple(pending["pos"])
001167:             print(
001168:                 f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
001169:                 f"rejected persistent blob {pending['pos']} "
001170:                 f"delta={displacement:.1f}px"
001171:             )
001172:             return True
001173:         else:
001174:             print(
001175:                 f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
001176:                 f"candidate moved {displacement:.1f}px; no ignore"
001177:             )
001178:         return False
001179: 
001180:     def _try_active_tight_local_ai_hold(
001181:             self, frame, tracked_position, previous_gray=None):
001182:         """Advance an active contact ROI when normal tracking only holds.
001183: 
001184:         Player-occlusion branches can return the previous marker without
001185:         reaching any candidate-commit hook.  During the bounded tight-ROI
001186:         contact lock, rank that small area directly instead of waiting five
001187:         frames for a broad full-frame reacquisition.
001188:         """
001189:         follow = getattr(self, '_local_ai_tight_roi_follow', None)
001190:         if (
001191:                 not isinstance(follow, dict) or frame is None or
001192:                 int(self.frame_count) > int(follow.get('deadline', -1)) or
001193:                 int(self.frame_count) <= int(follow.get('last_frame', -1)) or
001194:                 int(getattr(self, 'stuck_frame_count', 0)) < 3):
001195:             return None
001196: 
001197:         held_pos = tracked_position or self.ball_center or follow.get('last_pos')
001198:         if held_pos is None:
001199:             return None
001200:         held_pos = (int(held_pos[0]), int(held_pos[1]))
001201:         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
001202:         held_motion = self._candidate_motion_metrics(
001203:             frame_gray, held_pos[0], held_pos[1], previous_gray=previous_gray
001204:         )
001205:         held_mean = (
001206:             float(held_motion.get('mean', 0.0) or 0.0)
001207:             if held_motion else 0.0
001208:         )
001209:         held_max = (
001210:             float(held_motion.get('max', 0.0) or 0.0)
001211:             if held_motion else 0.0
001212:         )
001213:         height, width = frame.shape[:2]
001214:         px = max(0, min(width - 1, held_pos[0]))
001215:         py = max(0, min(height - 1, held_pos[1]))
001216:         proposal = {
001217:             'pos': held_pos,
001218:             'area': float(getattr(self, 'ball_size', 0.0) or 0.0),
001219:             'hsv': cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[py, px],
001220:             'motion_mean': held_mean,
001221:             'motion_max': held_max,
001222:             'source': 'tight_roi_held_marker',
001223:             'recovery_label': 'LOCAL AI TIGHT ROI HOLD',
001224:             'tight_roi_force_rank': True,
001225:         }
001226:         preferred = self._prefer_night_tight_local_ai_candidate(
001227:             frame, frame_gray, proposal, previous_gray=previous_gray
001228:         )
001229:         if preferred.get('source') != 'local_ai_tight_roi':
001230:             return None
001231:         return self._commit_night_visible_ball_recovery(preferred, frame)
001232: 
001233:     def _contact_local_ai_trigger(self, previous_position, tracked_position, snapshot):
001234:         """Return a generic racket-contact takeover reason for suspicious HSV.
001235: 
001236:         The trigger is deliberately player-relative rather than serve-relative:
001237:         it applies to serves, forehands, backhands and either player.  Merely
001238:         being near a player is not enough.  We require either an actual player
001239:         zone selection or a sharp trajectory/size discontinuity inside the
001240:         padded racket-contact corridor.
001241:         """
001242:         current = int(self.frame_count)
001243:         if (
001244:                 getattr(self, 'local_ai_recovery', None) is None or
001245:                 previous_position is None or
001246:                 current <= int(getattr(
001247:                     self, '_contact_local_ai_cooldown_until_frame', -1000000
001248:                 ))):
001249:             return None
001250: 
001251:         previous = (int(previous_position[0]), int(previous_position[1]))
001252:         watch_until = int(getattr(
001253:             self, '_contact_local_ai_watch_until_frame', -1000000
001254:         ))
001255:         if tracked_position is None:
001256:             if current <= watch_until:
001257:                 return 'contact-loss-after-player-proximity'
001258:             return None
001259:         tracked = (int(tracked_position[0]), int(tracked_position[1]))
001260:         prior_motion = dict((snapshot or {}).get('last_motion') or {})
001261:         prior_dx = float(prior_motion.get('dx', 0.0) or 0.0)
001262:         prior_dy = float(prior_motion.get('dy', 0.0) or 0.0)
001263:         prior_speed = float(
001264:             prior_motion.get('distance', math.hypot(prior_dx, prior_dy)) or 0.0
001265:         )
001266:         proposed_dx = float(tracked[0] - previous[0])
001267:         proposed_dy = float(tracked[1] - previous[1])
001268:         proposed_speed = math.hypot(proposed_dx, proposed_dy)
001269: 
001270:         predicted = (
001271:             int(round(float(previous[0]) + prior_dx)),
001272:             int(round(float(previous[1]) + prior_dy)),
001273:         )
001274:         prediction_error = math.hypot(
001275:             float(tracked[0] - predicted[0]), float(tracked[1] - predicted[1])
001276:         )
001277: 
001278:         angle_delta = 0.0
001279:         if prior_speed >= 4.0 and proposed_speed >= 4.0:
001280:             prior_angle = math.degrees(math.atan2(prior_dy, prior_dx))
001281:             proposed_angle = math.degrees(math.atan2(proposed_dy, proposed_dx))
```

## 'def _restore_tracking_state_for_provisional_guard' — 1 hit(s)
```python
001022:     def _snapshot_tracking_state_for_provisional_guard(self):
001023:         """Save only live tracking fields that a rejected contour can poison."""
001024:         def clone(value):
001025:             if isinstance(value, np.ndarray):
001026:                 return value.copy()
001027:             if isinstance(value, dict):
001028:                 return dict(value)
001029:             if isinstance(value, list):
001030:                 return list(value)
001031:             return value
001032: 
001033:         fields = (
001034:             "ball_center", "ball_size", "ball_hsv", "last_motion", "prev_motion",
001035:             "last_nonzero_motion", "last_delta", "last_direction",
001036:             "ball_velocity_history", "last_seen_frame", "stuck_frame_count",
001037:             "near_edge", "focus_loss_active", "focus_loss_frame",
001038:             "_focus_loss_guard_until_frame", "_recent_max_ball_size",
001039:             "_prev_frame_gray", "_last_tracked_candidate_motion_frame",
001040:             "_last_tracked_candidate_motion_mean", "_last_tracked_candidate_motion_max",
001041:         )
001042:         snapshot = {
001043:             field: clone(getattr(self, field))
001044:             for field in fields
001045:             if hasattr(self, field)
001046:         }
001047:         # HSV can register a bounce before its candidate is accepted.  Position
001048:         # rollback must also undo that candidate's bounce, queued terminal and
001049:         # serve-in accounting, or repeated rejected launches become two bounces.
001050:         snapshot['_provisional_bounce_state'] = copy.deepcopy({
001051:             field: getattr(self, field)
001052:             for field in self._PROVISIONAL_BOUNCE_FIELDS
001053:             if hasattr(self, field)
001054:         })
001055:         return snapshot
001056: 
001057:     def _restore_tracking_state_for_provisional_guard(self, snapshot):
001058:         if not snapshot:
001059:             return
001060:         for field, value in snapshot.items():
001061:             if field != '_provisional_bounce_state':
001062:                 setattr(self, field, copy.deepcopy(value))
001063:         if '_provisional_bounce_state' in snapshot:
001064:             bounce_state = snapshot['_provisional_bounce_state']
001065:             previous_count = int(getattr(self, 'ground_bounce_count', 0))
001066:             for field in self._PROVISIONAL_BOUNCE_FIELDS:
001067:                 if field in bounce_state:
001068:                     setattr(self, field, copy.deepcopy(bounce_state[field]))
001069:                 elif hasattr(self, field):
001070:                     delattr(self, field)
001071:             restored_count = int(getattr(self, 'ground_bounce_count', 0))
001072:             if previous_count != restored_count:
001073:                 print(
001074:                     f'[REJECTED_CANDIDATE_BOUNCE_ROLLBACK] f{self.frame_count}: '
001075:                     f'bounces={previous_count}->{restored_count}; '
001076:                     'restored pre-candidate bounce evidence'
001077:                 )
001078: 
001079:     def _static_blob_near(self, image, position, radius=14):
001080:         """Find a compact HSV blob near ``position`` in one raw frame.
001081: 
001082:         It is deliberately a local confirmation tool, not a second tracker:
001083:         this only decides whether a rejected *provisional serve* blob stayed
001084:         in the same screen location for an adjacent frame.
001085:         """
001086:         if image is None or position is None:
001087:             return None
001088:         try:
001089:             hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
001090:         except cv2.error:
001091:             return None
001092: 
001093:         x, y = int(position[0]), int(position[1])
001094:         height, width = hsv.shape[:2]
001095:         x1, y1 = max(0, x - radius), max(0, y - radius)
001096:         x2, y2 = min(width, x + radius + 1), min(height, y + radius + 1)
001097:         if x2 <= x1 or y2 <= y1:
001098:             return None
001099: 
001100:         specs = []
001101:         if self.hsv_lower is not None and self.hsv_upper is not None:
001102:             specs.append((self.hsv_lower, self.hsv_upper))
001103:         regular = getattr(self, "hsv_regular", None)
001104:         if regular is not None:
001105:             specs.append((regular["lower"], regular["upper"]))
001106: 
001107:         best = None
001108:         for lower, upper in specs:
001109:             mask = cv2.inRange(hsv[y1:y2, x1:x2], lower, upper)
001110:             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
001111:             for contour in contours:
001112:                 area = float(cv2.contourArea(contour))
001113:                 if area < 0.5 or area > 80.0:
001114:                     continue
001115:                 moments = cv2.moments(contour)
001116:                 if moments["m00"] == 0:
001117:                     continue
001118:                 cx = x1 + float(moments["m10"] / moments["m00"])
001119:                 cy = y1 + float(moments["m01"] / moments["m00"])
001120:                 distance = math.hypot(cx - x, cy - y)
001121:                 if distance > radius:
001122:                     continue
001123:                 candidate = {"pos": (cx, cy), "area": area, "distance": distance}
001124:                 if best is None or candidate["distance"] < best["distance"]:
001125:                     best = candidate
001126:         return best
001127: 
001128:     def _finalize_pending_provisional_static_candidate(self):
001129:         """Use the next raw frame to confirm/reject a provisional static blob."""
001130:         pending = getattr(self, "_pending_provisional_static_candidate", None)
001131:         if not pending:
001132:             return False
001133:         pending_frame = int(pending.get("frame", -1))
001134:         if self.frame_count <= pending_frame:
001135:             return False
001136:         # The guard is one-frame only.  An interrupted/seeked decode must not
001137:         # turn this into a global false-point rule.
001138:         self._pending_provisional_static_candidate = None
001139:         if self.frame_count != pending_frame + 1:
001140:             return False
001141: 
001142:         raw_frames = list(getattr(self, "_local_ai_frame_buffer", []))
001143:         current = raw_frames[-1].get("image") if raw_frames else None
001144:         now_blob = self._static_blob_near(current, pending["pos"])
001145:         if now_blob is None:
001146:             print(
001147:                 f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
001148:                 f"candidate {pending['pos']} did not persist; no ignore"
001149:             )
001150:             return False
001151: 
001152:         prior_blob = pending.get("blob")
001153:         prior_pos = prior_blob.get("pos") if prior_blob else pending["pos"]
001154:         displacement = math.hypot(
001155:             now_blob["pos"][0] - prior_pos[0], now_blob["pos"][1] - prior_pos[1]
001156:         )
001157:         if displacement <= 10.0:
001158:             # A three-sample rule in practice: the candidate was already
001159:             # measured as static against the prior frame, and it now survives
001160:             # at the same position in the following frame.  Restrict the
001161:             # ignore to a few frames so a later live ball remains eligible.
001162:             self._learn_ignored_tracking_position(
001163:                 pending["pos"], radius=18, ttl=4,
001164:                 reason="provisional three-frame static artifact",
001165:             )
001166:             self._last_static_guard_position = tuple(pending["pos"])
001167:             print(
001168:                 f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
001169:                 f"rejected persistent blob {pending['pos']} "
001170:                 f"delta={displacement:.1f}px"
001171:             )
001172:             return True
001173:         else:
001174:             print(
001175:                 f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
001176:                 f"candidate moved {displacement:.1f}px; no ignore"
001177:             )
001178:         return False
001179: 
001180:     def _try_active_tight_local_ai_hold(
001181:             self, frame, tracked_position, previous_gray=None):
001182:         """Advance an active contact ROI when normal tracking only holds.
001183: 
001184:         Player-occlusion branches can return the previous marker without
001185:         reaching any candidate-commit hook.  During the bounded tight-ROI
001186:         contact lock, rank that small area directly instead of waiting five
001187:         frames for a broad full-frame reacquisition.
001188:         """
001189:         follow = getattr(self, '_local_ai_tight_roi_follow', None)
001190:         if (
001191:                 not isinstance(follow, dict) or frame is None or
001192:                 int(self.frame_count) > int(follow.get('deadline', -1)) or
001193:                 int(self.frame_count) <= int(follow.get('last_frame', -1)) or
001194:                 int(getattr(self, 'stuck_frame_count', 0)) < 3):
001195:             return None
001196: 
001197:         held_pos = tracked_position or self.ball_center or follow.get('last_pos')
001198:         if held_pos is None:
001199:             return None
001200:         held_pos = (int(held_pos[0]), int(held_pos[1]))
001201:         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
001202:         held_motion = self._candidate_motion_metrics(
001203:             frame_gray, held_pos[0], held_pos[1], previous_gray=previous_gray
001204:         )
001205:         held_mean = (
001206:             float(held_motion.get('mean', 0.0) or 0.0)
001207:             if held_motion else 0.0
001208:         )
001209:         held_max = (
001210:             float(held_motion.get('max', 0.0) or 0.0)
001211:             if held_motion else 0.0
001212:         )
001213:         height, width = frame.shape[:2]
001214:         px = max(0, min(width - 1, held_pos[0]))
001215:         py = max(0, min(height - 1, held_pos[1]))
001216:         proposal = {
001217:             'pos': held_pos,
001218:             'area': float(getattr(self, 'ball_size', 0.0) or 0.0),
001219:             'hsv': cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[py, px],
001220:             'motion_mean': held_mean,
001221:             'motion_max': held_max,
001222:             'source': 'tight_roi_held_marker',
001223:             'recovery_label': 'LOCAL AI TIGHT ROI HOLD',
001224:             'tight_roi_force_rank': True,
001225:         }
001226:         preferred = self._prefer_night_tight_local_ai_candidate(
001227:             frame, frame_gray, proposal, previous_gray=previous_gray
001228:         )
001229:         if preferred.get('source') != 'local_ai_tight_roi':
001230:             return None
001231:         return self._commit_night_visible_ball_recovery(preferred, frame)
001232: 
001233:     def _contact_local_ai_trigger(self, previous_position, tracked_position, snapshot):
001234:         """Return a generic racket-contact takeover reason for suspicious HSV.
001235: 
001236:         The trigger is deliberately player-relative rather than serve-relative:
001237:         it applies to serves, forehands, backhands and either player.  Merely
001238:         being near a player is not enough.  We require either an actual player
001239:         zone selection or a sharp trajectory/size discontinuity inside the
001240:         padded racket-contact corridor.
001241:         """
001242:         current = int(self.frame_count)
001243:         if (
001244:                 getattr(self, 'local_ai_recovery', None) is None or
001245:                 previous_position is None or
001246:                 current <= int(getattr(
001247:                     self, '_contact_local_ai_cooldown_until_frame', -1000000
001248:                 ))):
001249:             return None
001250: 
001251:         previous = (int(previous_position[0]), int(previous_position[1]))
001252:         watch_until = int(getattr(
001253:             self, '_contact_local_ai_watch_until_frame', -1000000
001254:         ))
001255:         if tracked_position is None:
001256:             if current <= watch_until:
001257:                 return 'contact-loss-after-player-proximity'
001258:             return None
001259:         tracked = (int(tracked_position[0]), int(tracked_position[1]))
001260:         prior_motion = dict((snapshot or {}).get('last_motion') or {})
001261:         prior_dx = float(prior_motion.get('dx', 0.0) or 0.0)
001262:         prior_dy = float(prior_motion.get('dy', 0.0) or 0.0)
001263:         prior_speed = float(
001264:             prior_motion.get('distance', math.hypot(prior_dx, prior_dy)) or 0.0
001265:         )
001266:         proposed_dx = float(tracked[0] - previous[0])
001267:         proposed_dy = float(tracked[1] - previous[1])
001268:         proposed_speed = math.hypot(proposed_dx, proposed_dy)
001269: 
001270:         predicted = (
001271:             int(round(float(previous[0]) + prior_dx)),
001272:             int(round(float(previous[1]) + prior_dy)),
001273:         )
001274:         prediction_error = math.hypot(
001275:             float(tracked[0] - predicted[0]), float(tracked[1] - predicted[1])
001276:         )
001277: 
001278:         angle_delta = 0.0
001279:         if prior_speed >= 4.0 and proposed_speed >= 4.0:
001280:             prior_angle = math.degrees(math.atan2(prior_dy, prior_dx))
001281:             proposed_angle = math.degrees(math.atan2(proposed_dy, proposed_dx))
001282:             raw_delta = abs(proposed_angle - prior_angle) % 360.0
001283:             angle_delta = min(raw_delta, 360.0 - raw_delta)
001284: 
001285:         zone = self._player_point_zone(tracked)
001286:         contact_near = any(
001287:             self._point_in_player_contact_corridor(point)
001288:             for point in (previous, tracked, predicted)
001289:         )
001290:         watch_active = current <= watch_until
001291:         if zone is not None or contact_near:
001292:             self._contact_local_ai_watch_until_frame = max(watch_until, current + 8)
001293:             watch_active = True
001294: 
001295:         # Generic trajectory-break recovery. Do not invoke AI merely because
001296:         # the ball changed direction: real tennis hits and bounces do that all
001297:         # the time. Escalate only when the normal detector has just accepted a
001298:         # predicted-path background hotspot, local image motion is extremely
001299:         # weak, the turn is large, and there is no active serve/rally/bounce
001300:         # contact grace explaining the change.
001301:         trajectory_hotspot = getattr(self, '_trajectory_local_ai_hotspot', None)
001302:         if (
001303:                 isinstance(trajectory_hotspot, dict) and
001304:                 int(trajectory_hotspot.get('frame', -1)) == current):
001305:             hotspot_motion_mean = float(trajectory_hotspot.get('motion_mean', 999.0) or 0.0)
001306:             hotspot_motion_max = float(trajectory_hotspot.get('motion_max', 999.0) or 0.0)
001307: 
001308:             # A predicted-path candidate that is nearly static in the image and
001309:             # simultaneously creates a large trajectory break is suspicious on
001310:             # its own. Do not suppress this because a player/contact watch is
001311:             # active: that is exactly when Local AI can protect the trajectory.
001312:             strong_hotspot_break = (
001313:                 prior_speed >= 14.0 and proposed_speed >= 14.0 and
001314:                 angle_delta >= 50.0 and prediction_error >= 12.0 and
001315:                 hotspot_motion_mean <= 3.0 and hotspot_motion_max <= 12.0
001316:             )
```

## 'tracked_position = self._try_local_ai_recovery(' — 1 hit(s)
```python
026747:                                 )) == int(self.frame_count)
026748:                             )
026749:                             if verified_lower_contact_launch:
026750:                                 # A dedicated NIGHT LOWER CONTACT LAUNCH has already
026751:                                 # committed a geometrically verified outbound ball.
026752:                                 # Do not let a Contact Local-AI miss restore the stale
026753:                                 # pre-contact anchor on this exact frame.
026754:                                 self._contact_local_ai_state = None
026755:                                 self._contact_local_ai_cooldown_until_frame = max(
026756:                                     int(getattr(
026757:                                         self,
026758:                                         '_contact_local_ai_cooldown_until_frame',
026759:                                         -1000000,
026760:                                     )),
026761:                                     int(self.frame_count) + 2,
026762:                                 )
026763:                                 self._contact_normal_fallback_path = []
026764:                                 contact_reason = None
026765:                                 print(
026766:                                     f"[CONTACT_LOCAL_AI VERIFIED-LOWER-LAUNCH RELEASE] "
026767:                                     f"f{self.frame_count}: preserving normal launch="
026768:                                     f"{tuple(tracked_position)}"
026769:                                 )
026770:                             else:
026771:                                 contact_reason = self._contact_local_ai_trigger(
026772:                                     prev_ball_center, tracked_position, pre_track_snapshot
026773:                                 )
026774:                             if contact_reason is not None:
026775:                                 rejected_hsv = (
026776:                                     tuple(tracked_position)
026777:                                     if tracked_position is not None else None
026778:                                 )
026779:                                 post_track_snapshot = self._snapshot_tracking_state_for_provisional_guard()
026780:                                 self._restore_tracking_state_for_provisional_guard(
026781:                                     pre_track_snapshot
026782:                                 )
026783:                                 self._local_ai_tight_roi_previous_gray = (
026784:                                     pre_track_snapshot.get('_prev_frame_gray')
026785:                                 )
026786:                                 # Debug only: preserve the normal detector's contact
026787:                                 # hypothesis long enough to print candidate agreement.
026788:                                 self._contact_local_ai_debug_normal_candidate = rejected_hsv
026789:                                 tracked_position = self._run_contact_local_ai_frame(
026790:                                     frame, prev_ball_center, arm_reason=contact_reason
026791:                                 )
026792:                                 self._contact_local_ai_debug_normal_candidate = None
026793:                                 if tracked_position is not None:
026794:                                     contact_local_ai = True
026795:                                     print(
026796:                                         f"[CONTACT_LOCAL_AI_REPLACE] f{self.frame_count}: "
026797:                                         f"HSV={rejected_hsv} -> AI={tracked_position}"
026798:                                     )
026799:                                 else:
026800:                                     # AI is an assist: if it has no accepted candidate,
026801:                                     # keep the normal tracker result for this frame.
026802:                                     self._restore_tracking_state_for_provisional_guard(
026803:                                         pre_track_snapshot
026804:                                     )
026805:                                     tracked_position = prev_ball_center
026806:                                     contact_local_ai = False
026807:                                     print(
026808:                                         f"[CONTACT_LOCAL_AI_FALLBACK] f{self.frame_count}: "
026809:                                         f"AI miss; preserving trusted position={tracked_position}"
026810:                                     )
026811:                     if self.local_ai_recovery is not None and self._local_ai_frame_buffer:
026812:                         self._local_ai_frame_buffer[-1]["normal_position"] = (
026813:                             tuple(tracked_position) if tracked_position is not None else None
026814:                         )
026815:                     self._debug_local_ai_shadow_frame(frame, prev_ball_center, tracked_position)
026816:                     if not forced_local_ai and not contact_local_ai:
026817:                         tracked_position = self._try_local_ai_recovery(
026818:                             prev_ball_center, tracked_position, prev_stuck,
026819:                             pre_track_snapshot=pre_track_snapshot,
026820:                             frame=frame,
026821:                         )
026822:                     self._local_ai_tight_roi_previous_gray = None
026823:                     # Reject any position that jumps impossibly far in one frame (false positive).
026824:                     # When the tracker is in re-acquisition mode (stuck >= 5 before the call), allow
026825:                     # a larger jump because the ball may have traveled far while lost.
026826:                     if tracked_position and prev_ball_center:
026827:                         jump = math.hypot(tracked_position[0] - prev_ball_center[0],
026828:                                           tracked_position[1] - prev_ball_center[1])
026829:                         motion_reacq_this_frame = (
026830:                             getattr(self, '_last_motion_reacq_frame', -1000000) == self.frame_count and
026831:                             getattr(self, '_last_motion_reacq_pos', None) == tracked_position
026832:                         )
026833:                         trusted_rally_launch = (
026834:                             getattr(self, '_trusted_rally_launch_frame', -1000000) == self.frame_count and
026835:                             getattr(self, '_trusted_rally_launch_pos', None) == tracked_position
026836:                         )
026837:                         # Allow a larger jump when re-acquiring after being stuck for 5+ frames:
026838:                         # a racket hit can send the ball 800+ px in one frame, so we use 1500px
026839:                         # to let motion-based re-acquisition recover across the full court.
026840:                         max_jump = 1500 if (
026841:                             prev_stuck >= 5 or prev_top_return_wait or prev_back_return_wait or
026842:                             motion_reacq_this_frame or trusted_rally_launch
026843:                         ) else 400
026844:                         if jump > max_jump:
026845:                             print(f"[JUMP_REJECTED] f{self.frame_count}: jumped {jump:.0f}px from {prev_ball_center} to {tracked_position} (limit={max_jump}px, prev_stuck={prev_stuck}), keeping previous")
026846:                             self.ball_center = prev_ball_center
026847:                             self.stuck_frame_count = max(self.stuck_frame_count, prev_stuck + 1)
026848:                             tracked_position = prev_ball_center
026849: 
026850:                     # Validate only the position that survived the physical
026851:                     # jump gate above.  Previously the provisional serve
026852:                     # state consumed a 425px false contour at a prior frame before
026853:                     # that contour was rejected below.  Its bogus vector then
026854:                     # hid the real upward reversal at a prior frame.
026855:                     # Once a far toss is independently marked as caught, it
026856:                     # becomes a *history* question only.  Do not continue to
026857:                     # mutate the live provisional flags: the previous attempt
026858:                     # cleared those flags at a prior frame and changed the otherwise
026859:                     # stable HSV path of the following real rally.
026860:                     tainted_far_toss_active = (
026861:                         isinstance(
026862:                             getattr(self, '_tainted_provisional_serve_start', None),
026863:                             dict,
026864:                         ) and
026865:                         getattr(self, '_provisional_serve_start_kind', None) ==
026866:                         'far-top-post-hit'
026867:                     )
026868:                     provisional_kind_before_validation = getattr(
026869:                         self, '_provisional_serve_start_kind', None
026870:                     )
026871:                     provisional_reject_reason = (
026872:                         None if tainted_far_toss_active else
026873:                         self._validate_provisional_serve_start(
026874:                             prev_ball_center, tracked_position
026875:                         )
026876:                     )
026877:                     # A pending far-toss token is solely a fallback for a
026878:                     # *rejected* practice toss.  Once this tentative flight
026879:                     # has earned three coherent netward steps, it is a real
026880:                     # serve (for example a prior frame -> a prior frame) and the token must
026881:                     # not survive its fault/end state.  Leaving it alive
026882:                     # caused the local model to re-use that old start at
026883:                     # a prior frame and inject a second, false point.
026884:                     if provisional_reject_reason is not None:
026885:                         provisional_kind = getattr(self, '_provisional_serve_start_kind', None)
026886:                         rejected_start = self._current_history_serve_start_frame()
026887:                         if provisional_kind == 'far-top-post-hit':
026888:                             # This is a caught far-side toss.  Do *not* reset
026889:                             # to WAITING_FOR_SERVE: that was the rejected
026890:                             # approach and it changed later, already-verified
026891:                             # points.  Keep the normal HSV flow alive and ask
026892:                             # local AI to prove a later outgoing path in
026893:                             # shadow mode.  Only proven evidence may alter the
026894:                             # final history/serve-fault interpretation.
026895:                             self._tainted_provisional_serve_start = {
026896:                                 'start_frame': rejected_start,
026897:                                 'detected_frame': int(self.frame_count),
026898:                                 'anchor': tuple(prev_ball_center),
026899:                                 'reason': provisional_reject_reason,
026900:                             }
026901:                             print(
026902:                                 f"[SERVE_START_TAINTED] f{self.frame_count}: "
026903:                                 f"keeping tracker/provisional state for start f{rejected_start}; "
026904:                                 f"{provisional_reject_reason}"
026905:                             )
026906:                         else:
026907:                             print(
026908:                                 f"[SERVE_START_REJECTED] f{self.frame_count}: "
026909:                                 f"discarding provisional start f{rejected_start}; "
026910:                                 f"{provisional_reject_reason}"
026911:                             )
026912:                             self._point_history_current = None
026913:                             self._provisional_serve_start_kind = None
026914:                             self._serve_start_requires_confirmation = False
026915:                             game_state = "WAITING_FOR_SERVE"
026916:                             clear_waiting_serve_history()
026917:                             reset_tracking_state()
026918:                             continue
026919:                     # Shadow verification is intentionally side-effect free
026920:                     # for ball position/game state.  It only supplies a
026921:                     # later verified launch frame for history/fault scoring.
026922:                     reconcile_tainted_far_serve_start()
026923:                     if getattr(self, "_discard_provisional_serve_from_ai", False):
026924:                         self._discard_provisional_serve_from_ai = False
026925:                         self._ignore_unconfirmed_serve_start_result(
026926:                             "local AI rejected repeated all-player-body recovery path"
026927:                         )
026928:                         print(
026929:                             f"[SERVE_START_IGNORED] f{self.frame_count}: "
026930:                             "discarding false provisional serve after repeated local-AI body paths"
026931:                         )
026932:                         game_state = "WAITING_FOR_SERVE"
026933:                         clear_waiting_serve_history()
026934:                         reset_tracking_state()
026935:                         continue
026936:                 if tracked_position:
026937:                     vel = self.last_motion['distance'] if self.last_motion else 0
026938:                     size_text = f"{self.ball_size:.1f}px" if self.ball_size is not None else "unknown"
026939:                     vel_hist_tail = [round(v, 1) for v in getattr(self, 'ball_velocity_history', [])[-5:]]
026940:                     print(f"Frame {self.frame_count}: Ball tracked at {tracked_position} - Size: {size_text}")
026941:                     print(f"[TRACK] f{self.frame_count}: pos={tracked_position} vel={vel:.1f}px stuck={self.stuck_frame_count} vel_hist={vel_hist_tail}")
026942:                     # Infer a visually missed return before any queued point-end is scored.
026943:                     self._maybe_infer_return_contact_from_side_recross(tracked_position, frame)
026944:                     if self._point_history_current is not None:
026945:                         self._point_history_current.setdefault('tracking_trace', []).append({
026946:                             'frame': int(self.frame_count),
026947:                             'pos': [int(tracked_position[0]), int(tracked_position[1])],
026948:                             'size': float(self.ball_size) if self.ball_size is not None else None,
026949:                             'stuck': int(self.stuck_frame_count),
026950:                         })
026951:                     if self.ball_dataset_exporter is not None:
026952:                         self.ball_dataset_exporter.write_frame(
026953:                             frame,
026954:                             source_frame=self.frame_count,
026955:                             ball_center=tracked_position,
026956:                             ball_area=self.ball_size,
026957:                             stuck_frames=self.stuck_frame_count,
026958:                             motion_distance=(self.last_motion or {}).get('distance'),
026959:                             tracking_active=self.tracking,
026960:                             point_index=(
026961:                                 self._point_history_current.get('point_index')
026962:                                 if self._point_history_current is not None else None
026963:                             ),
026964:                         )
026965: 
026966:                     pending_reason = getattr(self, '_pending_rally_end_reason', None)
026967:                     if pending_reason and reference_target_hold:
026968:                         print(
026969:                             f"Frame {self.frame_count}: [REFERENCE_POINT_HOLD] "
026970:                             f"discarding premature queued end '{pending_reason}' "
026971:                             f"until f{reference_target_frame}"
026972:                         )
026973:                         self._pending_rally_end_reason = None
026974:                         self._pending_rally_end_frame = -1
026975:                         pending_reason = None
026976:                     if pending_reason:
026977:                         # Before scoring a service-box fault, prefer a separate
026978:                         # moving ball that continues the incoming trajectory.
026979:                         # This catches the f1785 stale-marker regression without
026980:                         # weakening genuine bounce reversals.
026981:                         if self._resume_false_serve_bounce_from_motion(
026982:                             pending_reason, tracked_position
026983:                         ):
026984:                             self._pending_rally_end_reason = None
026985:                             self._pending_rally_end_frame = -1
026986:                             continue
026987:                         # Point-end detection can queue an out reason one or
026988:                         # more frames before the stuck timeout is reached.  If
026989:                         # this is still an unconfirmed serve toss with no
026990:                         # racket/net contact and the marker has stopped, drop
026991:                         # the queued false end instead of writing a bogus row.
026992:                         suppressed_frame = int(
026993:                             getattr(self, '_last_out_bounce_suppressed_frame', -1000000)
026994:                         )
026995:                         suppressed_point = getattr(self, '_last_out_bounce_suppressed_point', None)
026996:                         recent_static_serve_abort = (
```

