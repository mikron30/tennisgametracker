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
002769:         # V33: Local-AI recovery runs after the normal tracker has already
002770:         # committed its candidate.  If AI replaces that candidate, repair the
002771:         # *entire* live tracking state atomically.  Keeping only the repaired
002772:         # coordinate leaves last_motion / velocity history from the rejected
002773:         # contour and poisons the next-frame prediction.
002774:         committed_position = (
002775:             tuple(tracked_position) if tracked_position is not None else None
002776:         )
002777:         recovery_replaced_committed_candidate = (
002778:             pre_track_snapshot is not None and
002779:             previous_position is not None and
002780:             committed_position is not None and
002781:             committed_position != tuple(repaired_position)
002782:         )
002783:         recovered_area = float(recovered.get("area", self.ball_size or 0.0))
002784: 
002785:         if recovery_replaced_committed_candidate:
002786:             self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
002787:             self._maybe_clear_post_serve_pre_net_recovery(
002788:                 previous_position, repaired_position, source='local-ai'
002789:             )
002790: 
002791:             repaired_dx = float(repaired_position[0]) - float(previous_position[0])
002792:             repaired_dy = float(repaired_position[1]) - float(previous_position[1])
002793:             repaired_distance = math.hypot(repaired_dx, repaired_dy)
002794:             repaired_direction = (
002795:                 math.degrees(math.atan2(repaired_dy, repaired_dx))
002796:                 if repaired_distance > 0.0 else None
002797:             )
002798:             repaired_motion = {
002799:                 'distance': repaired_distance,
002800:                 'dx': repaired_dx,
002801:                 'dy': repaired_dy,
002802:                 'direction_deg': repaired_direction,
002803:             }
002804: 
002805:             self.ball_center = repaired_position
002806:             self.ball_size = recovered_area
002807:             prior_motion = pre_track_snapshot.get("last_motion")
002808:             self.prev_motion = (
002809:                 dict(prior_motion) if isinstance(prior_motion, dict) else prior_motion
002810:             )
002811:             self.last_motion = repaired_motion
002812:             if repaired_distance >= 3.0:
002813:                 self.last_nonzero_motion = dict(repaired_motion)
002814:             prior_history = list(pre_track_snapshot.get("ball_velocity_history") or [])
002815:             prior_history.append(repaired_distance)
002816:             self.ball_velocity_history = prior_history[-5:]
002817:             self._held_direction_candidate = None
002818: 
002819:             # The rollback restores the prior gray frame as well.  For an
002820:             # accepted current-frame repair, the next frame must compare against
002821:             # this decoded frame, not against frame N-1 twice.
002822:             if frame is not None:
002823:                 try:
002824:                     self._prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
002825:                 except cv2.error:
002826:                     pass
002827: 
002828:             if (
002829:                 getattr(self, '_last_motion_reacq_frame', -1000000) == self.frame_count and
002830:                 getattr(self, '_last_motion_reacq_pos', None) == committed_position
002831:             ):
002832:                 self._last_motion_reacq_frame = -1000000
002833:                 self._last_motion_reacq_pos = None
002834: 
002835:             # Replace a same-frame motion-history sample from the rejected
002836:             # contour rather than leaving two contradictory versions of frame N.
002837:             if self.motion_history and int(self.motion_history[-1].get('frame', -1)) == int(self.frame_count):
002838:                 self.motion_history[-1] = {
002839:                     'frame': self.frame_count,
002840:                     'distance': repaired_distance,
002841:                     'direction_deg': repaired_direction,
002842:                     'pos': tuple(repaired_position),
002843:                     'prev_pos': tuple(previous_position),
002844:                 }
002845: 
002846:             print(
002847:                 f"[POST_TRACK_STATE_REPAIR] f{self.frame_count}: "
002848:                 f"committed={committed_position} -> repaired={repaired_position} "
002849:                 f"motion={repaired_distance:.1f}px "
002850:                 f"velocity_history={self.ball_velocity_history}"
002851:             )
002852:         else:
002853:             self._maybe_clear_post_serve_pre_net_recovery(
002854:                 previous_position, repaired_position, source='local-ai'
002855:             )
002856:             self.ball_center = repaired_position
002857:             self.ball_size = recovered_area
002858:             if recovered.get("trajectory_rescue"):
002859:                 trajectory_motion = dict(recovered.get("trajectory_motion") or {})
002860:                 prior_motion = (pre_track_snapshot or {}).get("last_motion")
002861:                 self.prev_motion = dict(prior_motion) if isinstance(prior_motion, dict) else prior_motion
002862:                 self.last_motion = trajectory_motion
002863:                 if float(trajectory_motion.get("distance", 0.0) or 0.0) >= 3.0:
002864:                     self.last_nonzero_motion = dict(trajectory_motion)
002865:                 prior_history = list((pre_track_snapshot or {}).get("ball_velocity_history") or [])
002866:                 prior_history.append(float(trajectory_motion.get("distance", 0.0) or 0.0))
002867:                 self.ball_velocity_history = prior_history[-5:]
002868:                 self._held_direction_candidate = None
002869:         self.last_seen_frame = self.frame_count
002870:         self.stuck_frame_count = 0
002871:         self._pending_rally_end_reason = None
002872:         self._pending_rally_end_frame = -1
002873:         self._local_ai_recovery_count += 1
002874:         if reason != "post-recovery-follow":
002875:             # A successful repair commonly happens at contact/occlusion.  The
002876:             # next few frames are the outgoing flight, where the normal tracker
002877:             # has the least reliable prediction.  Establish a bounded handoff
002878:             # window; each *verified* local-AI frame can extend the immediate
002879:             # follow-up by four frames, but never beyond this deadline.
002880:             self._local_ai_handoff_deadline_frame = self.frame_count + 14
002881: 
002882:         handoff_deadline = int(getattr(self, "_local_ai_handoff_deadline_frame", -1))
002883:         if handoff_deadline >= self.frame_count:
002884:             self._local_ai_follow_until_frame = min(
002885:                 handoff_deadline,
002886:                 max(
002887:                     int(getattr(self, "_local_ai_follow_until_frame", -1)),
002888:                     self.frame_count + 4,
002889:                 ),
002890:             )
002891:         print(
002892:             f"[LOCAL_AI_RECOVERY] f{self.frame_count}: accepted {repaired_position} "
002893:             f"score={float(recovered.get('ai_score', 0.0)):.6f} "
002894:             f"replayed={self.local_ai_recovery.lookback_frames}f reason={reason}"
002895:         )
002896:         return repaired_position
002897: 
002898:     def _player_serve_context(self, serve_position):
002899:         tracker = getattr(self, "player_tracker", None)
002900:         if tracker is None:
002901:             return {}
002902:         try:
002903:             server_idx = self._current_server_index()
002904:             context = tracker.serve_context(
002905:                 serve_position,
002906:                 player_name=self.player_names[server_idx] if server_idx is not None else "",
002907:             )
002908:             tracker.adopt_serve_context(context, frame_index=int(getattr(self, 'frame_count', -1)))
002909:             return context
002910:         except Exception:
002911:             return {}
002912: 
002913:     def _refresh_player_serve_context(self):
002914:         """Refresh server/receiver geometry while the serve is still starting."""
002915:         tracker = getattr(self, "player_tracker", None)
002916:         context = getattr(self, "_point_history_current", None)
002917:         if tracker is None or context is None:
002918:             return
002919:         serve_context = context.get('serve_context') or {}
002920:         serve_x = serve_context.get('serve_x')
002921:         serve_y = serve_context.get('serve_y')
002922:         if serve_x is None or serve_y is None:
002923:             return
002924:         try:
002925:             previous_receiver = serve_context.get('receiver_position')
002926:             refreshed = self._player_serve_context((serve_x, serve_y))
002927:             if not refreshed:
002928:                 return
002929:             # Do not replace a valid serve-ball coordinate; update the player
002930:             # detections as they become available during the opening frames.
002931:             for key, value in refreshed.items():
002932:                 if value is not None and value != '':
002933:                     serve_context[key] = value
002934:             context['serve_context'] = serve_context
002935:             current_receiver = serve_context.get('receiver_position')
002936:             if current_receiver and current_receiver != previous_receiver:
002937:                 print(
002938:                     f"[SERVE_PLAYERS] f{self.frame_count}: "
002939:                     f"server={serve_context.get('player') or 'unknown'} "
002940:                     f"side={serve_context.get('player_side')} "
002941:                     f"position={serve_context.get('player_position')} "
002942:                     f"receiver_side={serve_context.get('receiver_side')} "
002943:                     f"receiver_position={current_receiver} "
002944:                     f"area={serve_context.get('receiver_area')}"
002945:                 )
002946:         except Exception:
002947:             return
002948: 
002949:     def _draw_player_tracking(self, frame, scale=1.0):
002950:         tracker = getattr(self, "player_tracker", None)
002951:         if tracker is None:
002952:             return frame
002953:         if not getattr(self, "tracking", False) and getattr(self, "_point_history_current", None) is None:
002954:             return frame
002955:         result = frame
002956:         try:
002957:             for side, track in tracker.tracks.items():
002958:                 if track.bbox is None:
002959:                     continue
002960:                 x, y, w, h = track.bbox
002961:                 x1 = int(round(x * scale))
002962:                 y1 = int(round(y * scale))
002963:                 x2 = int(round((x + w) * scale))
002964:                 y2 = int(round((y + h) * scale))
002965:                 color = (255, 180, 40) if side == "near" else (40, 190, 255)
002966:                 thickness = 2 if track.visible else 1
002967:                 cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
002968:                 if track.head is not None:
002969:                     hx, hy = int(track.head[0] * scale), int(track.head[1] * scale)
002970:                     cv2.circle(result, (hx, hy), max(4, int(8 * scale)), color, 1)
002971:                 if track.shoes is not None:
002972:                     sx, sy = int(track.shoes[0] * scale), int(track.shoes[1] * scale)
002973:                     cv2.circle(result, (sx, sy), max(3, int(6 * scale)), color, 1)
002974:                 if track.racket and track.racket.get("line"):
002975:                     lx1, ly1, lx2, ly2 = track.racket["line"]
002976:                     cv2.line(
002977:                         result,
002978:                         (int(lx1 * scale), int(ly1 * scale)),
002979:                         (int(lx2 * scale), int(ly2 * scale)),
002980:                         (0, 165, 255),
002981:                         max(1, int(round(2 * scale))),
002982:                     )
002983:                 label = f"{side} {track.confidence:.2f}"
002984:                 cv2.putText(
002985:                     result,
002986:                     label,
002987:                     (x1, max(15, y1 - 5)),
002988:                     cv2.FONT_HERSHEY_SIMPLEX,
002989:                     max(0.35, 0.5 * scale),
002990:                     color,
002991:                     max(1, int(round(1.5 * scale))),
002992:                     cv2.LINE_AA,
002993:                 )
002994:         except Exception:
002995:             return result
002996:         return result
002997: 
002998:     def log_motion_metrics(self, prev_pos, dx, dy, distance, direction_deg):
002999:         """Log per-frame motion and raise a focus-loss flag when movement spikes."""
003000:         direction_text = f"{direction_deg:+.1f} deg" if direction_deg is not None else "N/A"
003001:         print(f"Frame {self.frame_count}: Movement {distance:.1f}px (dx={dx}, dy={dy}) Direction: {direction_text}")
003002: 
003003:         self.prev_motion = self.last_motion
003004:         self.last_motion = {
003005:             'distance': distance,
003006:             'dx': dx,
003007:             'dy': dy,
003008:             'direction_deg': direction_deg
003009:         }
003010:         if distance >= 3.0:
003011:             self.last_nonzero_motion = dict(self.last_motion)
003012:         angle_diff = None
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
002769:         # V33: Local-AI recovery runs after the normal tracker has already
002770:         # committed its candidate.  If AI replaces that candidate, repair the
002771:         # *entire* live tracking state atomically.  Keeping only the repaired
002772:         # coordinate leaves last_motion / velocity history from the rejected
002773:         # contour and poisons the next-frame prediction.
002774:         committed_position = (
002775:             tuple(tracked_position) if tracked_position is not None else None
002776:         )
002777:         recovery_replaced_committed_candidate = (
002778:             pre_track_snapshot is not None and
002779:             previous_position is not None and
002780:             committed_position is not None and
002781:             committed_position != tuple(repaired_position)
002782:         )
002783:         recovered_area = float(recovered.get("area", self.ball_size or 0.0))
002784: 
002785:         if recovery_replaced_committed_candidate:
002786:             self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
002787:             self._maybe_clear_post_serve_pre_net_recovery(
002788:                 previous_position, repaired_position, source='local-ai'
002789:             )
002790: 
002791:             repaired_dx = float(repaired_position[0]) - float(previous_position[0])
002792:             repaired_dy = float(repaired_position[1]) - float(previous_position[1])
002793:             repaired_distance = math.hypot(repaired_dx, repaired_dy)
002794:             repaired_direction = (
002795:                 math.degrees(math.atan2(repaired_dy, repaired_dx))
002796:                 if repaired_distance > 0.0 else None
002797:             )
002798:             repaired_motion = {
002799:                 'distance': repaired_distance,
002800:                 'dx': repaired_dx,
002801:                 'dy': repaired_dy,
002802:                 'direction_deg': repaired_direction,
002803:             }
002804: 
002805:             self.ball_center = repaired_position
002806:             self.ball_size = recovered_area
002807:             prior_motion = pre_track_snapshot.get("last_motion")
002808:             self.prev_motion = (
002809:                 dict(prior_motion) if isinstance(prior_motion, dict) else prior_motion
002810:             )
002811:             self.last_motion = repaired_motion
002812:             if repaired_distance >= 3.0:
002813:                 self.last_nonzero_motion = dict(repaired_motion)
002814:             prior_history = list(pre_track_snapshot.get("ball_velocity_history") or [])
002815:             prior_history.append(repaired_distance)
002816:             self.ball_velocity_history = prior_history[-5:]
002817:             self._held_direction_candidate = None
002818: 
002819:             # The rollback restores the prior gray frame as well.  For an
002820:             # accepted current-frame repair, the next frame must compare against
002821:             # this decoded frame, not against frame N-1 twice.
002822:             if frame is not None:
002823:                 try:
002824:                     self._prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
002825:                 except cv2.error:
002826:                     pass
002827: 
002828:             if (
002829:                 getattr(self, '_last_motion_reacq_frame', -1000000) == self.frame_count and
002830:                 getattr(self, '_last_motion_reacq_pos', None) == committed_position
002831:             ):
002832:                 self._last_motion_reacq_frame = -1000000
002833:                 self._last_motion_reacq_pos = None
002834: 
002835:             # Replace a same-frame motion-history sample from the rejected
002836:             # contour rather than leaving two contradictory versions of frame N.
002837:             if self.motion_history and int(self.motion_history[-1].get('frame', -1)) == int(self.frame_count):
002838:                 self.motion_history[-1] = {
002839:                     'frame': self.frame_count,
002840:                     'distance': repaired_distance,
002841:                     'direction_deg': repaired_direction,
002842:                     'pos': tuple(repaired_position),
002843:                     'prev_pos': tuple(previous_position),
002844:                 }
002845: 
002846:             print(
002847:                 f"[POST_TRACK_STATE_REPAIR] f{self.frame_count}: "
002848:                 f"committed={committed_position} -> repaired={repaired_position} "
002849:                 f"motion={repaired_distance:.1f}px "
002850:                 f"velocity_history={self.ball_velocity_history}"
002851:             )
002852:         else:
002853:             self._maybe_clear_post_serve_pre_net_recovery(
002854:                 previous_position, repaired_position, source='local-ai'
002855:             )
002856:             self.ball_center = repaired_position
002857:             self.ball_size = recovered_area
002858:             if recovered.get("trajectory_rescue"):
002859:                 trajectory_motion = dict(recovered.get("trajectory_motion") or {})
002860:                 prior_motion = (pre_track_snapshot or {}).get("last_motion")
002861:                 self.prev_motion = dict(prior_motion) if isinstance(prior_motion, dict) else prior_motion
002862:                 self.last_motion = trajectory_motion
002863:                 if float(trajectory_motion.get("distance", 0.0) or 0.0) >= 3.0:
002864:                     self.last_nonzero_motion = dict(trajectory_motion)
002865:                 prior_history = list((pre_track_snapshot or {}).get("ball_velocity_history") or [])
002866:                 prior_history.append(float(trajectory_motion.get("distance", 0.0) or 0.0))
002867:                 self.ball_velocity_history = prior_history[-5:]
002868:                 self._held_direction_candidate = None
002869:         self.last_seen_frame = self.frame_count
002870:         self.stuck_frame_count = 0
002871:         self._pending_rally_end_reason = None
002872:         self._pending_rally_end_frame = -1
002873:         self._local_ai_recovery_count += 1
002874:         if reason != "post-recovery-follow":
002875:             # A successful repair commonly happens at contact/occlusion.  The
002876:             # next few frames are the outgoing flight, where the normal tracker
002877:             # has the least reliable prediction.  Establish a bounded handoff
002878:             # window; each *verified* local-AI frame can extend the immediate
002879:             # follow-up by four frames, but never beyond this deadline.
002880:             self._local_ai_handoff_deadline_frame = self.frame_count + 14
002881: 
002882:         handoff_deadline = int(getattr(self, "_local_ai_handoff_deadline_frame", -1))
002883:         if handoff_deadline >= self.frame_count:
002884:             self._local_ai_follow_until_frame = min(
002885:                 handoff_deadline,
002886:                 max(
002887:                     int(getattr(self, "_local_ai_follow_until_frame", -1)),
002888:                     self.frame_count + 4,
002889:                 ),
002890:             )
002891:         print(
002892:             f"[LOCAL_AI_RECOVERY] f{self.frame_count}: accepted {repaired_position} "
002893:             f"score={float(recovered.get('ai_score', 0.0)):.6f} "
002894:             f"replayed={self.local_ai_recovery.lookback_frames}f reason={reason}"
002895:         )
002896:         return repaired_position
002897: 
002898:     def _player_serve_context(self, serve_position):
002899:         tracker = getattr(self, "player_tracker", None)
002900:         if tracker is None:
002901:             return {}
002902:         try:
002903:             server_idx = self._current_server_index()
002904:             context = tracker.serve_context(
002905:                 serve_position,
002906:                 player_name=self.player_names[server_idx] if server_idx is not None else "",
002907:             )
002908:             tracker.adopt_serve_context(context, frame_index=int(getattr(self, 'frame_count', -1)))
002909:             return context
002910:         except Exception:
002911:             return {}
002912: 
002913:     def _refresh_player_serve_context(self):
002914:         """Refresh server/receiver geometry while the serve is still starting."""
002915:         tracker = getattr(self, "player_tracker", None)
002916:         context = getattr(self, "_point_history_current", None)
002917:         if tracker is None or context is None:
002918:             return
002919:         serve_context = context.get('serve_context') or {}
002920:         serve_x = serve_context.get('serve_x')
002921:         serve_y = serve_context.get('serve_y')
002922:         if serve_x is None or serve_y is None:
002923:             return
002924:         try:
002925:             previous_receiver = serve_context.get('receiver_position')
002926:             refreshed = self._player_serve_context((serve_x, serve_y))
002927:             if not refreshed:
002928:                 return
002929:             # Do not replace a valid serve-ball coordinate; update the player
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
027024:                     lower_launch_frame = int(getattr(
027025:                         self, '_last_verified_lower_contact_launch_frame', -1000000
027026:                     ))
027027:                     lower_launch_age = int(self.frame_count) - lower_launch_frame
027028:                     verified_lower_contact_continuation = False
027029:                     continuation_step = 0.0
027030:                     continuation_cos = -1.0
027031:                     if (
027032:                         tracked_position is not None and
027033:                         prev_ball_center is not None and
027034:                         pre_track_snapshot is not None and
027035:                         1 <= lower_launch_age <= 2
027036:                     ):
027037:                         continuation_dx = (
027038:                             float(tracked_position[0]) - float(prev_ball_center[0])
027039:                         )
027040:                         continuation_dy = (
027041:                             float(tracked_position[1]) - float(prev_ball_center[1])
027042:                         )
027043:                         continuation_step = math.hypot(
027044:                             continuation_dx, continuation_dy
027045:                         )
027046:                         prior_motion = pre_track_snapshot.get('last_motion') or {}
027047:                         prior_dx = float(prior_motion.get('dx', 0.0) or 0.0)
027048:                         prior_dy = float(prior_motion.get('dy', 0.0) or 0.0)
027049:                         prior_step = float(
027050:                             prior_motion.get('distance', 0.0) or
027051:                             math.hypot(prior_dx, prior_dy)
027052:                         )
027053:                         if continuation_step > 0.0 and prior_step > 0.0:
027054:                             continuation_cos = (
027055:                                 continuation_dx * prior_dx +
027056:                                 continuation_dy * prior_dy
027057:                             ) / (continuation_step * prior_step)
027058:                         continuation_limit = max(
027059:                             180.0,
027060:                             min(320.0, prior_step * 2.2 + 60.0),
027061:                         )
027062:                         verified_lower_contact_continuation = (
027063:                             12.0 <= continuation_step <= continuation_limit and
027064:                             continuation_dy <= -12.0 and
027065:                             prior_dy <= -8.0 and
027066:                             continuation_cos >= 0.25
027067:                         )
027068: 
027069:                     verified_lower_contact_post_guard = (
027070:                         tracked_position is not None and
027071:                         (
027072:                             lower_launch_age == 0 or
027073:                             verified_lower_contact_launch or
027074:                             verified_lower_contact_continuation
027075:                         )
027076:                     )
027077:                     if verified_lower_contact_post_guard:
027078:                         guard_kind = (
027079:                             'launch'
027080:                             if lower_launch_age == 0 or verified_lower_contact_launch
027081:                             else 'continuation'
027082:                         )
027083:                         print(
027084:                             f"[VERIFIED-LOWER-CONTACT POST-RECOVERY SKIP] "
027085:                             f"f{self.frame_count}: kind={guard_kind} age={lower_launch_age} "
027086:                             f"keeping={tuple(tracked_position)} step={continuation_step:.1f}px "
027087:                             f"cos={continuation_cos:.3f}"
027088:                         )
027089:                     if (
027090:                         not forced_local_ai and
027091:                         not contact_local_ai and
027092:                         not verified_lower_contact_post_guard
027093:                     ):
027094:                         tracked_position = self._try_local_ai_recovery(
027095:                             prev_ball_center, tracked_position, prev_stuck,
027096:                             pre_track_snapshot=pre_track_snapshot,
027097:                             frame=frame,
027098:                         )
027099:                     self._local_ai_tight_roi_previous_gray = None
027100:                     # Reject any position that jumps impossibly far in one frame (false positive).
027101:                     # When the tracker is in re-acquisition mode (stuck >= 5 before the call), allow
027102:                     # a larger jump because the ball may have traveled far while lost.
027103:                     if tracked_position and prev_ball_center:
027104:                         jump = math.hypot(tracked_position[0] - prev_ball_center[0],
027105:                                           tracked_position[1] - prev_ball_center[1])
027106:                         motion_reacq_this_frame = (
027107:                             getattr(self, '_last_motion_reacq_frame', -1000000) == self.frame_count and
027108:                             getattr(self, '_last_motion_reacq_pos', None) == tracked_position
027109:                         )
027110:                         trusted_rally_launch = (
027111:                             getattr(self, '_trusted_rally_launch_frame', -1000000) == self.frame_count and
027112:                             getattr(self, '_trusted_rally_launch_pos', None) == tracked_position
027113:                         )
027114:                         # Allow a larger jump when re-acquiring after being stuck for 5+ frames:
027115:                         # a racket hit can send the ball 800+ px in one frame, so we use 1500px
027116:                         # to let motion-based re-acquisition recover across the full court.
027117:                         max_jump = 1500 if (
027118:                             prev_stuck >= 5 or prev_top_return_wait or prev_back_return_wait or
027119:                             motion_reacq_this_frame or trusted_rally_launch
027120:                         ) else 400
027121:                         if jump > max_jump:
027122:                             print(f"[JUMP_REJECTED] f{self.frame_count}: jumped {jump:.0f}px from {prev_ball_center} to {tracked_position} (limit={max_jump}px, prev_stuck={prev_stuck}), keeping previous")
027123:                             self.ball_center = prev_ball_center
027124:                             self.stuck_frame_count = max(self.stuck_frame_count, prev_stuck + 1)
027125:                             tracked_position = prev_ball_center
027126: 
027127:                     # Validate only the position that survived the physical
027128:                     # jump gate above.  Previously the provisional serve
027129:                     # state consumed a 425px false contour at a prior frame before
027130:                     # that contour was rejected below.  Its bogus vector then
027131:                     # hid the real upward reversal at a prior frame.
027132:                     # Once a far toss is independently marked as caught, it
027133:                     # becomes a *history* question only.  Do not continue to
027134:                     # mutate the live provisional flags: the previous attempt
027135:                     # cleared those flags at a prior frame and changed the otherwise
027136:                     # stable HSV path of the following real rally.
027137:                     tainted_far_toss_active = (
027138:                         isinstance(
027139:                             getattr(self, '_tainted_provisional_serve_start', None),
027140:                             dict,
027141:                         ) and
027142:                         getattr(self, '_provisional_serve_start_kind', None) ==
027143:                         'far-top-post-hit'
027144:                     )
027145:                     provisional_kind_before_validation = getattr(
027146:                         self, '_provisional_serve_start_kind', None
027147:                     )
027148:                     provisional_reject_reason = (
027149:                         None if tainted_far_toss_active else
027150:                         self._validate_provisional_serve_start(
027151:                             prev_ball_center, tracked_position
027152:                         )
027153:                     )
027154:                     # A pending far-toss token is solely a fallback for a
027155:                     # *rejected* practice toss.  Once this tentative flight
027156:                     # has earned three coherent netward steps, it is a real
027157:                     # serve (for example a prior frame -> a prior frame) and the token must
027158:                     # not survive its fault/end state.  Leaving it alive
027159:                     # caused the local model to re-use that old start at
027160:                     # a prior frame and inject a second, false point.
027161:                     if provisional_reject_reason is not None:
027162:                         provisional_kind = getattr(self, '_provisional_serve_start_kind', None)
027163:                         rejected_start = self._current_history_serve_start_frame()
027164:                         if provisional_kind == 'far-top-post-hit':
027165:                             # This is a caught far-side toss.  Do *not* reset
027166:                             # to WAITING_FOR_SERVE: that was the rejected
027167:                             # approach and it changed later, already-verified
027168:                             # points.  Keep the normal HSV flow alive and ask
027169:                             # local AI to prove a later outgoing path in
027170:                             # shadow mode.  Only proven evidence may alter the
027171:                             # final history/serve-fault interpretation.
027172:                             self._tainted_provisional_serve_start = {
027173:                                 'start_frame': rejected_start,
027174:                                 'detected_frame': int(self.frame_count),
027175:                                 'anchor': tuple(prev_ball_center),
027176:                                 'reason': provisional_reject_reason,
027177:                             }
027178:                             print(
027179:                                 f"[SERVE_START_TAINTED] f{self.frame_count}: "
027180:                                 f"keeping tracker/provisional state for start f{rejected_start}; "
027181:                                 f"{provisional_reject_reason}"
027182:                             )
027183:                         else:
027184:                             print(
027185:                                 f"[SERVE_START_REJECTED] f{self.frame_count}: "
027186:                                 f"discarding provisional start f{rejected_start}; "
027187:                                 f"{provisional_reject_reason}"
027188:                             )
027189:                             self._point_history_current = None
027190:                             self._provisional_serve_start_kind = None
027191:                             self._serve_start_requires_confirmation = False
027192:                             game_state = "WAITING_FOR_SERVE"
027193:                             clear_waiting_serve_history()
027194:                             reset_tracking_state()
027195:                             continue
027196:                     # Shadow verification is intentionally side-effect free
027197:                     # for ball position/game state.  It only supplies a
027198:                     # later verified launch frame for history/fault scoring.
027199:                     reconcile_tainted_far_serve_start()
027200:                     if getattr(self, "_discard_provisional_serve_from_ai", False):
027201:                         self._discard_provisional_serve_from_ai = False
027202:                         self._ignore_unconfirmed_serve_start_result(
027203:                             "local AI rejected repeated all-player-body recovery path"
027204:                         )
027205:                         print(
027206:                             f"[SERVE_START_IGNORED] f{self.frame_count}: "
027207:                             "discarding false provisional serve after repeated local-AI body paths"
027208:                         )
027209:                         game_state = "WAITING_FOR_SERVE"
027210:                         clear_waiting_serve_history()
027211:                         reset_tracking_state()
027212:                         continue
027213:                 if tracked_position:
027214:                     vel = self.last_motion['distance'] if self.last_motion else 0
027215:                     size_text = f"{self.ball_size:.1f}px" if self.ball_size is not None else "unknown"
027216:                     vel_hist_tail = [round(v, 1) for v in getattr(self, 'ball_velocity_history', [])[-5:]]
027217:                     print(f"Frame {self.frame_count}: Ball tracked at {tracked_position} - Size: {size_text}")
027218:                     print(f"[TRACK] f{self.frame_count}: pos={tracked_position} vel={vel:.1f}px stuck={self.stuck_frame_count} vel_hist={vel_hist_tail}")
027219:                     # Infer a visually missed return before any queued point-end is scored.
027220:                     self._maybe_infer_return_contact_from_side_recross(tracked_position, frame)
027221:                     if self._point_history_current is not None:
027222:                         self._point_history_current.setdefault('tracking_trace', []).append({
027223:                             'frame': int(self.frame_count),
027224:                             'pos': [int(tracked_position[0]), int(tracked_position[1])],
027225:                             'size': float(self.ball_size) if self.ball_size is not None else None,
027226:                             'stuck': int(self.stuck_frame_count),
027227:                         })
027228:                     if self.ball_dataset_exporter is not None:
027229:                         self.ball_dataset_exporter.write_frame(
027230:                             frame,
027231:                             source_frame=self.frame_count,
027232:                             ball_center=tracked_position,
027233:                             ball_area=self.ball_size,
027234:                             stuck_frames=self.stuck_frame_count,
027235:                             motion_distance=(self.last_motion or {}).get('distance'),
027236:                             tracking_active=self.tracking,
027237:                             point_index=(
027238:                                 self._point_history_current.get('point_index')
027239:                                 if self._point_history_current is not None else None
027240:                             ),
027241:                         )
027242: 
027243:                     pending_reason = getattr(self, '_pending_rally_end_reason', None)
027244:                     if pending_reason and reference_target_hold:
027245:                         print(
027246:                             f"Frame {self.frame_count}: [REFERENCE_POINT_HOLD] "
027247:                             f"discarding premature queued end '{pending_reason}' "
027248:                             f"until f{reference_target_frame}"
027249:                         )
027250:                         self._pending_rally_end_reason = None
027251:                         self._pending_rally_end_frame = -1
027252:                         pending_reason = None
027253:                     if pending_reason:
027254:                         # Before scoring a service-box fault, prefer a separate
027255:                         # moving ball that continues the incoming trajectory.
027256:                         # This catches the f1785 stale-marker regression without
027257:                         # weakening genuine bounce reversals.
027258:                         if self._resume_false_serve_bounce_from_motion(
027259:                             pending_reason, tracked_position
027260:                         ):
027261:                             self._pending_rally_end_reason = None
027262:                             self._pending_rally_end_frame = -1
027263:                             continue
027264:                         # Point-end detection can queue an out reason one or
027265:                         # more frames before the stuck timeout is reached.  If
027266:                         # this is still an unconfirmed serve toss with no
027267:                         # racket/net contact and the marker has stopped, drop
027268:                         # the queued false end instead of writing a bogus row.
027269:                         suppressed_frame = int(
027270:                             getattr(self, '_last_out_bounce_suppressed_frame', -1000000)
027271:                         )
027272:                         suppressed_point = getattr(self, '_last_out_bounce_suppressed_point', None)
027273:                         recent_static_serve_abort = (
```

