# V26 exact evidence targets

## '_pending_night_static_out = {' at line 24716
```python
024681:         outside_depth = 0.0
024682:         if side == 'left' and outside_left_x is not None:
024683:             outside_depth = max(0.0, float(outside_left_x) - float(outside_x))
024684:         elif side == 'right' and outside_right_x is not None:
024685:             outside_depth = max(0.0, float(outside_x) - float(outside_right_x))
024686:         current_ball_size = float(getattr(self, 'ball_size', 0.0) or 0.0)
024687:         if self._is_night_session_config():
024688:             last_candidate_motion_frame = int(
024689:                 getattr(self, '_last_tracked_candidate_motion_frame', -1000000)
024690:             )
024691:             last_candidate_motion_mean = float(
024692:                 getattr(self, '_last_tracked_candidate_motion_mean', 0.0) or 0.0
024693:             )
024694:             last_candidate_motion_max = float(
024695:                 getattr(self, '_last_tracked_candidate_motion_max', 0.0) or 0.0
024696:             )
024697:             weak_current_candidate = (
024698:                 last_candidate_motion_frame == self.frame_count and
024699:                 last_candidate_motion_mean < 3.5 and
024700:                 last_candidate_motion_max < 18.0
024701:             )
024702:             deep_static_out = (
024703:                 weak_current_candidate and
024704:                 current_ball_size <= 90.0 and
024705:                 (
024706:                     outside_depth >= max(90.0, sideline_margin * 5.0) or
024707:                     x <= max(70, int(frame.shape[1] * 0.018)) or
024708:                     x >= frame.shape[1] - max(70, int(frame.shape[1] * 0.018))
024709:                 )
024710:             )
024711:             if deep_static_out:
024712:                 # V14: only arm a pending OUT when the static-looking point is
024713:                 # still supported by a coherent descending flight. Pure side
024714:                 # hotspots remain ordinary suppressions and never get promoted.
024715:                 if recent_descending and curr_dy >= 6.0 and curr_speed >= 6.0:
024716:                     self._pending_night_static_out = {
024717:                         'frame': int(self.frame_count),
024718:                         'position': tuple(ball_position),
024719:                         'side': side,
024720:                         'dx': curr_dx,
024721:                         'dy': curr_dy,
024722:                         'speed': curr_speed,
024723:                     }
024724:                     print(
024725:                         f"Frame {self.frame_count}: [OUT-BOUNCE PENDING] "
024726:                         f"point={tuple(ball_position)} side={side} "
024727:                         f"motion=({curr_dx:.1f},{curr_dy:.1f}) speed={curr_speed:.1f}px"
024728:                     )
024729:                 self._last_out_bounce_suppressed_frame = self.frame_count
024730:                 self._last_out_bounce_suppressed_point = tuple(ball_position)
024731:                 print(
024732:                     f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
024733:                     f"night static side artifact pos={ball_position} "
024734:                     f"size={current_ball_size:.1f}px outside_depth={outside_depth:.1f} "
024735:                     f"motion={last_candidate_motion_mean:.1f}/{last_candidate_motion_max:.1f}"
024736:                 )
024737:                 return False, None
024738: 
024739:         # A ball that was travelling down and *outward* can only be a
024740:         # sideline-bounce candidate if its rebound remains physically
024741:         # compatible with that path.  At a prior frame the near player returns the
024742:         # ball: it is large/visible, reverses upward and strongly back toward
024743:         # the court.  Treating that as an out bounce because a prior frame happened to
024744:         # be outside the left line ends the rally before the return.  Keep this
024745:         # deliberately narrow so ordinary small/airborne sideline outs still
024746:         # use the existing reversal detector.
024747:         inward_racket_return = (
024748:             prev_pos_outside and
024749:             curr_pos_outside and
024750:             current_ball_size >= 260.0 and
024751:             prev_speed >= 45.0 and
```

## 'HSV: H=' at line 15583
```python
015548:                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
015549: 
015550:         lower = model.get('white_hsv_lower', (0, 0, 170))
015551:         upper = model.get('white_hsv_upper', (179, 70, 255))
015552:         cv2.putText(
015553:             result,
015554:             f"WHITE HSV LINES H={lower[0]}-{upper[0]} S={lower[1]}-{upper[1]} V={lower[2]}-{upper[2]}",
015555:             (10, max(210, int(210 * scale))),
015556:             cv2.FONT_HERSHEY_SIMPLEX,
015557:             0.55,
015558:             (255, 255, 0),
015559:             2,
015560:         )
015561: 
015562:     def print_click_hsv_neighborhood(self, frame, point, display_point=None, scale=None):
015563:         """Print the clicked frame position plus a 3x3 HSV neighborhood."""
015564:         print("\n=== MOUSE CLICK HSV DEBUG ===")
015565:         if frame is None:
015566:             print("No current frame available for HSV sampling.")
015567:             return
015568: 
015569:         x, y = int(point[0]), int(point[1])
015570:         frame_height, frame_width = frame.shape[:2]
015571:         if display_point is not None:
015572:             scale_text = f", display_scale={scale:.4f}" if scale else ""
015573:             print(f"Display click: ({display_point[0]}, {display_point[1]}) -> frame: ({x}, {y}){scale_text}")
015574:         else:
015575:             print(f"Frame click: ({x}, {y})")
015576: 
015577:         if x < 0 or y < 0 or x >= frame_width or y >= frame_height:
015578:             print(f"Click is outside frame bounds: width={frame_width}, height={frame_height}")
015579:             return
015580: 
015581:         hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
015582:         center_hsv = hsv_frame[y, x]
015583:         print(f"Center HSV: H={int(center_hsv[0])}, S={int(center_hsv[1])}, V={int(center_hsv[2])}")
015584:         print("3x3 HSV neighborhood (center marked with *):")
015585: 
015586:         for dy in (-1, 0, 1):
015587:             row_values = []
015588:             for dx in (-1, 0, 1):
015589:                 px = x + dx
015590:                 py = y + dy
015591:                 marker = "*" if dx == 0 and dy == 0 else " "
015592:                 if 0 <= px < frame_width and 0 <= py < frame_height:
015593:                     hsv = hsv_frame[py, px]
015594:                     row_values.append(
015595:                         f"{marker}({px},{py})=H{int(hsv[0])} S{int(hsv[1])} V{int(hsv[2])}"
015596:                     )
015597:                 else:
015598:                     row_values.append(f"{marker}({px},{py})=OUT")
015599:             print("  " + " | ".join(row_values))
015600:     
015601:     def open_hsv_tuner(self, frame, point):
015602:         """Open HSV filter tuner with 100x100 region around the ball."""
015603:         x, y = point
015604:         
015605:         # Extract larger region around the ball for debugging - MUST BE FRESH COPY
015606:         region_size = 150  # 150 pixels radius = 300x300 total
015607:         x1 = max(0, x - region_size)
015608:         y1 = max(0, y - region_size)
015609:         x2 = min(frame.shape[1], x + region_size)
015610:         y2 = min(frame.shape[0], y + region_size)
015611:         
015612:         print(f"\n=== EXTRACTING NEW REGION ===")
015613:         print(f"Click position: ({x}, {y})")
015614:         print(f"Extracting region: ({x1},{y1}) to ({x2},{y2}) from frame {frame.shape}")
015615:         
015616:         # IMPORTANT: Extract fresh region from the current frame
015617:         region = frame[y1:y2, x1:x2].copy()  # .copy() ensures fresh data
015618:         if region.size == 0:
```

## 'HSV: H=' at line 15649
```python
015614:         print(f"Extracting region: ({x1},{y1}) to ({x2},{y2}) from frame {frame.shape}")
015615:         
015616:         # IMPORTANT: Extract fresh region from the current frame
015617:         region = frame[y1:y2, x1:x2].copy()  # .copy() ensures fresh data
015618:         if region.size == 0:
015619:             print("Error: Cannot extract 100x100 region around ball")
015620:             return False
015621:         
015622:         print(f"Extracted region shape: {region.shape}")
015623:         
015624:         # Resize to exactly 300x300 if needed
015625:         if region.shape[0] != 300 or region.shape[1] != 300:
015626:             region = cv2.resize(region, (300, 300))
015627:             print(f"Resized region to: {region.shape}")
015628:         
015629:         # Get initial HSV values at the clicked point (relative to region)
015630:         rel_x = x - x1
015631:         rel_y = y - y1
015632:         
015633:         # Adjust relative coordinates if resizing happened
015634:         if x2 - x1 != 300:
015635:             rel_x = int((rel_x / (x2 - x1)) * 300)
015636:         if y2 - y1 != 300:
015637:             rel_y = int((rel_y / (y2 - y1)) * 300)
015638:         
015639:         # Ensure relative coordinates are within bounds
015640:         rel_x = max(0, min(299, rel_x))
015641:         rel_y = max(0, min(299, rel_y))
015642:         
015643:         print(f"Relative coordinates: ({rel_x}, {rel_y}) in 100x100 region")
015644:         
015645:         hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
015646:         initial_hsv = hsv_region[rel_y, rel_x]
015647:         
015648:         print(f"Ball position: ({x}, {y})")
015649:         print(f"Initial HSV: H={initial_hsv[0]}, S={initial_hsv[1]}, V={initial_hsv[2]}")
015650:         print("Adjust sliders until ball appears WHITE, then press 'S' to save")
015651:         
015652:         # Create HSV tuner window
015653:         tuner_window = "HSV Filter Tuner - 300x300"
015654:         cv2.namedWindow(tuner_window, cv2.WINDOW_NORMAL)
015655:         cv2.resizeWindow(tuner_window, 800, 600)
015656:         cv2.waitKey(1)  # Allow window to be created
015657:         
015658:         # Use current active HSV values as starting point
015659:         h_min = int(self.hsv_lower[0]) if self.hsv_lower is not None else 20
015660:         h_max = int(self.hsv_upper[0]) if self.hsv_upper is not None else 90
015661:         s_min = int(self.hsv_lower[1]) if self.hsv_lower is not None else 20
015662:         s_max = int(self.hsv_upper[1]) if self.hsv_upper is not None else 255
015663:         v_min = int(self.hsv_lower[2]) if self.hsv_lower is not None else 70
015664:         v_max = int(self.hsv_upper[2]) if self.hsv_upper is not None else 255
015665:         
015666:         # Bulb size filter (wider to catch first frames)
015667:         bulb_min = 1
015668:         bulb_max = 30
015669:         
015670:         # Create trackbars for HSV and bulb size
015671:         cv2.createTrackbar('H Min', tuner_window, h_min, 179, lambda x: None)
015672:         cv2.createTrackbar('H Max', tuner_window, h_max, 179, lambda x: None)
015673:         cv2.createTrackbar('S Min', tuner_window, s_min, 255, lambda x: None)
015674:         cv2.createTrackbar('S Max', tuner_window, s_max, 255, lambda x: None)
015675:         cv2.createTrackbar('V Min', tuner_window, v_min, 255, lambda x: None)
015676:         cv2.createTrackbar('V Max', tuner_window, v_max, 255, lambda x: None)
015677:         cv2.createTrackbar('Bulb Min', tuner_window, bulb_min, 100, lambda x: None)
015678:         cv2.createTrackbar('Bulb Max', tuner_window, bulb_max, 100, lambda x: None)
015679:         cv2.waitKey(1)  # Allow trackbars to be created
015680:         
015681:         def update_display():
015682:             # Get current trackbar values
015683:             h_min = cv2.getTrackbarPos('H Min', tuner_window)
015684:             h_max = cv2.getTrackbarPos('H Max', tuner_window)
```

## 'HSV: H=' at line 21978
```python
021943:             if len(self.ball_velocity_history) >= 5:
021944:                 avg_velocity = sum(self.ball_velocity_history) / len(self.ball_velocity_history)
021945:                 if avg_velocity < 2.0:
021946:                     if not hasattr(self, 'ball_stopped') or not self.ball_stopped:
021947:                         print(f"  DEBUG: [BALL STOPPED] Avg velocity: {avg_velocity:.1f}px/frame")
021948:                         print(f"  DEBUG: Will search from initial position {self.initial_ball_position} for next ball")
021949:                         self.ball_stopped = True
021950:                     self.stuck_frame_count += 1
021951:                 else:
021952:                     self.ball_stopped = False
021953:                     self.stuck_frame_count = 0
021954: 
021955:             # Also count stuck frames when position hasn't moved
021956:             if self.last_motion and self.last_motion['distance'] < 1.5:
021957:                 self.stuck_frame_count += 1
021958:             elif self.last_motion and self.last_motion['distance'] >= 3.0:
021959:                 self.stuck_frame_count = 0
021960:             
021961:             # Add to HSV table using the final (possibly retracked) values
021962:             final_pos = self.ball_center
021963:             final_hsv = self.ball_hsv
021964:             final_size = self.ball_size
021965:             final_velocity = self.last_motion['distance'] if self.last_motion else velocity
021966:             self.hsv_table.append({
021967:                 'frame': self.frame_count,
021968:                 'position': final_pos,
021969:                 'hsv': final_hsv.tolist() if final_hsv is not None else hsv_values.tolist(),
021970:                 'bulb_size': final_size,
021971:                 'velocity': final_velocity,
021972:                 'hsv_range': [self.hsv_lower[0], self.hsv_upper[0], 
021973:                              self.hsv_lower[1], self.hsv_upper[1], 
021974:                              self.hsv_lower[2], self.hsv_upper[2]]
021975:             })
021976:             
021977:             if final_pos == (cx, cy):
021978:                 print(f"Frame {self.frame_count}: Ball at ({cx}, {cy}) - HSV: H={hsv_values[0]}, S={hsv_values[1]}, V={hsv_values[2]} - Size: {bulb_size:.1f}px - Velocity: {velocity:.1f}px/frame")
021979:             
021980:             return self.ball_center
021981:         
021982:         _verbose_debug_print(f"  DEBUG: [PROBLEM] No valid candidate found!")
021983:         _verbose_debug_print(f"  DEBUG: Total contours: {len(contours)}, Valid candidates: {len(candidates)}")
021984:         if self.ball_center is not None and not allow_inactive:
021985:             self._record_ball_loss_event(
021986:                 'all HSV candidates rejected',
021987:                 position=self.ball_center,
021988:                 recovery='reacquisition/alternate HSV search attempted',
021989:             )
021990:         if len(contours) > 0 and len(candidates) == 0:
021991:             tracking_size_cap = self._tracking_ball_size_max()
021992:             if 'ball_size_max_tracking' in locals():
021993:                 tracking_size_cap = ball_size_max_tracking
021994:             size_cap = f"{self.serve_ball_size_min}-{self.serve_ball_size_max}px (serve scan)" if allow_inactive else f"1-{tracking_size_cap}px"
021995:             _verbose_debug_print(f"  DEBUG: All {len(contours)} contours were rejected by size filter ({size_cap})")
021996:             # Show the actual sizes that were rejected
021997:             rejected_sizes = []
021998:             for source, contour in contours[:5]:  # Show first 5
021999:                 rejected_sizes.append(f"{cv2.contourArea(contour):.1f}px")
022000:             _verbose_debug_print(f"  DEBUG: Rejected sizes (first 5): {', '.join(rejected_sizes)}")
022001:             _verbose_debug_print(f"  DEBUG: REASON: Ball size changed outside {size_cap}")
022002:             _verbose_debug_print(f"  DEBUG:   - Ball may be too small (far away) or too large (very close)")
022003:             _verbose_debug_print(f"  DEBUG:   - Consider adjusting size filter if ball is visible")
022004:             if predicted_point:
022005:                 _verbose_debug_print(f"  DEBUG: Predicted point was {predicted_point}, consider widening search around it")
022006:         if (best_contour is None and not allow_inactive and self._is_night_session_config()
022007:                 and self.ball_center is not None):
022008:             night_recover = self._find_night_visible_ball_candidate(frame, frame_gray)
022009:             if night_recover is not None:
022010:                 return self._commit_night_visible_ball_recovery(night_recover, frame)
022011:         if best_contour is None and top_return_search_context and self.ball_center is not None:
022012:             self.stuck_frame_count = min(self.stuck_frame_count, 4)
022013:             print(f"Frame {self.frame_count}: [TOP-RETURN WAIT] no valid re-entry candidate, holding {self.ball_center}")
```

## 'HSV: H=' at line 24958
```python
024923:             bulb_max = cv2.getTrackbarPos("Bulb Max", tuner_window)
024924:             
024925:             # Apply HSV filter to serve area
024926:             hsv_serve = cv2.cvtColor(serve_roi, cv2.COLOR_BGR2HSV)
024927:             hsv_lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
024928:             hsv_upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
024929:             mask = cv2.inRange(hsv_serve, hsv_lower, hsv_upper)
024930:             
024931:             # Find contours in serve area
024932:             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
024933:             
024934:             # Draw detected balls on serve area
024935:             display_serve = serve_roi.copy()
024936:             ball_count = 0
024937:             
024938:             for contour in contours:
024939:                 area = cv2.contourArea(contour)
024940:                 if bulb_min < area < bulb_max:
024941:                     # Calculate center
024942:                     M = cv2.moments(contour)
024943:                     if M["m00"] != 0:
024944:                         cx = int(M["m10"] / M["m00"])
024945:                         cy = int(M["m01"] / M["m00"])
024946:                         
024947:                         # Draw circle and size label
024948:                         cv2.circle(display_serve, (cx, cy), 8, (0, 255, 0), 2)
024949:                         cv2.putText(display_serve, f"{area:.0f}", (cx + 12, cy), 
024950:                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
024951:                         ball_count += 1
024952:             
024953:             # Create side-by-side display
024954:             mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
024955:             combined = np.hstack([display_serve, mask_bgr])
024956:             
024957:             # Add text info
024958:             info_text = f"SERVE AREA | Balls: {ball_count} | HSV: H={h_min}-{h_max} S={s_min}-{s_max} V={v_min}-{v_max} | Size: {bulb_min}-{bulb_max}px"
024959:             cv2.putText(combined, info_text, (10, 20), 
024960:                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
024961:             
024962:             cv2.imshow(tuner_window, combined)
024963:             
024964:             key = cv2.waitKey(50) & 0xFF
024965:             if key == ord('q') or key == 27:  # 'q' or ESC
024966:                 break
024967:         
024968:         # Update HSV values with tuned values
024969:         self.hsv_lower = hsv_lower
024970:         self.hsv_upper = hsv_upper
024971:         
024972:         # Save updated HSV values to config
024973:         self.save_hsv_to_config()
024974:         
024975:         cv2.destroyWindow(tuner_window)
024976:         print("Serve area HSV tuner closed. Settings saved.")
024977:     
024978:     def save_hsv_to_config(self):
024979:         """Save current HSV values to config file."""
024980:         try:
024981:             # Load existing config
024982:             config = {}
024983:             if os.path.exists('hsv_config.json'):
024984:                 with open('hsv_config.json', 'r') as f:
024985:                     config = json.load(f)
024986:             
024987:             # Update HSV values
024988:             if "regular_court" in config:
024989:                 # Update regular court HSV values
024990:                 config["regular_court"]["h_min"] = int(self.hsv_lower[0])
024991:                 config["regular_court"]["h_max"] = int(self.hsv_upper[0])
024992:                 config["regular_court"]["s_min"] = int(self.hsv_lower[1])
024993:                 config["regular_court"]["s_max"] = int(self.hsv_upper[1])
```

## 'ball_size =' at line 361
```python
000326:         # A provisional serve can briefly select a static, ball-coloured court
000327:         # artifact.  Keep that candidate separate from the live track until
000328:         # the following decoded frame can prove that it really persists.
000329:         # This is intentionally runtime-only: a coordinate that is false
000330:         # during a tossed-and-caught serve may be a legitimate ball location
000331:         # later in a rally.
000332:         self._pending_provisional_static_candidate = None
000333:         # Keep history reconciliation separate from the live tracking state.
000334:         # A verified caught toss can change the tennis interpretation of two
000335:         # subsequent serves, but changing ``current_serve_attempt`` while the
000336:         # HSV tracker is still in flight was shown to alter later candidate
000337:         # selection.  This record lets the CSV reflect the verified sequence
000338:         # after the normal tracker has completed each physical flight.
000339:         self._serve_history_reconciliation = None
000340:         self._local_ai_follow_until_frame = -1
000341:         # A recovery usually happens at a player/racket occlusion.  The ball
000342:         # needs more than a single four-frame sample to leave that region, but
000343:         # this handoff must remain bounded so local-AI scoring cannot replace
000344:         # normal tracking for an entire rally.
000345:         self._local_ai_handoff_deadline_frame = -1
000346:         if local_ai_model:
000347:             self.local_ai_recovery = LocalBallAIRecovery(
000348:                 local_ai_model,
000349:                 config_file,
000350:                 python_executable=local_ai_python,
000351:                 work_dir=local_ai_recovery_dir or "tmp/local_ai_recovery",
000352:             )
000353:             print(
000354:                 f"[LOCAL_AI] Buffered recovery enabled: model={self.local_ai_recovery.model_path} "
000355:                 f"lookback={self.local_ai_recovery.lookback_frames}f"
000356:             )
000357:         
000358:         # Ball analysis state
000359:         self.ball_center = None
000360:         self.ball_hsv = None
000361:         self.ball_size = None
000362:         self.tracking = False
000363:         self.ball_stopped = False
000364:         self.stuck_frame_count = 0
000365:         # After a player-occlusion re-acquisition, keep rejecting static HSV
000366:         # blobs for a short window.  The player/racket can contain a large
000367:         # ball-coloured patch, and accepting it immediately after reacquisition
000368:         # causes a false bounce/end while the real ball is still in flight.
000369:         self._player_reacq_protect_until_frame = -1
000370:         # A failed full-frame motion reacquisition should not be followed by
000371:         # another expensive full-frame HSV contour scan on every frame.  The
000372:         # motion pass already examines the whole image; until the next stuck
000373:         # milestone, keep the normal local search around the last valid ball.
000374:         self._player_reacq_motion_failed_until_frame = -1
000375:         self.ball_velocity_history = []
000376:         self.initial_ball_position = None
000377:         self.last_seen_frame = None
000378:         self.edge_wait = False
000379:         self.last_delta = None
000380:         self.primary_hsv_lower = None
000381:         self.primary_hsv_upper = None
000382:         self.alt_hsv_lower = None
000383:         self.alt_hsv_upper = None
000384:         self.hsv_regular = None
000385:         self.hsv_behind_net = None
000386:         
000387:         # HSV filter (will be set based on clicked ball)
000388:         self.hsv_lower = None
000389:         self.hsv_upper = None
000390:         
000391:         # HSV values table for analysis
000392:         self.hsv_table = []
000393:         
000394:         self.motion_history = []
000395:         self.focus_loss_frame = None
000396:         self.focus_loss_active = False
```

## 'ball_size =' at line 465
```python
000430:                 np.array([103, 38, 255], dtype=np.uint8),
000431:                 0.0,
000432:             ),
000433:             (
000434:                 "click_upper_bright",
000435:                 np.array([90, 10, 170], dtype=np.uint8),
000436:                 np.array([108, 45, 255], dtype=np.uint8),
000437:                 18.0,
000438:             ),
000439:         ]
000440:         self._click_upper_hsv_recover_frame = -1000000
000441:         self.disable_click_upper_hsv = os.environ.get("DISABLE_CLICK_UPPER_HSV", "0") == "1"
000442:         self._top_far_out_deferred_candidate = None
000443:         # Keep behind-net / near-net HSV fully disabled by default.
000444:         # Set DISABLE_BEHIND_NET=0 to temporarily re-enable it if needed.
000445:         self.disable_behind_net_mode = os.environ.get("DISABLE_BEHIND_NET", "1") == "1"
000446:         self.direction_change_points = []
000447:         self.direction_change_events = []
000448:         self.show_event_markers = True
000449:         self.last_direction = None
000450:         self.near_edge = False
000451:         self.net_contact_points = []
000452:         self.prev_motion = None
000453:         self.max_ball_speed = 70  # overridden per-court via config ball_max_speed
000454:         self.direction_change_streak = 0
000455:         # Flag when per-frame movement jumps well beyond recent median
000456:         self.focus_loss_distance_threshold = 60.0
000457:         self.pause_requested = False
000458:         self.serve_direction_dx = 1
000459:         self.serve_direction_dy = 0
000460:         self.serve_directional_search = False
000461:         self.serve_directional_search_frames = 20
000462:         self.serve_contact_y_ratio = 0.7
000463:         self.serve_contact_min_prev_dy = 20
000464:         self.serve_contact_max_prev_dx = 40
000465:         self.serve_contact_min_ball_size = 100
000466:         self.serve_contact_min_dx = 80
000467:         self.serve_contact_min_dy = 0
000468:         self._base_serve_area = None
000469:         self._base_serve_area_end = None
000470:         self._configured_far_serve_area = None
000471:         self._active_serve_area_end = None
000472:         self._base_serve_direction_dx = 1
000473:         self._base_serve_direction_dy = -1
000474:         self._base_serve_ball_size_min = 3
000475:         self._base_serve_ball_size_max = 80
000476:         self.far_serve_ball_size_min = 1
000477:         self.far_serve_ball_size_max = 120
000478:         self._start_frame_match_seed_applied = False
000479:         self._court2_video_phase_seed_applied = False
000480:         self._serve_scan_block_until_frame = -1
000481:         self.point_start_frame_internal = None
000482:         self._serve_contact_grace_frames = 0
000483:         self._serve_launch_direction_x = 0
000484:         self._rally_contact_grace_frames = 0
000485:         self._rally_contact_large_ref_frames = 0
000486:         self._rally_contact_ref_size = None
000487:         self._rally_contact_origin = None
000488:         self._rally_contact_expected = None
000489:         self._rally_contact_progress = 0.0
000490:         self._rally_contact_high_launch_until_frame = -1
000491:         self._trusted_rally_launch_frame = -1000000
000492:         self._trusted_rally_launch_pos = None
000493:         self._last_racket_contact_frame = -1000000
000494:         self._last_racket_contact_point = None
000495:         self._last_racket_contact_player = None
000496:         # Metadata for the most recent motion-based re-acquisition.  A first
000497:         # candidate after a player/racket occlusion is provisional: tiny
000498:         # moving fragments on the net/court must not become the ball merely
000499:         # because they are the best HSV match.
000500:         self._last_reacq_candidate_area = 0.0
```

## 'ball_size =' at line 1915
```python
001880:                 f"ai={item['score']:.12f} ai_gap={ai_gap:.12g} "
001881:                 f"pred_dist={pd} cos={cs} anchor_dist={anchor_dist:.1f} "
001882:                 f"trajectory_cost={trajectory_cost:.1f} "
001883:                 f"normal_ref={debug_normal_candidate} normal_dist={nd}"
001884:             )
001885: 
001886:         if not eligible:
001887:             if bool(getattr(self, "_force_local_ai_return_none_on_miss", False)):
001888:                 print(
001889:                     f"[FORCE_LOCAL_AI_MISS] f{current}: no clear candidate >= {min_score:.3f}; "
001890:                     f"anchor={anchor} predicted={predicted} candidates={len(candidates)} "
001891:                     f"clear={len(clear_candidates)} scored={len(scored)} "
001892:                     f"continuity_rejected={continuity_rejected}"
001893:                 )
001894:                 return None
001895:             self.ball_center = tuple(anchor)
001896:             self.stuck_frame_count = max(1, int(getattr(self, "stuck_frame_count", 0)) + 1)
001897:             print(
001898:                 f"[FORCE_LOCAL_AI_HOLD] f{current}: no clear candidate >= {min_score:.3f}; "
001899:                 f"anchor={anchor} predicted={predicted} candidates={len(candidates)} "
001900:                 f"clear={len(clear_candidates)} scored={len(scored)}"
001901:             )
001902:             return tuple(anchor)
001903: 
001904:         selected = eligible[0]
001905:         candidate = selected["candidate"]
001906:         point = selected["point"]
001907:         area = float(candidate.get("area", 0.0) or 0.0)
001908: 
001909:         dx = int(point[0] - anchor[0])
001910:         dy = int(point[1] - anchor[1])
001911:         distance = math.hypot(dx, dy)
001912:         direction_deg = math.degrees(math.atan2(dy, dx)) if distance > 0.0 else None
001913: 
001914:         self.ball_center = point
001915:         self.ball_size = area
001916:         height, width = frame.shape[:2]
001917:         px = max(0, min(width - 1, point[0]))
001918:         py = max(0, min(height - 1, point[1]))
001919:         hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
001920:         self.ball_hsv = hsv_frame[py, px]
001921: 
001922:         self.prev_motion = self.last_motion
001923:         self.last_motion = {
001924:             "distance": float(distance), "dx": dx, "dy": dy,
001925:             "direction_deg": direction_deg,
001926:         }
001927:         if distance >= 3.0:
001928:             self.last_nonzero_motion = dict(self.last_motion)
001929:         if direction_deg is not None:
001930:             self.last_direction = direction_deg
001931:         self.last_delta = (dx, dy)
001932:         self.stuck_frame_count = 0
001933:         self.ball_stopped = False
001934:         self.last_seen_frame = current
001935: 
001936:         self.ball_velocity_history.append(float(distance))
001937:         if len(self.ball_velocity_history) > 10:
001938:             self.ball_velocity_history = self.ball_velocity_history[-10:]
001939:         self.motion_history.append({
001940:             "frame": current,
001941:             "distance": float(distance),
001942:             "direction_deg": direction_deg,
001943:             "pos": tuple(point),
001944:             "prev_pos": tuple(anchor),
001945:             "forced_local_ai": True,
001946:         })
001947:         if len(self.motion_history) > 200:
001948:             self.motion_history = self.motion_history[-200:]
001949:         self._prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
001950: 
```

## 'ball_size =' at line 2773
```python
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
```

## 'ball_size =' at line 7153
```python
007118:                     },
007119:                 ))
007120:         if not candidates:
007121:             print(
007122:                 f"Frame {self.frame_count}: [LOWER-HIT OUTBOUND MISS] "
007123:                 f"expected=({expected[0]:.0f},{expected[1]:.0f}) "
007124:                 f"nearby={' | '.join(nearby_debug[:8]) or 'none'}"
007125:             )
007126:             return None
007127:         # This helper returns before the normal grace-window decrement below.
007128:         # Consume one frame here so a good contact continuation cannot extend
007129:         # the special relaxed rules indefinitely into unrelated court blobs.
007130:         self._rally_contact_grace_frames = max(
007131:             0, int(getattr(self, '_rally_contact_grace_frames', 0)) - 1
007132:         )
007133:         return min(candidates, key=lambda item: item[:3])[3]
007134: 
007135:     def _commit_night_visible_ball_recovery(self, candidate, frame):
007136:         # Some visible-ball recovery paths (notably player-reacquisition) jump
007137:         # directly here and bypass the startup candidate arbiter.  Reuse the
007138:         # same pre-commit tight-ROI guard so an active near-player lock can
007139:         # reject that jump before it mutates motion/contact state.
007140:         if (
007141:                 frame is not None and
007142:                 candidate.get('source') != 'local_ai_tight_roi' and
007143:                 int(getattr(self, '_local_ai_tight_roi_attempt_frame', -1000000)) !=
007144:                 int(self.frame_count)):
007145:             commit_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
007146:             candidate = self._prefer_night_tight_local_ai_candidate(
007147:                 frame, commit_gray, candidate
007148:             )
007149:         prev_pos = self.ball_center
007150:         new_pos = candidate['pos']
007151:         self.ball_center = new_pos
007152:         self.ball_hsv = candidate['hsv']
007153:         self.ball_size = candidate['area']
007154:         self._update_recovered_motion(prev_pos, new_pos)
007155:         motion_override = candidate.get('local_ai_motion_override')
007156:         if isinstance(motion_override, dict):
007157:             self.last_motion = dict(motion_override)
007158:             self.last_delta = (
007159:                 float(motion_override.get('dx', 0.0) or 0.0),
007160:                 float(motion_override.get('dy', 0.0) or 0.0),
007161:             )
007162:             self.last_direction = float(
007163:                 motion_override.get('direction_deg', 0.0) or 0.0
007164:             )
007165:             velocity_history = candidate.get('local_ai_velocity_history')
007166:             if isinstance(velocity_history, list):
007167:                 self.ball_velocity_history = list(velocity_history)[-5:]
007168:             if float(motion_override.get('distance', 0.0) or 0.0) > 0.0:
007169:                 self.last_nonzero_motion = dict(motion_override)
007170:         self.stuck_frame_count = 0
007171:         # Short grace period for a player/racket pixel held during occlusion.
007172:         self._player_occlusion_hold_frames = 0
007173:         self.direction_change_streak = 0
007174:         self._post_reacq_frames = max(getattr(self, '_post_reacq_frames', 0), 3)
007175:         self._last_motion_reacq_frame = self.frame_count
007176:         # Pair the recovery marker with the exact accepted position. The outer
007177:         # main-loop jump guard intentionally trusts a large same-frame motion
007178:         # reacquisition only when both frame and position match. Other motion
007179:         # reacquisition paths already maintain this pair; visible-ball recovery
007180:         # previously set only the frame, so a genuine recovery could be rejected
007181:         # after a poisoned HSV anchor.
007182:         self._last_motion_reacq_pos = tuple(new_pos)
007183:         self.last_seen_frame = self.frame_count
007184:         self._last_tracked_candidate_frame = self.frame_count
007185:         self._last_tracked_candidate_motion_frame = self.frame_count
007186:         self._last_tracked_candidate_motion_mean = float(candidate.get('motion_mean', 0.0) or 0.0)
007187:         self._last_tracked_candidate_motion_max = float(candidate.get('motion_max', 0.0) or 0.0)
007188:         if candidate.get('source') == 'local_ai_tight_roi':
```

## 'ball_size =' at line 8143
```python
008108:         if abs(ball_x - contact_x) > max(420, int(frame_shape[1] * 0.16)):
008109:             return False
008110:         if self.ball_size is not None and self.ball_size > 35:
008111:             return False
008112: 
008113:         motion = self.last_motion
008114:         if motion is None or motion.get('distance', 0.0) < 3.0:
008115:             motion = getattr(self, 'last_nonzero_motion', None)
008116:         if motion is None:
008117:             return False
008118: 
008119:         dy = float(motion.get('dy', 0.0) or 0.0)
008120:         dist = float(motion.get('distance', 0.0) or 0.0)
008121:         recent_vel = max(self.ball_velocity_history[-3:]) if getattr(self, 'ball_velocity_history', None) else 0.0
008122:         return dy <= -5.0 or dist >= 20.0 or recent_vel >= 18.0
008123: 
008124:     def _should_start_top_return_wait(self, frame_shape):
008125:         """Return the top-return mode for long upper exits that can re-enter later."""
008126:         if self.ball_center is None:
008127:             return None
008128:         if self._top_return_recently_expired_at_current_anchor():
008129:             return None
008130:         motion = self.last_motion
008131:         if motion is None or motion.get('distance', 0.0) < 3.0:
008132:             motion = getattr(self, 'last_nonzero_motion', None)
008133:         if motion is None:
008134:             return None
008135:         if getattr(self, '_top_return_reentry_grace_frames', 0) > 0:
008136:             return None
008137:         frame_height, _ = frame_shape[:2]
008138:         x_prev, y_prev = self.ball_center
008139:         dy = float(motion.get('dy', 0.0) or 0.0)
008140:         dist = float(motion.get('distance', 0.0) or 0.0)
008141:         predicted_y = y_prev + dy
008142:         recent_vel = max(self.ball_velocity_history[-3:]) if getattr(self, 'ball_velocity_history', None) else 0.0
008143:         ball_size = float(self.ball_size or 0.0)
008144:         recent_upper_racket_exit = self._recent_upper_racket_top_exit_context(frame_shape)
008145: 
008146:         if recent_upper_racket_exit:
008147:             clipped_recent_exit = (
008148:                 max(34, int(frame_height * 0.015)) <= y_prev <= 40 and
008149:                 dy < -12.0 and
008150:                 dist >= 18.0 and
008151:                 predicted_y < 12.0
008152:             )
008153:             if clipped_recent_exit:
008154:                 return "upper_racket"
008155: 
008156:         if y_prev <= 40 and dy < -14.0 and dist >= 20.0:
008157:             if self.ground_bounce_count > 0 and (self.ball_size is None or self.ball_size <= 18.0):
008158:                 return "edge"
008159:             if (
008160:                 dy <= -35.0 and
008161:                 dist >= 45.0 and
008162:                 predicted_y < 8.0 and
008163:                 (ball_size >= 60.0 or recent_vel >= 50.0)
008164:             ):
008165:                 return "upper_side"
008166:         projected_top_exit = (
008167:             y_prev <= max(72, int(frame_height * 0.035)) and
008168:             dy <= -35.0 and
008169:             dist >= 45.0 and
008170:             predicted_y <= max(8, int(frame_height * 0.006)) and
008171:             (
008172:                 self.ground_bounce_count > 0 or
008173:                 ball_size >= 60.0 or
008174:                 recent_vel >= 50.0
008175:             )
008176:         )
008177:         if projected_top_exit:
008178:             return "edge"
```

## 'ball_size =' at line 8706
```python
008671:                     (cx, cy), area, motion_mean, motion_max, frame.shape
008672:                 )
008673:                 if not ok:
008674:                     continue
008675: 
008676:                 if mode == "edge_clip":
008677:                     score = abs(cx - expected_x) * 0.9
008678:                     score += cy * 1.8
008679:                 else:
008680:                     score = abs(cy - 28) * 2.2
008681:                     score += abs(cx - (self._top_return_anchor[0] + self._top_return_exit_dx * 13.0)) * 0.05
008682:                 score -= min(220.0, motion_max * 1.2)
008683:                 score -= min(120.0, motion_mean * 3.0)
008684:                 score -= min(90.0, area * 4.0)
008685:                 if score < best_score:
008686:                     local_x = max(0, min(search_frame.shape[1] - 1, cx - x1))
008687:                     local_y = max(0, min(search_frame.shape[0] - 1, cy - y1))
008688:                     best_score = score
008689:                     best = {
008690:                         'pos': (cx, cy),
008691:                         'area': area,
008692:                         'hsv': hsv_frame[local_y, local_x],
008693:                         'label': label,
008694:                         'motion_mean': motion_mean,
008695:                         'motion_max': motion_max,
008696:                         'score': score,
008697:                     }
008698: 
008699:         return best
008700: 
008701:     def _commit_top_return_reentry(self, reentry, label):
008702:         prev_pos = self.ball_center
008703:         new_pos = reentry['pos']
008704:         self.ball_center = new_pos
008705:         self.ball_hsv = reentry['hsv']
008706:         self.ball_size = reentry['area']
008707:         self._update_recovered_motion(prev_pos, new_pos)
008708:         self.stuck_frame_count = 0
008709:         self._recent_offscreen_return_frame = self.frame_count
008710:         self._top_return_reentry_grace_frames = max(
008711:             getattr(self, '_top_return_reentry_grace_frames', 0), 4
008712:         )
008713:         self._top_return_wait_frames = 0
008714:         self._top_return_anchor = None
008715:         self._top_return_origin_frame = -1
008716:         self._top_return_mode = None
008717:         self._top_return_exit_dx = 0.0
008718:         self.last_seen_frame = self.frame_count
008719:         print(
008720:             f"Frame {self.frame_count}: [{label}] Ball at {new_pos} "
008721:             f"via {reentry['label']} motion={reentry['motion_mean']:.1f}/"
008722:             f"{reentry['motion_max']:.1f}"
008723:         )
008724:         return self.ball_center
008725: 
008726:     def _active_top_return_overlay_region(self, frame_shape):
008727:         if getattr(self, '_top_return_anchor', None) is None:
008728:             return None
008729:         if not (
008730:             self._top_return_wait_active() or
008731:             getattr(self, '_top_return_reentry_grace_frames', 0) > 0
008732:         ):
008733:             return None
008734:         _, _, x1, y1, x2, y2 = self._build_top_return_search_region(frame_shape)
008735:         return x1, y1, x2, y2
008736: 
008737:     def _prune_bounce_markers(self):
008738:         self.recent_bounce_markers = [
008739:             marker for marker in getattr(self, 'recent_bounce_markers', [])
008740:             if marker.get('expires', -1) >= self.frame_count
008741:         ]
```

## 'ball_size =' at line 12961
```python
012926:             return False
012927:         if motion_mean is not None and motion_max is not None:
012928:             if motion_mean < 12.0 and motion_max < 70.0:
012929:                 return False
012930:         return True
012931: 
012932:     def _upper_slow_arc_active(self):
012933:         return self.frame_count <= getattr(self, '_upper_slow_arc_until_frame', -1)
012934: 
012935:     def _upper_fence_fall_end_candidate(self, ball_position, frame_shape):
012936:         if ball_position is None or self.last_motion is None:
012937:             self._upper_fence_fall_frames = 0
012938:             return False
012939:         # A point inside a tracked player's body/head/racket is not evidence
012940:         # of a fence contact.  During this bounded recovery window the ball is
012941:         # intentionally held at its last trusted position, so never let the
012942:         # stale fragment score as an upper-fence fall.
012943:         if int(getattr(self, '_player_reacq_protect_until_frame', -1)) >= self.frame_count:
012944:             self._upper_fence_fall_frames = 0
012945:             return False
012946:         if self._player_point_zone(ball_position) in (
012947:                 'player_body', 'player_head_hat', 'player_shoes', 'racket_fragment'):
012948:             self._upper_fence_fall_frames = 0
012949:             return False
012950:         if not self._upper_slow_arc_active():
012951:             self._upper_fence_fall_frames = 0
012952:             return False
012953: 
012954:         frame_height, frame_width = frame_shape[:2]
012955:         x, y = ball_position
012956:         upper_fence_y_min = max(150, int(frame_height * 0.070))
012957:         upper_fence_y_max = max(205, int(frame_height * 0.095))
012958:         lane_min_x = int(frame_width * 0.35)
012959:         lane_max_x = int(frame_width * 0.70)
012960: 
012961:         ball_size = float(self.ball_size or 0.0)
012962:         speed = float(self.last_motion.get('distance', 0.0) or 0.0)
012963:         dy = float(self.last_motion.get('dy', 0.0) or 0.0)
012964:         continuing_upper_flight = (
012965:             self._upper_slow_arc_active() and
012966:             speed > 10.0 and
012967:             dy < -3.0
012968:         )
012969:         in_upper_fence_band = (
012970:             lane_min_x <= x <= lane_max_x and
012971:             upper_fence_y_min <= y <= upper_fence_y_max
012972:         )
012973:         tiny_visible_ball = 3.0 <= ball_size <= 16.0
012974:         settled_or_falling = (speed <= 6.5 and dy >= -3.0) or getattr(self, 'stuck_frame_count', 0) >= 1
012975: 
012976:         if self._is_night_session_config():
012977:             night_confirmed_fall = (
012978:                 (speed <= 7.0 and dy >= 1.0) or
012979:                 getattr(self, 'stuck_frame_count', 0) >= 4
012980:             )
012981:             if not night_confirmed_fall:
012982:                 self._upper_fence_fall_frames = 0
012983:                 return False
012984: 
012985:         if (not in_upper_fence_band or not tiny_visible_ball or
012986:                 continuing_upper_flight or not settled_or_falling):
012987:             self._upper_fence_fall_frames = 0
012988:             return False
012989: 
012990:         self._upper_fence_fall_frames += 1
012991:         if self._upper_fence_fall_frames < 2:
012992:             return False
012993: 
012994:         self._add_impact_marker(
012995:             ball_position,
012996:             kind="fence_contact",
```

## 'current_ball_size' at line 24686
```python
024651:             right_gap = float(curr_right_x) - float(x)
024652:             near_band = max(24.0, sideline_margin * 2.0)
024653:             if 0.0 <= left_gap <= near_band and curr_dx <= -4.0:
024654:                 soft_out_near_sideline = True
024655:                 curr_side = 'left'
024656:             elif 0.0 <= right_gap <= near_band and curr_dx >= 4.0:
024657:                 soft_out_near_sideline = True
024658:                 curr_side = 'right'
024659:             if soft_out_near_sideline:
024660:                 print(
024661:                     f"Frame {self.frame_count}: [SOFT OUT SIDELINE BAND] "
024662:                     f"pos={ball_position} side={curr_side} "
024663:                     f"left_gap={left_gap:.1f} right_gap={right_gap:.1f} "
024664:                     f"band={near_band:.1f}"
024665:                 )
024666:         if not prev_pos_outside and not curr_pos_outside:
024667:             if not soft_out_near_sideline:
024668:                 return False, None
024669: 
024670:         side = prev_side or curr_side
024671:         left_x = prev_left_x if prev_pos_outside else curr_left_x
024672:         right_x = prev_right_x if prev_pos_outside else curr_right_x
024673:         if side is None or left_x is None or right_x is None:
024674:             return False, None
024675: 
024676:         model = self._build_singles_sideline_model(frame)
024677:         sideline_margin = float(model['margin']) if model is not None else 12.0
024678:         outside_x = prev_pos[0] if prev_pos_outside else x
024679:         outside_left_x = prev_left_x if prev_pos_outside else curr_left_x
024680:         outside_right_x = prev_right_x if prev_pos_outside else curr_right_x
024681:         outside_depth = 0.0
024682:         if side == 'left' and outside_left_x is not None:
024683:             outside_depth = max(0.0, float(outside_left_x) - float(outside_x))
024684:         elif side == 'right' and outside_right_x is not None:
024685:             outside_depth = max(0.0, float(outside_x) - float(outside_right_x))
024686:         current_ball_size = float(getattr(self, 'ball_size', 0.0) or 0.0)
024687:         if self._is_night_session_config():
024688:             last_candidate_motion_frame = int(
024689:                 getattr(self, '_last_tracked_candidate_motion_frame', -1000000)
024690:             )
024691:             last_candidate_motion_mean = float(
024692:                 getattr(self, '_last_tracked_candidate_motion_mean', 0.0) or 0.0
024693:             )
024694:             last_candidate_motion_max = float(
024695:                 getattr(self, '_last_tracked_candidate_motion_max', 0.0) or 0.0
024696:             )
024697:             weak_current_candidate = (
024698:                 last_candidate_motion_frame == self.frame_count and
024699:                 last_candidate_motion_mean < 3.5 and
024700:                 last_candidate_motion_max < 18.0
024701:             )
024702:             deep_static_out = (
024703:                 weak_current_candidate and
024704:                 current_ball_size <= 90.0 and
024705:                 (
024706:                     outside_depth >= max(90.0, sideline_margin * 5.0) or
024707:                     x <= max(70, int(frame.shape[1] * 0.018)) or
024708:                     x >= frame.shape[1] - max(70, int(frame.shape[1] * 0.018))
024709:                 )
024710:             )
024711:             if deep_static_out:
024712:                 # V14: only arm a pending OUT when the static-looking point is
024713:                 # still supported by a coherent descending flight. Pure side
024714:                 # hotspots remain ordinary suppressions and never get promoted.
024715:                 if recent_descending and curr_dy >= 6.0 and curr_speed >= 6.0:
024716:                     self._pending_night_static_out = {
024717:                         'frame': int(self.frame_count),
024718:                         'position': tuple(ball_position),
024719:                         'side': side,
024720:                         'dx': curr_dx,
024721:                         'dy': curr_dy,
```

## 'current_ball_size' at line 24704
```python
024669: 
024670:         side = prev_side or curr_side
024671:         left_x = prev_left_x if prev_pos_outside else curr_left_x
024672:         right_x = prev_right_x if prev_pos_outside else curr_right_x
024673:         if side is None or left_x is None or right_x is None:
024674:             return False, None
024675: 
024676:         model = self._build_singles_sideline_model(frame)
024677:         sideline_margin = float(model['margin']) if model is not None else 12.0
024678:         outside_x = prev_pos[0] if prev_pos_outside else x
024679:         outside_left_x = prev_left_x if prev_pos_outside else curr_left_x
024680:         outside_right_x = prev_right_x if prev_pos_outside else curr_right_x
024681:         outside_depth = 0.0
024682:         if side == 'left' and outside_left_x is not None:
024683:             outside_depth = max(0.0, float(outside_left_x) - float(outside_x))
024684:         elif side == 'right' and outside_right_x is not None:
024685:             outside_depth = max(0.0, float(outside_x) - float(outside_right_x))
024686:         current_ball_size = float(getattr(self, 'ball_size', 0.0) or 0.0)
024687:         if self._is_night_session_config():
024688:             last_candidate_motion_frame = int(
024689:                 getattr(self, '_last_tracked_candidate_motion_frame', -1000000)
024690:             )
024691:             last_candidate_motion_mean = float(
024692:                 getattr(self, '_last_tracked_candidate_motion_mean', 0.0) or 0.0
024693:             )
024694:             last_candidate_motion_max = float(
024695:                 getattr(self, '_last_tracked_candidate_motion_max', 0.0) or 0.0
024696:             )
024697:             weak_current_candidate = (
024698:                 last_candidate_motion_frame == self.frame_count and
024699:                 last_candidate_motion_mean < 3.5 and
024700:                 last_candidate_motion_max < 18.0
024701:             )
024702:             deep_static_out = (
024703:                 weak_current_candidate and
024704:                 current_ball_size <= 90.0 and
024705:                 (
024706:                     outside_depth >= max(90.0, sideline_margin * 5.0) or
024707:                     x <= max(70, int(frame.shape[1] * 0.018)) or
024708:                     x >= frame.shape[1] - max(70, int(frame.shape[1] * 0.018))
024709:                 )
024710:             )
024711:             if deep_static_out:
024712:                 # V14: only arm a pending OUT when the static-looking point is
024713:                 # still supported by a coherent descending flight. Pure side
024714:                 # hotspots remain ordinary suppressions and never get promoted.
024715:                 if recent_descending and curr_dy >= 6.0 and curr_speed >= 6.0:
024716:                     self._pending_night_static_out = {
024717:                         'frame': int(self.frame_count),
024718:                         'position': tuple(ball_position),
024719:                         'side': side,
024720:                         'dx': curr_dx,
024721:                         'dy': curr_dy,
024722:                         'speed': curr_speed,
024723:                     }
024724:                     print(
024725:                         f"Frame {self.frame_count}: [OUT-BOUNCE PENDING] "
024726:                         f"point={tuple(ball_position)} side={side} "
024727:                         f"motion=({curr_dx:.1f},{curr_dy:.1f}) speed={curr_speed:.1f}px"
024728:                     )
024729:                 self._last_out_bounce_suppressed_frame = self.frame_count
024730:                 self._last_out_bounce_suppressed_point = tuple(ball_position)
024731:                 print(
024732:                     f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
024733:                     f"night static side artifact pos={ball_position} "
024734:                     f"size={current_ball_size:.1f}px outside_depth={outside_depth:.1f} "
024735:                     f"motion={last_candidate_motion_mean:.1f}/{last_candidate_motion_max:.1f}"
024736:                 )
024737:                 return False, None
024738: 
024739:         # A ball that was travelling down and *outward* can only be a
```

## 'current_ball_size' at line 24734
```python
024699:                 last_candidate_motion_mean < 3.5 and
024700:                 last_candidate_motion_max < 18.0
024701:             )
024702:             deep_static_out = (
024703:                 weak_current_candidate and
024704:                 current_ball_size <= 90.0 and
024705:                 (
024706:                     outside_depth >= max(90.0, sideline_margin * 5.0) or
024707:                     x <= max(70, int(frame.shape[1] * 0.018)) or
024708:                     x >= frame.shape[1] - max(70, int(frame.shape[1] * 0.018))
024709:                 )
024710:             )
024711:             if deep_static_out:
024712:                 # V14: only arm a pending OUT when the static-looking point is
024713:                 # still supported by a coherent descending flight. Pure side
024714:                 # hotspots remain ordinary suppressions and never get promoted.
024715:                 if recent_descending and curr_dy >= 6.0 and curr_speed >= 6.0:
024716:                     self._pending_night_static_out = {
024717:                         'frame': int(self.frame_count),
024718:                         'position': tuple(ball_position),
024719:                         'side': side,
024720:                         'dx': curr_dx,
024721:                         'dy': curr_dy,
024722:                         'speed': curr_speed,
024723:                     }
024724:                     print(
024725:                         f"Frame {self.frame_count}: [OUT-BOUNCE PENDING] "
024726:                         f"point={tuple(ball_position)} side={side} "
024727:                         f"motion=({curr_dx:.1f},{curr_dy:.1f}) speed={curr_speed:.1f}px"
024728:                     )
024729:                 self._last_out_bounce_suppressed_frame = self.frame_count
024730:                 self._last_out_bounce_suppressed_point = tuple(ball_position)
024731:                 print(
024732:                     f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
024733:                     f"night static side artifact pos={ball_position} "
024734:                     f"size={current_ball_size:.1f}px outside_depth={outside_depth:.1f} "
024735:                     f"motion={last_candidate_motion_mean:.1f}/{last_candidate_motion_max:.1f}"
024736:                 )
024737:                 return False, None
024738: 
024739:         # A ball that was travelling down and *outward* can only be a
024740:         # sideline-bounce candidate if its rebound remains physically
024741:         # compatible with that path.  At a prior frame the near player returns the
024742:         # ball: it is large/visible, reverses upward and strongly back toward
024743:         # the court.  Treating that as an out bounce because a prior frame happened to
024744:         # be outside the left line ends the rally before the return.  Keep this
024745:         # deliberately narrow so ordinary small/airborne sideline outs still
024746:         # use the existing reversal detector.
024747:         inward_racket_return = (
024748:             prev_pos_outside and
024749:             curr_pos_outside and
024750:             current_ball_size >= 260.0 and
024751:             prev_speed >= 45.0 and
024752:             curr_speed >= 60.0 and
024753:             prev_dy >= 30.0 and
024754:             curr_dy <= -50.0 and
024755:             (
024756:                 (side == 'left' and prev_dx <= -30.0 and curr_dx >= 30.0) or
024757:                 (side == 'right' and prev_dx >= 30.0 and curr_dx <= -30.0)
024758:             )
024759:         )
024760:         if inward_racket_return:
024761:             self._last_out_bounce_suppressed_frame = self.frame_count
024762:             self._last_out_bounce_suppressed_point = tuple(ball_position)
024763:             self._record_racket_contact(ball_position, label='inward sideline return')
024764:             print(
024765:                 f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
024766:                 f"large inward racket return pos={ball_position} side={side} "
024767:                 f"prev_motion=({prev_dx:.1f},{prev_dy:.1f}) "
024768:                 f"curr_motion=({curr_dx:.1f},{curr_dy:.1f}) "
024769:                 f"size={current_ball_size:.1f}px"
```

## 'current_ball_size' at line 24750
```python
024715:                 if recent_descending and curr_dy >= 6.0 and curr_speed >= 6.0:
024716:                     self._pending_night_static_out = {
024717:                         'frame': int(self.frame_count),
024718:                         'position': tuple(ball_position),
024719:                         'side': side,
024720:                         'dx': curr_dx,
024721:                         'dy': curr_dy,
024722:                         'speed': curr_speed,
024723:                     }
024724:                     print(
024725:                         f"Frame {self.frame_count}: [OUT-BOUNCE PENDING] "
024726:                         f"point={tuple(ball_position)} side={side} "
024727:                         f"motion=({curr_dx:.1f},{curr_dy:.1f}) speed={curr_speed:.1f}px"
024728:                     )
024729:                 self._last_out_bounce_suppressed_frame = self.frame_count
024730:                 self._last_out_bounce_suppressed_point = tuple(ball_position)
024731:                 print(
024732:                     f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
024733:                     f"night static side artifact pos={ball_position} "
024734:                     f"size={current_ball_size:.1f}px outside_depth={outside_depth:.1f} "
024735:                     f"motion={last_candidate_motion_mean:.1f}/{last_candidate_motion_max:.1f}"
024736:                 )
024737:                 return False, None
024738: 
024739:         # A ball that was travelling down and *outward* can only be a
024740:         # sideline-bounce candidate if its rebound remains physically
024741:         # compatible with that path.  At a prior frame the near player returns the
024742:         # ball: it is large/visible, reverses upward and strongly back toward
024743:         # the court.  Treating that as an out bounce because a prior frame happened to
024744:         # be outside the left line ends the rally before the return.  Keep this
024745:         # deliberately narrow so ordinary small/airborne sideline outs still
024746:         # use the existing reversal detector.
024747:         inward_racket_return = (
024748:             prev_pos_outside and
024749:             curr_pos_outside and
024750:             current_ball_size >= 260.0 and
024751:             prev_speed >= 45.0 and
024752:             curr_speed >= 60.0 and
024753:             prev_dy >= 30.0 and
024754:             curr_dy <= -50.0 and
024755:             (
024756:                 (side == 'left' and prev_dx <= -30.0 and curr_dx >= 30.0) or
024757:                 (side == 'right' and prev_dx >= 30.0 and curr_dx <= -30.0)
024758:             )
024759:         )
024760:         if inward_racket_return:
024761:             self._last_out_bounce_suppressed_frame = self.frame_count
024762:             self._last_out_bounce_suppressed_point = tuple(ball_position)
024763:             self._record_racket_contact(ball_position, label='inward sideline return')
024764:             print(
024765:                 f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
024766:                 f"large inward racket return pos={ball_position} side={side} "
024767:                 f"prev_motion=({prev_dx:.1f},{prev_dy:.1f}) "
024768:                 f"curr_motion=({curr_dx:.1f},{curr_dy:.1f}) "
024769:                 f"size={current_ball_size:.1f}px"
024770:             )
024771:             return False, None
024772: 
024773:         outward_motion = (
024774:             (side == 'left' and curr_dx <= -4.0) or
024775:             (side == 'right' and curr_dx >= 4.0)
024776:         )
024777:         previous_outward_motion = (
024778:             (side == 'left' and prev_dx <= -4.0) or
024779:             (side == 'right' and prev_dx >= 4.0)
024780:         )
024781:         # A real sideline OUT can bounce with only a shallow upward Y reversal.
024782:         # Do not require the old -12px vertical rebound when the ball was already
024783:         # clearly outside, descending for several frames, and continues outward.
024784:         # Reviewed f2486->f2487: (-23,+26) -> (-36,-6).
024785:         max_physical_out_bounce_step = max(125.0, float(frame.shape[1]) * 0.035)
```

## 'current_ball_size' at line 24769
```python
024734:                     f"size={current_ball_size:.1f}px outside_depth={outside_depth:.1f} "
024735:                     f"motion={last_candidate_motion_mean:.1f}/{last_candidate_motion_max:.1f}"
024736:                 )
024737:                 return False, None
024738: 
024739:         # A ball that was travelling down and *outward* can only be a
024740:         # sideline-bounce candidate if its rebound remains physically
024741:         # compatible with that path.  At a prior frame the near player returns the
024742:         # ball: it is large/visible, reverses upward and strongly back toward
024743:         # the court.  Treating that as an out bounce because a prior frame happened to
024744:         # be outside the left line ends the rally before the return.  Keep this
024745:         # deliberately narrow so ordinary small/airborne sideline outs still
024746:         # use the existing reversal detector.
024747:         inward_racket_return = (
024748:             prev_pos_outside and
024749:             curr_pos_outside and
024750:             current_ball_size >= 260.0 and
024751:             prev_speed >= 45.0 and
024752:             curr_speed >= 60.0 and
024753:             prev_dy >= 30.0 and
024754:             curr_dy <= -50.0 and
024755:             (
024756:                 (side == 'left' and prev_dx <= -30.0 and curr_dx >= 30.0) or
024757:                 (side == 'right' and prev_dx >= 30.0 and curr_dx <= -30.0)
024758:             )
024759:         )
024760:         if inward_racket_return:
024761:             self._last_out_bounce_suppressed_frame = self.frame_count
024762:             self._last_out_bounce_suppressed_point = tuple(ball_position)
024763:             self._record_racket_contact(ball_position, label='inward sideline return')
024764:             print(
024765:                 f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
024766:                 f"large inward racket return pos={ball_position} side={side} "
024767:                 f"prev_motion=({prev_dx:.1f},{prev_dy:.1f}) "
024768:                 f"curr_motion=({curr_dx:.1f},{curr_dy:.1f}) "
024769:                 f"size={current_ball_size:.1f}px"
024770:             )
024771:             return False, None
024772: 
024773:         outward_motion = (
024774:             (side == 'left' and curr_dx <= -4.0) or
024775:             (side == 'right' and curr_dx >= 4.0)
024776:         )
024777:         previous_outward_motion = (
024778:             (side == 'left' and prev_dx <= -4.0) or
024779:             (side == 'right' and prev_dx >= 4.0)
024780:         )
024781:         # A real sideline OUT can bounce with only a shallow upward Y reversal.
024782:         # Do not require the old -12px vertical rebound when the ball was already
024783:         # clearly outside, descending for several frames, and continues outward.
024784:         # Reviewed f2486->f2487: (-23,+26) -> (-36,-6).
024785:         max_physical_out_bounce_step = max(125.0, float(frame.shape[1]) * 0.035)
024786:         shallow_outward_rebound = (
024787:             prev_pos_outside and
024788:             curr_pos_outside and
024789:             recent_descending and
024790:             previous_outward_motion and
024791:             outward_motion and
024792:             prev_dy >= 14.0 and
024793:             -12.0 < curr_dy <= -3.0 and
024794:             prev_speed >= 18.0 and
024795:             18.0 <= curr_speed <= max_physical_out_bounce_step
024796:         )
024797:         soft_out_reversal = (
024798:             (
024799:                 (
024800:                     curr_pos_outside and
024801:                     outside_depth >= max(20.0, sideline_margin * 1.5)
024802:                 ) or
024803:                 soft_out_near_sideline
024804:             ) and
```

## 'current_ball_size' at line 24815
```python
024780:         )
024781:         # A real sideline OUT can bounce with only a shallow upward Y reversal.
024782:         # Do not require the old -12px vertical rebound when the ball was already
024783:         # clearly outside, descending for several frames, and continues outward.
024784:         # Reviewed f2486->f2487: (-23,+26) -> (-36,-6).
024785:         max_physical_out_bounce_step = max(125.0, float(frame.shape[1]) * 0.035)
024786:         shallow_outward_rebound = (
024787:             prev_pos_outside and
024788:             curr_pos_outside and
024789:             recent_descending and
024790:             previous_outward_motion and
024791:             outward_motion and
024792:             prev_dy >= 14.0 and
024793:             -12.0 < curr_dy <= -3.0 and
024794:             prev_speed >= 18.0 and
024795:             18.0 <= curr_speed <= max_physical_out_bounce_step
024796:         )
024797:         soft_out_reversal = (
024798:             (
024799:                 (
024800:                     curr_pos_outside and
024801:                     outside_depth >= max(20.0, sideline_margin * 1.5)
024802:                 ) or
024803:                 soft_out_near_sideline
024804:             ) and
024805:             (soft_turn_motion or shallow_outward_rebound) and
024806:             outward_motion and
024807:             curr_speed >= 18.0
024808:         )
024809:         if soft_out_reversal:
024810:             frame_height = frame.shape[0] if frame is not None else 0
024811:             lower_court_y = max(
024812:                 int(frame_height * 0.62),
024813:                 int(getattr(self, 'net_area_y_min', 0)) + 520,
024814:             )
024815:             if y >= lower_court_y and current_ball_size <= 3.0:
024816:                 print(
024817:                     f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
024818:                     f"tiny lower-court soft sideline trace pos={ball_position} "
024819:                     f"size={current_ball_size:.1f}px lower_y={lower_court_y}"
024820:                 )
024821:                 return False, None
024822:         if (prev_speed < 20.0 or curr_speed < 12.0) and not soft_out_reversal:
024823:             return False, None
024824: 
024825:         if prev_pos[1] < near_side_min_y and y < near_side_min_y and not soft_out_reversal:
024826:             return False, None
024827: 
024828:         prev_dir = self.prev_motion.get('direction_deg')
024829:         curr_dir = self.last_motion.get('direction_deg')
024830:         angle_diff = 0.0
024831:         if prev_dir is not None and curr_dir is not None:
024832:             delta = abs(curr_dir - prev_dir) % 360
024833:             angle_diff = min(delta, 360 - delta)
024834: 
024835:         # A very large inward X reversal while already outside is a
024836:         # reacquisition/artifact jump, not a ground bounce.
024837:         inward_horizontal_reversal = (
024838:             (side == 'left' and prev_dx <= -8.0 and curr_dx >= 8.0) or
024839:             (side == 'right' and prev_dx >= 8.0 and curr_dx <= -8.0)
024840:         )
024841:         if (
024842:             prev_pos_outside and
024843:             inward_horizontal_reversal and
024844:             curr_speed > max_physical_out_bounce_step
024845:         ):
024846:             self._last_out_bounce_suppressed_frame = self.frame_count
024847:             self._last_out_bounce_suppressed_point = tuple(ball_position)
024848:             print(
024849:                 f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
024850:                 f"implausible inward jump pos={ball_position} side={side} "
```

## 'current_ball_size' at line 24819
```python
024784:         # Reviewed f2486->f2487: (-23,+26) -> (-36,-6).
024785:         max_physical_out_bounce_step = max(125.0, float(frame.shape[1]) * 0.035)
024786:         shallow_outward_rebound = (
024787:             prev_pos_outside and
024788:             curr_pos_outside and
024789:             recent_descending and
024790:             previous_outward_motion and
024791:             outward_motion and
024792:             prev_dy >= 14.0 and
024793:             -12.0 < curr_dy <= -3.0 and
024794:             prev_speed >= 18.0 and
024795:             18.0 <= curr_speed <= max_physical_out_bounce_step
024796:         )
024797:         soft_out_reversal = (
024798:             (
024799:                 (
024800:                     curr_pos_outside and
024801:                     outside_depth >= max(20.0, sideline_margin * 1.5)
024802:                 ) or
024803:                 soft_out_near_sideline
024804:             ) and
024805:             (soft_turn_motion or shallow_outward_rebound) and
024806:             outward_motion and
024807:             curr_speed >= 18.0
024808:         )
024809:         if soft_out_reversal:
024810:             frame_height = frame.shape[0] if frame is not None else 0
024811:             lower_court_y = max(
024812:                 int(frame_height * 0.62),
024813:                 int(getattr(self, 'net_area_y_min', 0)) + 520,
024814:             )
024815:             if y >= lower_court_y and current_ball_size <= 3.0:
024816:                 print(
024817:                     f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
024818:                     f"tiny lower-court soft sideline trace pos={ball_position} "
024819:                     f"size={current_ball_size:.1f}px lower_y={lower_court_y}"
024820:                 )
024821:                 return False, None
024822:         if (prev_speed < 20.0 or curr_speed < 12.0) and not soft_out_reversal:
024823:             return False, None
024824: 
024825:         if prev_pos[1] < near_side_min_y and y < near_side_min_y and not soft_out_reversal:
024826:             return False, None
024827: 
024828:         prev_dir = self.prev_motion.get('direction_deg')
024829:         curr_dir = self.last_motion.get('direction_deg')
024830:         angle_diff = 0.0
024831:         if prev_dir is not None and curr_dir is not None:
024832:             delta = abs(curr_dir - prev_dir) % 360
024833:             angle_diff = min(delta, 360 - delta)
024834: 
024835:         # A very large inward X reversal while already outside is a
024836:         # reacquisition/artifact jump, not a ground bounce.
024837:         inward_horizontal_reversal = (
024838:             (side == 'left' and prev_dx <= -8.0 and curr_dx >= 8.0) or
024839:             (side == 'right' and prev_dx >= 8.0 and curr_dx <= -8.0)
024840:         )
024841:         if (
024842:             prev_pos_outside and
024843:             inward_horizontal_reversal and
024844:             curr_speed > max_physical_out_bounce_step
024845:         ):
024846:             self._last_out_bounce_suppressed_frame = self.frame_count
024847:             self._last_out_bounce_suppressed_point = tuple(ball_position)
024848:             print(
024849:                 f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
024850:                 f"implausible inward jump pos={ball_position} side={side} "
024851:                 f"prev_motion=({prev_dx:.1f},{prev_dy:.1f}) "
024852:                 f"curr_motion=({curr_dx:.1f},{curr_dy:.1f}) "
024853:                 f"speed={curr_speed:.1f} limit={max_physical_out_bounce_step:.1f}"
024854:             )
```
