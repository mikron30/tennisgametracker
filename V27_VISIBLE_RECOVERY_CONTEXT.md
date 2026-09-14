# V27 visible-ball recovery source context

Generated from the candidate branch to make the f4287 reanchor safe.

## Lines 7065-7285
```python
007065:                 # static regular-HSV court artifacts cannot use this escape.
007066:                 line_overlap_continuation = (
007067:                     source in ('h_10', 's_30') and
007068:                     area <= 80.0 and
007069:                     prediction_error <= min(60.0, max_prediction_error * 0.60)
007070:                 )
007071:                 # The ball is often momentarily measured against its own
007072:                 # previous blurred image immediately after the lower-player
007073:                 # contact, so frame difference can be near zero even though
007074:                 # the regular mask has the real, sizeable outgoing blob.  In
007075:                 # the first night rally these are a compact sequence:
007076:                 # a prior frame (692,1173), a prior frame (759,1043), a prior frame (827,925), and
007077:                 # a prior frame (891,819).  The former line-overlap escape admitted a
007078:                 # tiny alternate-HSV contour instead and the tracker lagged
007079:                 # behind the visible ball.  Admit the large *regular* blob
007080:                 # only when it is very close to the physical post-contact
007081:                 # prediction; the four-frame grace window and shape/player
007082:                 # gates above keep this from becoming a general static-artifact
007083:                 # exception.
007084:                 tight_regular_contact_continuation = (
007085:                     source == 'regular' and
007086:                     area >= 100.0 and
007087:                     prediction_error <= max_prediction_error
007088:                 )
007089:                 if (
007090:                         motion is not None and motion_mean < 4.0 and
007091:                         motion_max < 22.0 and not (
007092:                             line_overlap_continuation or
007093:                             tight_regular_contact_continuation
007094:                         )
007095:                 ):
007096:                     continue
007097:                 # The first decoded frame after a racket hit can be delayed
007098:                 # relative to the last compact launch blob, so a sizeable
007099:                 # regular candidate may be 80-110 px past a simple
007100:                 # constant-velocity prediction.  Its size/shape is stronger
007101:                 # evidence than a tiny line fragment which happens to be
007102:                 # nearer to that stale prediction.  Apply this preference
007103:                 # only to the explicit four-frame post-contact continuation.
007104:                 continuation_score = prediction_error + (
007105:                     0.0 if source == 'regular' else 7.0
007106:                 )
007107:                 if tight_regular_contact_continuation:
007108:                     continuation_score -= min(90.0, area * 0.20)
007109:                 candidates.append((
007110:                     continuation_score,
007111:                     -motion_max,
007112:                     -area,
007113:                     {
007114:                         'pos': (cx, cy), 'area': area, 'hsv': hsv_frame[cy, cx],
007115:                         'motion_mean': motion_mean, 'motion_max': motion_max,
007116:                         'source': source,
007117:                         'recovery_label': 'NIGHT LOWER CONTACT CONTINUATION',
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
007189:             self._local_ai_recovery_count = int(
007190:                 getattr(self, '_local_ai_recovery_count', 0)
007191:             ) + 1
007192:             # Suppress broad local-AI replay and endpoint inference only for
007193:             # this already-verified frame.  The dedicated one-frame ROI state
007194:             # independently handles the adjacent confirmation frame.
007195:             self._local_ai_follow_until_frame = max(
007196:                 int(getattr(self, '_local_ai_follow_until_frame', -1)),
007197:                 int(self.frame_count),
007198:             )
007199:             self._pending_rally_end_reason = None
007200:             self._pending_rally_end_frame = -1
007201:         self._activate_regular_hsv()
007202:         if candidate.get('lower_contact_approach', False):
007203:             # Keep a tightly bounded launch prediction for the frame after
007204:             # the incoming ball overlaps the player's racket.
007205:             launch_context = self._get_lower_contact_launch_context(frame.shape)
007206:             if launch_context is not None:
007207:                 self._pending_lower_contact_launch_context = launch_context
007208:                 self._pending_lower_contact_launch_until_frame = self.frame_count + 3
007209:         elif candidate.get('recovery_label') == 'NIGHT LOWER CONTACT LAUNCH':
007210:             # The launch is now verified; allow its fast upward continuation
007211:             # through the normal tracker without retaining the contact window.
007212:             self._pending_lower_contact_launch_context = None
007213:             self._pending_lower_contact_launch_until_frame = -1000000
007214:             self._rally_contact_grace_frames = max(
007215:                 getattr(self, '_rally_contact_grace_frames', 0), 4
007216:             )
007217:             self._rally_contact_origin = prev_pos
007218:             self._rally_contact_expected = new_pos
007219:         if frame is not None:
007220:             self._prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
007221:         recovery_label = candidate.get('recovery_label', 'NIGHT VISIBLE BALL RECOVER')
007222:         print(
007223:             f"Frame {self.frame_count}: [{recovery_label}] Ball at {new_pos} "
007224:             f"area={candidate['area']:.1f}px motion="
007225:             f"{candidate.get('motion_mean', 0.0):.1f}/{candidate.get('motion_max', 0.0):.1f}"
007226:         )
007227:         return self.ball_center
007228: 
007229:     def _player_reacq_static_candidate(self, candidate, predicted_point=None):
007230:         """Return True for a static blob that is unsafe just after player reacq.
007231: 
007232:         A real ball immediately following an occlusion should have either local
007233:         frame motion or a position consistent with the predicted flight path.
007234:         Static yellow/green patches (often racket or court artifacts) should not
007235:         be allowed to become the new ball center merely because their area looks
007236:         like the ball.
007237:         """
007238:         if self.frame_count > int(getattr(self, '_player_reacq_protect_until_frame', -1)):
007239:             return False
007240:         if not candidate:
007241:             return False
007242:         motion_mean = float(candidate.get('motion_mean', 0.0) or 0.0)
007243:         motion_max = float(candidate.get('motion_max', 0.0) or 0.0)
007244:         if motion_mean >= 6.0 or motion_max >= 25.0:
007245:             return False
007246:         pos = candidate.get('pos')
007247:         if pos is None or self.ball_center is None:
007248:             return False
007249:         prev_distance = math.hypot(pos[0] - self.ball_center[0], pos[1] - self.ball_center[1])
007250:         predicted_distance = (
007251:             math.hypot(pos[0] - predicted_point[0], pos[1] - predicted_point[1])
007252:             if predicted_point is not None else prev_distance
007253:         )
007254:         # Keep a static candidate only when it is essentially on the predicted
007255:         # path; a distant static blob is an artifact even if its area is large.
007256:         return prev_distance > 100.0 and predicted_distance > 90.0
007257: 
007258:     def _night_stuck_player_artifact_candidate(
007259:             self, position, motion_mean, motion_max, distance):
007260:         """Reject a static player/body contour during a stuck re-acquisition.
007261: 
007262:         Candidate scoring already penalizes player regions, but if every real
007263:         ball contour is temporarily occluded the penalized body contour can
007264:         still be the only candidate and therefore win.  A genuine ball that
007265:         reappears after contact has local frame motion; the f97 neck/body
007266:         takeover has none and is more than 500 px from the last ball.
007267:         """
007268:         if not self._is_night_session_config() or position is None:
007269:             return False
007270:         zone = self._player_point_zone(position)
007271:         if zone not in (
007272:                 'player_head_hat', 'player_body', 'player_shoes', 'racket_fragment'):
007273:             return False
007274:         return (
007275:             float(distance or 0.0) > 120.0 and
007276:             float(motion_mean or 0.0) < 6.0 and
007277:             float(motion_max or 0.0) < 25.0
007278:         )
007279: 
007280:     def _collect_override_candidate_metrics(self, pos, area, prev_pos, predicted_point, frame_gray):
007281:         motion_metrics = self._candidate_motion_metrics(frame_gray, pos[0], pos[1])
007282:         motion_mean = motion_metrics['mean'] if motion_metrics is not None else 0.0
007283:         motion_max = motion_metrics['max'] if motion_metrics is not None else 0.0
007284:         prev_distance = math.hypot(pos[0] - prev_pos[0], pos[1] - prev_pos[1]) if prev_pos is not None else 0.0
007285:         predicted_distance = (
```

## Lines 22976-23111
```python
022976:         recent_ground_bounce = frames_since_ground_bounce <= 2
022977:         
022978:         # Early serve grace: do not end point in first frames after start_frame
022979:         if self.frame_count <= (self.start_frame + 15):
022980:             return False, "Early-serve grace"
022981:         # If we're waiting near an edge, don't end the point
022982:         if getattr(self, 'edge_wait', False):
022983:             return False, "Edge wait"
022984:         reference_override = self._reference_point_end_override()
022985:         if reference_override is not None:
022986:             return True, reference_override['reason']
022987:         # A reviewed endpoint means that every earlier terminal heuristic is
022988:         # known to be premature for this specific rally.  This must be here,
022989:         # before the geometric/net/bounce checks below, rather than only in
022990:         # the timeout path; otherwise a false "net" or "out" event can end
022991:         # the point before the reviewed frame is reached.
022992:         continuation_frame = self._reviewed_point_continuation_frame()
022993:         if continuation_frame is not None and self.frame_count < continuation_frame:
022994:             return False, "Reviewed continuation hold"
022995:         reference_target = self._reference_point_end_target()
022996:         if (reference_target is not None and
022997:                 self.frame_count < int(reference_target['point_end_frame'])):
022998:             return False, "Reviewed endpoint hold"
022999:         top_far_out, top_far_reason = self._top_far_baseline_fall_out_candidate(ball_position, frame)
023000:         if top_far_out:
023001:             return True, top_far_reason
023002:         if (self._back_return_wait_active() or
023003:                 getattr(self, '_back_return_reentry_grace_frames', 0) > 0 or
023004:                 self._recent_offscreen_return_hold_active(window_frames=24)):
023005:             return False, "Back-return wait"
023006:         if getattr(self, 'ground_bounce_count', 0) >= 2:
023007:             return True, "Ball bounced twice on court"
023008:         
023009:         # Check if ball is out of court bounds
023010:         if x < 0 or x > width or y < 0 or y > height:
023011:             return True, "Ball out of court bounds"
023012: 
023013:         if self._upper_fence_fall_end_candidate(ball_position, frame.shape):
023014:             return True, "Ball hit upper fence and fell down"
023015: 
023016:         suppress_out_bounce = (
023017:             self._back_return_wait_active() or
023018:             getattr(self, '_back_return_reentry_grace_frames', 0) > 0
023019:         )
023020:         if not suppress_out_bounce:
023021:             out_bounce_detected, out_bounce_reason = self._detect_out_of_court_bounce(ball_position, frame)
023022:             if out_bounce_detected:
023023:                 from out_bounce_verification import recover_continuing_ball
023024:                 if recover_continuing_ball(self, ball_position):
023025:                     return False, "Out candidate replaced by continuing ball"
023026:                 return True, out_bounce_reason
023027:         
023028:         # Check if ball is in or just above the marked net area.
023029:         # A little extra top margin helps catch real tape clips when the marked
023030:         # net band sits slightly low in the frame.
023031:         net_x_limit = int(width * 0.60)
023032:         net_y_top_margin = 220
023033:         net_y_bottom_margin = 80
023034:         if hasattr(self, 'net_area_y_min') and hasattr(self, 'net_area_y_max'):
023035:             expanded_net_zone = (
023036:                 (self.net_area_y_min - net_y_top_margin) <= y <= (self.net_area_y_max + net_y_bottom_margin)
023037:                 and x < net_x_limit
023038:             )
023039:             core_net_zone = (
023040:                 max(0, self.net_area_y_min - 20) <= y <= (self.net_area_y_max + 40)
023041:                 and x < net_x_limit
023042:             )
023043:             if expanded_net_zone and not recent_ground_bounce:
023044:                 ball_is_fast = self.last_motion and self.last_motion['distance'] > 15
023045:                 # Only count "lingering" inside the tighter core net band.
023046:                 # A normal descending ball can legitimately pass through the larger
023047:                 # expanded zone for many frames without touching the net.
023048:                 if core_net_zone and not ball_is_fast:
023049:                     if not hasattr(self, 'net_area_frames'):
023050:                         self.net_area_frames = 0
023051:                     self.net_area_frames += 1
023052:                 else:
023053:                     self.net_area_frames = 0
023054:                 # Consider it a net hit if it lingers - use longer threshold for fast balls
023055:                 net_linger_limit = 8 if ball_is_fast else 6
023056:                 if getattr(self, 'net_area_frames', 0) > net_linger_limit:
023057:                     return True, "Ball hit the net"
023058:                 # Immediate stop near net
023059:                 if self.last_motion and self.last_motion['distance'] < 2.5:
023060:                     # Only if previous frame was also slow (not a tracking glitch)
023061:                     prev_was_slow = self.prev_motion and self.prev_motion['distance'] < 10
023062:                     if prev_was_slow:
023063:                         return True, "Ball hit the net"
023064:                 # Direction change with deceleration — classic net hit signature
023065:                 if self.prev_motion and self.last_motion:
023066:                     prev_dir = self.prev_motion.get('direction_deg')
023067:                     curr_dir = self.last_motion.get('direction_deg')
023068:                     prev_dist = self.prev_motion.get('distance', 0)
023069:                     curr_dist = self.last_motion.get('distance', 0)
023070:                     prev_dx = self.prev_motion.get('dx', 0)
023071:                     prev_dy = self.prev_motion.get('dy', 0)
023072:                     curr_dx = self.last_motion.get('dx', 0)
023073:                     curr_dy = self.last_motion.get('dy', 0)
023074:                     if prev_dir is not None and curr_dir is not None and prev_dist > 0:
023075:                         delta = abs(curr_dir - prev_dir) % 360
023076:                         angle_diff = min(delta, 360 - delta)
023077:                         speed_ratio = curr_dist / prev_dist if prev_dist else 1.0
023078:                         reversed_y = (
023079:                             prev_dy * curr_dy < 0
023080:                             and abs(prev_dy) >= 3
023081:                             and abs(curr_dy) >= 3
023082:                         )
023083:                         strong_x_reversal = (
023084:                             prev_dx * curr_dx < 0
023085:                             and abs(prev_dx) >= 6
023086:                             and abs(curr_dx) >= 6
023087:                         )
023088:                         direction_reversed = reversed_y or strong_x_reversal
023089:                         slowed_after_impact = curr_dist <= max(8.0, prev_dist * 0.80)
023090:                         if direction_reversed and slowed_after_impact:
023091:                             return True, "Ball hit the net"
023092:                         if angle_diff >= 60 and slowed_after_impact:
023093:                             return True, "Ball hit the net"
023094:                         if angle_diff >= 90 and speed_ratio < 0.6:
023095:                             if prev_dist < 30:
023096:                                 return True, "Ball hit the net"
023097:             else:
023098:                 # Reset counter if ball is not in net area
023099:                 if hasattr(self, 'net_area_frames'):
023100:                     self.net_area_frames = 0
023101:         
023102:         # Check if ball has been stationary for too long (double bounce)
023103:         if hasattr(self, 'ball_velocity_history') and len(self.ball_velocity_history) > 10:
023104:             recent_velocities = self.ball_velocity_history[-10:]
023105:             avg_velocity = sum(recent_velocities) / len(recent_velocities)
023106:             print(f"[BALL_STOPPED_CHECK] f{self.frame_count}: avg_vel={avg_velocity:.1f} hist={[round(v,1) for v in recent_velocities]}")
023107:             if avg_velocity < 5:  # Very slow movement
023108:                 print(f"[BALL_STOPPED] f{self.frame_count}: avg_vel={avg_velocity:.1f} < 5 → POINT ENDS")
023109:                 return True, "Ball stopped (possible double bounce)"
023110:         
023111:         # Check if ball is near court edges (likely out)
```

## Lines 23197-23332
```python
023197:             not self._serve_net_pass_active(window_frames=36) and
023198:             any(token in str(pending_reason).lower() for token in ('out', 'bounce', 'stuck', 'lost'))
023199:         )
023200:         if pending_static_artifact:
023201:             print(
023202:                 f"Frame {self.frame_count}: [STALE-END CLEARED] "
023203:                 f"discarding queued '{pending_reason}' at static marker {ball_position}"
023204:             )
023205:             self._pending_rally_end_reason = None
023206:             self._pending_rally_end_frame = -1
023207:         reference_override = self._reference_point_end_override()
023208:         if reference_override is not None:
023209:             return True, reference_override['reason']
023210:         # This is the active point-end implementation.  A reviewed endpoint
023211:         # suppresses every earlier geometric/timeout terminal heuristic, not
023212:         # only a queued timeout, because an earlier apparent net/out can be a
023213:         # racket-contact or player-occlusion artifact.
023214:         continuation_frame = self._reviewed_point_continuation_frame()
023215:         if continuation_frame is not None and self.frame_count < continuation_frame:
023216:             return False, "Reviewed continuation hold"
023217:         reference_target = self._reference_point_end_target()
023218:         if (reference_target is not None and
023219:                 self.frame_count < int(reference_target['point_end_frame'])):
023220:             return False, "Reviewed endpoint hold"
023221:         top_far_out, top_far_reason = self._top_far_baseline_fall_out_candidate(ball_position, frame)
023222:         if top_far_out:
023223:             return True, top_far_reason
023224:         if (self._back_return_wait_active() or
023225:                 getattr(self, '_back_return_reentry_grace_frames', 0) > 0 or
023226:                 self._recent_offscreen_return_hold_active(window_frames=24)):
023227:             return False, "Back-return wait"
023228:         if getattr(self, 'ground_bounce_count', 0) >= 2:
023229:             return True, "Ball bounced twice on court"
023230: 
023231:         if x < 0 or x > width or y < 0 or y > height:
023232:             return True, "Ball out of court bounds"
023233: 
023234:         if self._upper_fence_fall_end_candidate(ball_position, frame.shape):
023235:             return True, "Ball hit upper fence and fell down"
023236: 
023237:         suppress_out_bounce = (
023238:             self._back_return_wait_active() or
023239:             getattr(self, '_back_return_reentry_grace_frames', 0) > 0
023240:         )
023241:         if not suppress_out_bounce:
023242:             out_bounce_detected, out_bounce_reason = self._detect_out_of_court_bounce(ball_position, frame)
023243:             if out_bounce_detected:
023244:                 from out_bounce_verification import recover_continuing_ball
023245:                 if recover_continuing_ball(self, ball_position):
023246:                     return False, "Out candidate replaced by continuing ball"
023247:                 return True, out_bounce_reason
023248: 
023249:         if self._mark_serve_net_contact_candidate(ball_position, frame):
023250:             return False, "Serve net touch, waiting for service bounce"
023251: 
023252:         same_side_bounce, same_side_reason = self._same_side_pre_net_bounce_candidate(ball_position, frame)
023253:         if int(getattr(self, "_player_occlusion_hold_frames", 0)) > 0:
023254:             if same_side_bounce:
023255:                 print(
023256:                     f"Frame {self.frame_count}: [SAME-SIDE PRE-NET SUPPRESSED] "
023257:                     f"player-occlusion grace active ({self._player_occlusion_hold_frames} frames)"
023258:                 )
023259:             same_side_bounce, same_side_reason = False, None
023260:         if same_side_bounce:
023261:             return True, same_side_reason
023262: 
023263:         recent_serve_bounce_in = (
023264:             getattr(self, '_last_serve_bounce_frame', -1000000) > -1000000 and
023265:             0 <= self.frame_count - getattr(self, '_last_serve_bounce_frame', -1000000) <= 10 and
023266:             not getattr(self, '_awaiting_serve_bounce', False)
023267:         )
023268:         net_geometry = self._net_contact_geometry(ball_position)
023269:         far_side_serve_attempt = getattr(self, '_active_serve_area_end', None) == "far"
023270:         if (far_side_serve_attempt and recent_serve_bounce_in and
023271:                 getattr(self, '_last_serve_bounce_net_contact_like', False)):
023272:             net_zone_active = (
023273:                 net_geometry is not None and
023274:                 (net_geometry['inside'] or net_geometry['near_polygon'] or net_geometry['near_top_tape'])
023275:             )
023276:             current_speed = float(self.last_motion.get('distance', 0.0)) if self.last_motion else 0.0
023277:             if net_zone_active and current_speed <= 14.0:
023278:                 return True, "Serve let"
023279: 
023280:         if net_geometry is not None and recent_ground_bounce and not recent_serve_bounce_in:
023281:             bounce_origin = getattr(self, '_ground_bounce_origin', None)
023282:             if bounce_origin is not None and 1 <= frames_since_ground_bounce <= 2:
023283:                 origin_x, origin_y = bounce_origin
023284:                 current_dy = float(self.last_motion.get('dy', 0.0) or 0.0) if self.last_motion else 0.0
023285:                 current_speed = float(self.last_motion.get('distance', 0.0) or 0.0) if self.last_motion else 0.0
023286:                 net_y_max = float(getattr(self, 'net_area_y_max', 0.0))
023287:                 low_net_y_min = float(getattr(self, 'low_net_y_min', getattr(self, 'net_area_y_min', 0)))
023288:                 low_net_y_max = float(getattr(self, 'low_net_y_max', getattr(self, 'net_area_y_max', 0)))
023289:                 origin_below_net = (
023290:                     net_y_max + 10.0 <= float(origin_y) <= net_y_max + max(65.0, height * 0.030)
023291:                 )
023292:                 climbing_into_low_net = (
023293:                     current_dy <= -12.0 and
023294:                     current_speed <= max(70.0, height * 0.035) and
023295:                     float(y) <= float(origin_y) - 45.0 and
023296:                     low_net_y_min - 8.0 <= float(y) <= low_net_y_max + 8.0
023297:                 )
023298:                 near_net_surface = (
023299:                     net_geometry['inside'] or
023300:                     net_geometry['near_top_tape'] or
023301:                     net_geometry['signed_dist'] >= -42.0
023302:                 )
023303:                 close_to_bounce_lane = (
023304:                     abs(float(x) - float(origin_x)) <= 90.0 and
023305:                     math.hypot(float(x) - float(origin_x), float(y) - float(origin_y)) <= 115.0
023306:                 )
023307:                 if origin_below_net and climbing_into_low_net and near_net_surface and close_to_bounce_lane:
023308:                     print(
023309:                         f"Frame {self.frame_count}: [POST-BOUNCE NET] "
023310:                         f"origin={bounce_origin} pos={ball_position} "
023311:                         f"signed_dist={net_geometry['signed_dist']:.1f}"
023312:                     )
023313:                     return True, "Ball hit the net"
023314: 
023315:         if net_geometry is not None and not recent_ground_bounce and not recent_serve_bounce_in:
023316:             net_zone_active = net_geometry['near_polygon'] or net_geometry['near_top_tape']
023317:             core_net_zone = net_geometry['inside']
023318:             if net_zone_active:
023319:                 ball_is_fast = self.last_motion and self.last_motion['distance'] > 15
023320:                 if core_net_zone and not ball_is_fast:
023321:                     if not hasattr(self, 'net_area_frames'):
023322:                         self.net_area_frames = 0
023323:                     self.net_area_frames += 1
023324:                 else:
023325:                     self.net_area_frames = 0
023326: 
023327:                 net_linger_limit = 8 if ball_is_fast else 6
023328:                 if getattr(self, 'net_area_frames', 0) > net_linger_limit:
023329:                     return True, "Ball hit the net"
023330: 
023331:                 if self.last_motion and self.last_motion['distance'] < 2.5:
023332:                     prev_was_slow = self.prev_motion and self.prev_motion['distance'] < 10
```

## Lines 24322-24457
```python
024322:             ),
024323:             'soft_vertical_reversal': soft_vertical_reversal,
024324:             'sharp_turn': sharp_turn,
024325:             'shallow_perspective_bounce': shallow_perspective_bounce,
024326:             'shallow_serve_turn': shallow_serve_turn,
024327:         }
024328: 
024329:     def _commit_serve_bounce_in_event(self, event, frame):
024330:         bounce_point = event['point']
024331:         seed_ground_bounce = not bool(event.get('shallow_perspective_bounce'))
024332:         if seed_ground_bounce:
024333:             self.ground_bounce_count += 1
024334:             self.last_ground_bounce_frame = self.frame_count
024335:         self._last_serve_bounce_frame = self.frame_count
024336:         self._last_serve_bounce_point = bounce_point
024337:         self._last_serve_bounce_net_contact_like = bool(event.get('net_contact_like'))
024338:         self._last_serve_bounce_was_in = True
024339:         if seed_ground_bounce:
024340:             print(
024341:                 f"Frame {self.frame_count}: Ground bounce #{self.ground_bounce_count} "
024342:                 f"detected (serve box jump)"
024343:             )
024344:         else:
024345:             print(
024346:                 f"Frame {self.frame_count}: Serve bounce detected "
024347:                 f"(shallow perspective, rally count deferred)"
024348:             )
024349:         self._handle_ground_bounce_event(
024350:             bounce_point,
024351:             frame,
024352:             accepted_in_reason="Serve bounce in",
024353:         )
024354:         # A committed in-service bounce is the serve->rally boundary.  The
024355:         # generic handler normally closes it; force the state closed here as
024356:         # well so a specialized bounce path can never leak serve logic into
024357:         # the later rally.  A true net-touch serve remains a replayable let.
024358:         if not self._serve_net_touch_active(window_frames=120):
024359:             self._serve_phase_active = False
024360:             self._serve_phase_closed_frame = int(self.frame_count)
024361:             self._awaiting_serve_bounce = False
024362:         print(
024363:             f"Frame {self.frame_count}: Allowing serve-bounce-in continuation "
024364:             f"at {bounce_point}"
024365:         )
024366: 
024367:     def _detect_out_of_court_bounce(self, ball_position, frame):
024368:         if self.prev_motion is None or self.last_motion is None:
024369:             return False, None
024370:         if not hasattr(self, 'net_area_y_min'):
024371:             return False, None
024372:         if getattr(self, '_last_motion_reacq_frame', -1000000) == self.frame_count:
024373:             return False, None
024374:         # V14: a deep night candidate may look static exactly at the bounce.
024375:         # Keep that candidate briefly and confirm it only when a later, bounded
024376:         # opposite-direction track proves a real rebound. This runs before the
024377:         # legacy two-frame static-artifact wait so the original OUT location is
024378:         # not lost while the tracker holds the stale point for two frames.
024379:         pending_static_out = getattr(self, '_pending_night_static_out', None)
024380:         if isinstance(pending_static_out, dict):
024381:             try:
024382:                 pending_frame = int(pending_static_out.get('frame', -1000000))
024383:                 pending_pos = tuple(pending_static_out.get('position', ()))
024384:                 pending_dx = float(pending_static_out.get('dx', 0.0) or 0.0)
024385:                 pending_dy = float(pending_static_out.get('dy', 0.0) or 0.0)
024386:             except (TypeError, ValueError):
024387:                 pending_frame = -1000000
024388:                 pending_pos = ()
024389:                 pending_dx = pending_dy = 0.0
024390: 
024391:             pending_age = int(self.frame_count) - pending_frame
024392:             if pending_age < 0 or pending_age > 4 or len(pending_pos) < 2:
024393:                 self._pending_night_static_out = None
024394:             elif pending_age >= 3:
024395:                 current_dx = float(self.last_motion.get('dx', 0.0) or 0.0)
024396:                 current_dy = float(self.last_motion.get('dy', 0.0) or 0.0)
024397:                 current_speed = float(self.last_motion.get('distance', 0.0) or 0.0)
024398:                 pending_speed = math.hypot(pending_dx, pending_dy)
024399:                 displacement = math.hypot(
024400:                     float(ball_position[0]) - float(pending_pos[0]),
024401:                     float(ball_position[1]) - float(pending_pos[1]),
024402:                 )
024403:                 turn_angle = 0.0
024404:                 if pending_speed > 0.0 and current_speed > 0.0:
024405:                     cosine = (
024406:                         pending_dx * current_dx + pending_dy * current_dy
024407:                     ) / (pending_speed * current_speed)
024408:                     cosine = max(-1.0, min(1.0, cosine))
024409:                     turn_angle = math.degrees(math.acos(cosine))
024410: 
024411:                 width = int(frame.shape[1]) if frame is not None else 0
024412:                 max_rebound_step = max(180.0, float(width) * 0.055)
024413:                 max_rebound_displacement = max(260.0, float(width) * 0.08)
024414:                 coherent_rebound = (
024415:                     pending_dy >= 6.0 and
024416:                     current_dy <= -12.0 and
024417:                     turn_angle >= 120.0 and
024418:                     24.0 <= current_speed <= max_rebound_step and
024419:                     50.0 <= displacement <= max_rebound_displacement
024420:                 )
024421: 
024422:                 # V20: a suppressed night-side hotspot is only allowed to come
024423:                 # back as a confirmed OUT when the later rebound is independent
024424:                 # ball evidence.  In the reviewed f4284->f4287 failure, the
024425:                 # original point was correctly suppressed as a static side
024426:                 # artifact, but ALT4 then selected a tiny player-body fragment
024427:                 # (H=97, 10.5px).  Its 142px upward jump looked like a perfect
024428:                 # geometric rebound and incorrectly promoted the old hotspot to
024429:                 # an OUT.  Player/racket overlap is therefore not sufficient
024430:                 # evidence to resurrect a suppressed static endpoint.  Keep the
024431:                 # pending candidate alive for its final bounded frame so a real,
024432:                 # clear rebound can still confirm it.
024433:                 rebound_player_zone = self._player_point_zone(ball_position)
024434:                 rebound_overlaps_player = rebound_player_zone in (
024435:                     'player_head_hat',
024436:                     'player_shoes',
024437:                     'racket_fragment',
024438:                     'player_body',
024439:                 )
024440:                 if coherent_rebound and rebound_overlaps_player:
024441:                     print(
024442:                         f"Frame {self.frame_count}: [OUT-BOUNCE PENDING REBOUND REJECT] "
024443:                         f"source_f={pending_frame} point={pending_pos} "
024444:                         f"rebound={tuple(ball_position)} zone={rebound_player_zone} "
024445:                         f"speed={current_speed:.1f}px turn={turn_angle:.1f}deg "
024446:                         f"displacement={displacement:.1f}px"
024447:                     )
024448:                     coherent_rebound = False
024449: 
024450:                 # V26: the V20 player-zone guard can miss when player tracks are
024451:                 # stale/invisible.  Do not resurrect a suppressed static night
024452:                 # endpoint from a dark ALT4-like fragment when the existing
024453:                 # visible-ball recovery independently sees a clear ball elsewhere
024454:                 # in the same frame.  This is deliberately a two-signal guard:
024455:                 # a dark candidate alone remains legal, and a second visible
024456:                 # candidate alone does not veto a normal rebound.
024457:                 if coherent_rebound and frame is not None:
```

## Lines 24467-24602
```python
024467:                             rebound_y = int(round(float(ball_position[1])))
024468:                             frame_h, frame_w = frame.shape[:2]
024469:                             rebound_x = max(0, min(frame_w - 1, rebound_x))
024470:                             rebound_y = max(0, min(frame_h - 1, rebound_y))
024471:                             hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
024472:                             rebound_hsv = hsv_frame[rebound_y, rebound_x]
024473:                             rebound_h = int(rebound_hsv[0])
024474:                             rebound_s = int(rebound_hsv[1])
024475:                             rebound_v = int(rebound_hsv[2])
024476:                         dark_alt4_like = (
024477:                             current_rebound_size <= 30.0 and
024478:                             rebound_h >= 88 and
024479:                             rebound_s >= 85 and
024480:                             rebound_v <= 130
024481:                         )
024482:                     except Exception:
024483:                         rebound_h = rebound_s = rebound_v = -1
024484:                         current_rebound_size = 0.0
024485:                         dark_alt4_like = False
024486: 
024487:                     visible_rebound = None
024488:                     if dark_alt4_like:
024489:                         try:
024490:                             rebound_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
024491:                             visible_rebound = self._find_night_visible_ball_candidate(
024492:                                 frame, rebound_gray
024493:                             )
024494:                         except Exception:
024495:                             visible_rebound = None
024496: 
024497:                     if visible_rebound is not None:
024498:                         try:
024499:                             visible_pos = tuple(visible_rebound.get('pos', ()))
024500:                             visible_conflict_distance = math.hypot(
024501:                                 float(visible_pos[0]) - float(ball_position[0]),
024502:                                 float(visible_pos[1]) - float(ball_position[1]),
024503:                             ) if len(visible_pos) >= 2 else 0.0
024504:                         except (TypeError, ValueError, IndexError):
024505:                             visible_pos = ()
024506:                             visible_conflict_distance = 0.0
024507: 
024508:                         visible_conflict_limit = max(80.0, float(width) * 0.02)
024509:                         if visible_conflict_distance >= visible_conflict_limit:
024510:                             print(
024511:                                 f"Frame {self.frame_count}: "
024512:                                 f"[OUT-BOUNCE PENDING VISIBLE-CONFLICT REJECT] "
024513:                                 f"source_f={pending_frame} point={pending_pos} "
024514:                                 f"rebound={tuple(ball_position)} "
024515:                                 f"hsv=({rebound_h},{rebound_s},{rebound_v}) "
024516:                                 f"size={current_rebound_size:.1f}px "
024517:                                 f"visible={visible_pos} "
024518:                                 f"conflict={visible_conflict_distance:.1f}px"
024519:                             )
024520:                             coherent_rebound = False
024521: 
024522:                 if coherent_rebound:
024523:                     original_position = (
024524:                         int(round(float(pending_pos[0]))),
024525:                         int(round(float(pending_pos[1]))),
024526:                     )
024527:                     original_side = str(pending_static_out.get('side') or 'unknown')
024528:                     self._pending_night_static_out = None
024529:                     self.ball_center = original_position
024530:                     self._last_confirmed_pending_out_frame = pending_frame
024531:                     self._last_confirmed_pending_out_position = original_position
024532:                     confirmed_reason = f"Ball bounced out of court ({original_side} sideline)"
024533:                     self._last_confirmed_pending_out_confirm_frame = int(self.frame_count)
024534:                     self._last_confirmed_pending_out_reason = confirmed_reason
024535:                     print(
024536:                         f"Frame {self.frame_count}: [OUT-BOUNCE PENDING CONFIRMED] "
024537:                         f"source_f={pending_frame} point={original_position} side={original_side} "
024538:                         f"age={pending_age} rebound={tuple(ball_position)} "
024539:                         f"speed={current_speed:.1f}px turn={turn_angle:.1f}deg "
024540:                         f"displacement={displacement:.1f}px"
024541:                     )
024542:                     return True, confirmed_reason
024543:                 if pending_age >= 4:
024544:                     self._pending_night_static_out = None
024545: 
024546:         # A static side artifact can be followed by a valid airborne candidate
024547:         # one frame later. Do not use that artifact's stale outside position as
024548:         # the previous leg of a sideline reversal (point 9 a prior frame).
024549:         if 0 <= self.frame_count - int(getattr(self, '_last_out_bounce_suppressed_frame', -1000000)) <= 2:
024550:             print(
024551:                 f"Frame {self.frame_count}: [OUT-BOUNCE WAIT] "
024552:                 "previous frame was a suppressed static side artifact"
024553:             )
024554:             return False, None
024555:         if (self.frame_count - getattr(self, '_last_racket_contact_frame', -1000000)) <= 1:
024556:             return False, None
024557:         if (
024558:             getattr(self, '_awaiting_serve_bounce', False) and
024559:             self.point_start_frame_internal is not None and
024560:             (self.frame_count - self.point_start_frame_internal) <= 12 and
024561:             getattr(self, '_serve_contact_grace_frames', 0) > 0
024562:         ):
024563:             return False, None
024564: 
024565:         # A first-serve out-bounce must not be inferred from a near-player
024566:         # fragment before the ball has reached the receiver's half.  The
024567:         # generic sideline reversal path runs before the more specific serve
024568:         # turn handlers, so a stale racket/body mask can otherwise end the
024569:         # point while the server is still holding the ball (point 9, a prior frame).
024570:         # Net-touch/pass serves remain eligible for the existing net-fault
024571:         # handling because they are allowed to bounce on the hitter's side.
024572:         if (
024573:             getattr(self, '_awaiting_serve_bounce', False) and
024574:             not self._serve_net_touch_active(window_frames=120) and
024575:             not self._serve_net_pass_active(window_frames=36) and
024576:             int(getattr(self, 'serve_direction_dy', 0)) != 0 and
024577:             self.point_start_frame_internal is not None and
024578:             (self.frame_count - self.point_start_frame_internal) >= 12
024579:         ):
024580:             serve_model = self._build_service_box_model(frame)
024581:             if serve_model is not None:
024582:                 net_y = float(serve_model.get('net_y', (self.net_area_y_min + self.net_area_y_max) * 0.5))
024583:             elif hasattr(self, 'net_area_y_min') and hasattr(self, 'net_area_y_max'):
024584:                 net_y = float((self.net_area_y_min + self.net_area_y_max) * 0.5)
024585:             else:
024586:                 net_y = None
024587:             if net_y is not None:
024588:                 crossing_slack = max(60.0, frame.shape[0] * 0.035)
024589:                 still_on_server_half = (
024590:                     self.serve_direction_dy < 0 and float(ball_position[1]) > net_y + crossing_slack
024591:                 ) or (
024592:                     self.serve_direction_dy > 0 and float(ball_position[1]) < net_y - crossing_slack
024593:                 )
024594:                 if still_on_server_half:
024595:                     print(
024596:                         f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
024597:                         f"serve candidate {ball_position} is still on server half "
024598:                         f"(net_y={net_y:.1f}, slack={crossing_slack:.1f})"
024599:                     )
024600:                     return False, None
024601: 
024602:         # During the first serve flight, a stale HSV fragment near the net can
```

