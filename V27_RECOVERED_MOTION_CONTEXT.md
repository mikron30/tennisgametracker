# V27 recovered-motion context

## def _update_recovered_motion line 12126
```python
012101:         self._back_return_origin_frame = self.frame_count
012102:         self._back_return_timed_out = False
012103: 
012104:     def _back_return_wait_active(self):
012105:         anchor = getattr(self, '_back_return_anchor', None)
012106:         return (
012107:             getattr(self, '_back_return_wait_frames', 0) > 0 and
012108:             anchor is not None and
012109:             self.ball_center is not None
012110:         )
012111: 
012112:     def _recent_offscreen_return_hold_active(self, window_frames=8):
012113:         last_frame = int(getattr(self, '_recent_offscreen_return_frame', -1000000))
012114:         return (self.frame_count - last_frame) <= max(1, int(window_frames))
012115: 
012116:     def _recent_return_bounce_recover_active(self, window_frames=2):
012117:         last_frame = int(getattr(self, '_recent_return_bounce_recover_frame', -1000000))
012118:         return (self.frame_count - last_frame) <= max(1, int(window_frames))
012119: 
012120:     def _late_contact_prior_bounce_active(self):
012121:         return (
012122:             int(getattr(self, '_late_contact_prior_bounce_count', 0)) > 0 and
012123:             self.frame_count <= int(getattr(self, '_late_contact_prior_bounce_until_frame', -1000000))
012124:         )
012125: 
012126:     def _update_recovered_motion(self, prev_pos, new_pos):
012127:         if prev_pos is None or new_pos is None:
012128:             return
012129:         new_dx = int(new_pos[0] - prev_pos[0])
012130:         new_dy = int(new_pos[1] - prev_pos[1])
012131:         new_velocity = math.hypot(new_dx, new_dy)
012132:         new_direction = math.degrees(math.atan2(new_dy, new_dx)) if new_velocity > 0 else 0.0
012133:         self.last_delta = (new_dx, new_dy)
012134:         self.last_motion = {
012135:             'distance': new_velocity,
012136:             'dx': new_dx,
012137:             'dy': new_dy,
012138:             'direction_deg': new_direction,
012139:         }
012140:         vel_hist = list(getattr(self, 'ball_velocity_history', []))
012141:         vel_hist.append(new_velocity)
012142:         self.ball_velocity_history = vel_hist[-5:]
012143:         self.last_direction = new_direction
012144: 
012145:     def _back_return_reentry_ok(self, pos, area, motion_mean, motion_max, frame_shape):
012146:         """Validate a delayed re-entry from the right/back side after a lower-right exit."""
012147:         anchor = getattr(self, '_back_return_anchor', None)
012148:         if anchor is None:
012149:             return True, None
012150: 
012151:         cx, cy = pos
012152:         frame_height, frame_width = frame_shape[:2]
012153:         elapsed = max(0, self.frame_count - getattr(self, '_back_return_origin_frame', self.frame_count))
012154:         x_floor = max(int(frame_width * 0.60), anchor[0] - 1000)
012155:         x_ceil = min(frame_width - 180, anchor[0] + 120)
012156:         min_reentry_y = max(int(frame_height * 0.12), anchor[1] - 1450)
012157:         max_reentry_y = min(int(frame_height * 0.92), anchor[1] - 80)
012158:         strong_motion = motion_max >= 35.0 or motion_mean >= 6.0 or area >= 8.0
012159: 
012160:         early_bottom_return = (
012161:             elapsed >= 4 and
012162:             abs(cx - anchor[0]) <= 180 and
012163:             cy >= anchor[1] - 340 and
012164:             cy <= frame_height - 2 and
012165:             (motion_max >= 25.0 or motion_mean >= 6.0 or area >= 15.0)
012166:         )
012167:         if early_bottom_return:
012168:             return True, None
012169: 
012170:         if elapsed < 4:
012171:             return False, f"back-return blind wait elapsed={elapsed}f"
012172:         if cx < x_floor:
012173:             return False, f"back-return x {cx} < min_x {x_floor}"
012174:         if cx > x_ceil:
012175:             return False, f"back-return x {cx} > max_x {x_ceil}"
012176:         if cy < min_reentry_y:
012177:             return False, f"back-return y {cy} < min_y {min_reentry_y}"
012178:         if cy > max_reentry_y:
012179:             return False, f"back-return y {cy} > max_y {max_reentry_y}"
012180:         if not strong_motion:
012181:             return False, f"back-return weak motion mean={motion_mean:.1f} max={motion_max:.1f} area={area:.1f}"
012182:         return True, None
012183: 
012184:     def _get_lower_contact_launch_context(self, frame_shape):
012185:         """Predict a launch window after a lower-court racket hit using ball-only motion."""
012186:         if self.ball_center is None or self.last_motion is None or self.ball_size is None:
012187:             return None
012188: 
012189:         frame_height, frame_width = frame_shape[:2]
012190:         origin_x, origin_y = self.ball_center
012191:         incoming_dx = float(self.last_motion.get('dx', 0.0))
012192:         incoming_dy = float(self.last_motion.get('dy', 0.0))
012193:         incoming_dist = float(self.last_motion.get('distance', 0.0))
012194: 
012195:         if self.ball_size < 300:
012196:             return None
012197:         if origin_y < int(frame_height * 0.58):
012198:             return None
012199:         if incoming_dy < max(50.0, frame_height * 0.023):
012200:             return None
012201:         if incoming_dist < 60.0:
012202:             return None
012203: 
012204:         lateral_scale = 2.5
012205:         vertical_scale = 2.4
012206:         raw_expected_dx = -incoming_dx * lateral_scale
012207:         raw_expected_up = abs(incoming_dy) * vertical_scale
012208: 
012209:         min_lateral = max(70.0, frame_width * 0.018)
012210:         max_lateral = max(220.0, frame_width * 0.065)
012211:         if abs(raw_expected_dx) < min_lateral:
012212:             raw_expected_dx = math.copysign(min_lateral, -incoming_dx if abs(incoming_dx) > 1 else 1.0)
012213:         expected_dx = max(-max_lateral, min(max_lateral, raw_expected_dx))
012214: 
012215:         min_up = max(95.0, frame_height * 0.042)
012216:         max_up = max(260.0, frame_height * 0.13)
012217:         expected_up = max(min_up, min(max_up, raw_expected_up))
012218: 
012219:         expected_x = int(max(0, min(frame_width - 1, origin_x + expected_dx)))
012220:         expected_y = int(max(0, min(frame_height - 1, origin_y - expected_up)))
012221: 
012222:         return {
012223:             'origin': (origin_x, origin_y),
012224:             'expected': (expected_x, expected_y),
012225:             # The first visible post-contact frame can still be close to the
012226:             # incoming ball (especially when the racket is near the camera).
012227:             # Requiring 1.6x the incoming displacement discarded real launches
012228:             # that moved upward only ~100 px in one frame.
012229:             'min_launch_dist': max(90.0, incoming_dist * 1.15),
012230:             'max_launch_dist': max(360.0, incoming_dist * 4.5),
012231:             'min_upward': max(85.0, incoming_dy * 0.95),
012232:             'ref_size': max(35.0, min(self.ball_size * 0.30, 130.0)),
012233:         }
012234: 
012235:     def _large_lower_launch_candidate_ok(self, pos, area, frame_shape):
012236:         """Recognize the close-ball launch that follows a one-frame lower-court hold."""
012237:         if self.ball_center is None or self.last_motion is None or self.ball_size is None:
012238:             return False
012239:         if self.stuck_frame_count < 1:
012240:             return False
012241: 
012242:         frame_height, frame_width = frame_shape[:2]
012243:         prev_size = float(self.ball_size)
012244:         origin_x, origin_y = self.ball_center
012245:         if not (145.0 <= prev_size <= 260.0):
```

## def _detect_out_of_court_bounce line 24367
```python
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
024458:                     try:
024459:                         tracked_hsv = getattr(self, 'ball_hsv', None)
024460:                         current_rebound_size = float(getattr(self, 'ball_size', 0.0) or 0.0)
024461:                         if tracked_hsv is not None and len(tracked_hsv) >= 3:
024462:                             rebound_h = int(tracked_hsv[0])
024463:                             rebound_s = int(tracked_hsv[1])
024464:                             rebound_v = int(tracked_hsv[2])
024465:                         else:
024466:                             rebound_x = int(round(float(ball_position[0])))
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
```

