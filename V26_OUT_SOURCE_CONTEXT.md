# V26 pending OUT source context

```python
024267:         )
024268:         same_target_half = False
024269:         center_x = geometry.get('center_x')
024270:         if center_x is None and hasattr(self, 'serve_area_x_min') and hasattr(self, 'serve_area_x_max'):
024271:             center_x = (self.serve_area_x_min + self.serve_area_x_max) / 2.0
024272:         if center_x is not None:
024273:             if target_side == "left":
024274:                 same_target_half = bounce_point[0] <= center_x
024275:             elif target_side == "right":
024276:                 same_target_half = bounce_point[0] >= center_x
024277:         net_y = geometry.get('net_y')
024278:         net_margin = geometry.get('net_margin', 0.0)
024279:         service_y = geometry.get('service_y')
024280:         shallow_service_box_slack = False
024281:         if (
024282:                 (shallow_perspective_bounce or shallow_serve_turn) and
024283:                 same_target_half and
024284:                 net_y is not None and
024285:                 service_y is not None):
024286:             line_slack = max(90.0, frame.shape[0] * 0.045)
024287:             if self.serve_direction_dy < 0:
024288:                 shallow_service_box_slack = (
024289:                     (service_y - line_slack) <= bounce_point[1] <= (net_y + net_margin)
024290:                 )
024291:             elif self.serve_direction_dy > 0:
024292:                 shallow_service_box_slack = (
024293:                     (net_y - net_margin) <= bounce_point[1] <= (service_y + line_slack)
024294:                 )
024295:         near_target_net_band = False
024296:         if net_y is not None and same_target_half:
024297:             extra_net_margin = max(60.0, frame.shape[0] * 0.035)
024298:             if self.serve_direction_dy < 0:
024299:                 near_target_net_band = (net_y - extra_net_margin) <= bounce_point[1] <= (net_y + net_margin)
024300:             elif self.serve_direction_dy > 0:
024301:                 near_target_net_band = (net_y - net_margin) <= bounce_point[1] <= (net_y + extra_net_margin)
024302:         if not near_target_net_band and same_target_half:
024303:             net_geometry = self._net_contact_geometry(bounce_point)
024304:             near_target_net_band = (
024305:                 net_geometry is not None and
024306:                 (net_geometry['inside'] or net_geometry['near_top_tape'])
024307:             )
024308:         else:
024309:             net_geometry = self._net_contact_geometry(bounce_point)
024310:         if service_ok is not True and not near_target_net_band and not shallow_service_box_slack:
024311:             return None
024312: 
024313:         return {
024314:             'point': bounce_point,
024315:             'target_side': target_side,
024316:             'geometry': geometry,
024317:             'near_target_net_band': near_target_net_band,
024318:             'shallow_service_box_slack': shallow_service_box_slack,
024319:             'net_contact_like': (
024320:                 net_geometry is not None and
024321:                 (net_geometry['inside'] or net_geometry['near_polygon'] or net_geometry['near_top_tape'])
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
024450:                 if coherent_rebound:
024451:                     original_position = (
024452:                         int(round(float(pending_pos[0]))),
024453:                         int(round(float(pending_pos[1]))),
024454:                     )
024455:                     original_side = str(pending_static_out.get('side') or 'unknown')
024456:                     self._pending_night_static_out = None
024457:                     self.ball_center = original_position
024458:                     self._last_confirmed_pending_out_frame = pending_frame
024459:                     self._last_confirmed_pending_out_position = original_position
024460:                     confirmed_reason = f"Ball bounced out of court ({original_side} sideline)"
024461:                     self._last_confirmed_pending_out_confirm_frame = int(self.frame_count)
024462:                     self._last_confirmed_pending_out_reason = confirmed_reason
024463:                     print(
024464:                         f"Frame {self.frame_count}: [OUT-BOUNCE PENDING CONFIRMED] "
024465:                         f"source_f={pending_frame} point={original_position} side={original_side} "
024466:                         f"age={pending_age} rebound={tuple(ball_position)} "
024467:                         f"speed={current_speed:.1f}px turn={turn_angle:.1f}deg "
024468:                         f"displacement={displacement:.1f}px"
024469:                     )
024470:                     return True, confirmed_reason
024471:                 if pending_age >= 4:
024472:                     self._pending_night_static_out = None
024473: 
024474:         # A static side artifact can be followed by a valid airborne candidate
024475:         # one frame later. Do not use that artifact's stale outside position as
024476:         # the previous leg of a sideline reversal (point 9 a prior frame).
024477:         if 0 <= self.frame_count - int(getattr(self, '_last_out_bounce_suppressed_frame', -1000000)) <= 2:
024478:             print(
024479:                 f"Frame {self.frame_count}: [OUT-BOUNCE WAIT] "
024480:                 "previous frame was a suppressed static side artifact"
024481:             )
024482:             return False, None
024483:         if (self.frame_count - getattr(self, '_last_racket_contact_frame', -1000000)) <= 1:
024484:             return False, None
024485:         if (
024486:             getattr(self, '_awaiting_serve_bounce', False) and
024487:             self.point_start_frame_internal is not None and
024488:             (self.frame_count - self.point_start_frame_internal) <= 12 and
024489:             getattr(self, '_serve_contact_grace_frames', 0) > 0
024490:         ):
024491:             return False, None
024492: 
024493:         # A first-serve out-bounce must not be inferred from a near-player
024494:         # fragment before the ball has reached the receiver's half.  The
024495:         # generic sideline reversal path runs before the more specific serve
024496:         # turn handlers, so a stale racket/body mask can otherwise end the
024497:         # point while the server is still holding the ball (point 9, a prior frame).
024498:         # Net-touch/pass serves remain eligible for the existing net-fault
024499:         # handling because they are allowed to bounce on the hitter's side.
024500:         if (
024501:             getattr(self, '_awaiting_serve_bounce', False) and
024502:             not self._serve_net_touch_active(window_frames=120) and
024503:             not self._serve_net_pass_active(window_frames=36) and
024504:             int(getattr(self, 'serve_direction_dy', 0)) != 0 and
024505:             self.point_start_frame_internal is not None and
024506:             (self.frame_count - self.point_start_frame_internal) >= 12
024507:         ):
024508:             serve_model = self._build_service_box_model(frame)
024509:             if serve_model is not None:
024510:                 net_y = float(serve_model.get('net_y', (self.net_area_y_min + self.net_area_y_max) * 0.5))
024511:             elif hasattr(self, 'net_area_y_min') and hasattr(self, 'net_area_y_max'):
024512:                 net_y = float((self.net_area_y_min + self.net_area_y_max) * 0.5)
024513:             else:
024514:                 net_y = None
024515:             if net_y is not None:
024516:                 crossing_slack = max(60.0, frame.shape[0] * 0.035)
024517:                 still_on_server_half = (
024518:                     self.serve_direction_dy < 0 and float(ball_position[1]) > net_y + crossing_slack
024519:                 ) or (
024520:                     self.serve_direction_dy > 0 and float(ball_position[1]) < net_y - crossing_slack
024521:                 )
024522:                 if still_on_server_half:
024523:                     print(
024524:                         f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
024525:                         f"serve candidate {ball_position} is still on server half "
024526:                         f"(net_y={net_y:.1f}, slack={crossing_slack:.1f})"
024527:                     )
024528:                     return False, None
024529: 
024530:         # During the first serve flight, a stale HSV fragment near the net can
024531:         # look like a sideline reversal (for example point 10 a prior frame at
024532:         # (2598,553)) even though the real ball is still descending on the
024533:         # server's side.  Do not let the generic out-of-court path terminate
024534:         # the point in this narrow net band; the dedicated serve/net handlers
024535:         # below still get to classify a real net fault or service bounce.
024536:         if (
024537:             self.point_start_frame_internal is not None and
024538:             int(getattr(self, 'serve_direction_dy', 0)) != 0 and
024539:             (self.frame_count - self.point_start_frame_internal) <= self._serve_bounce_frame_limit() and
024540:             not self._serve_net_touch_active(window_frames=120) and
024541:             not self._serve_net_pass_active(window_frames=36) and
024542:             int(getattr(self, 'ground_bounce_count', 0)) <= 0
024543:         ):
024544:             net_geometry = self._net_contact_geometry(ball_position)
024545:             net_y = float(
024546:                 net_geometry.get('net_y', (self.net_area_y_min + self.net_area_y_max) * 0.5)
024547:                 if net_geometry is not None else
024548:                 (self.net_area_y_min + self.net_area_y_max) * 0.5
024549:             )
024550:             net_band = max(100.0, frame.shape[0] * 0.040)
024551:             if abs(float(ball_position[1]) - net_y) <= net_band:
024552:                 print(
024553:                     f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
024554:                     f"early-serve net-band artifact at {ball_position} "
024555:                     f"(net_y={net_y:.1f}, band={net_band:.1f})"
024556:                 )
024557:                 return False, None
024558: 
024559:         x, y = ball_position
024560:         curr_dx = float(self.last_motion.get('dx', 0.0) or 0.0)
024561:         curr_dy = float(self.last_motion.get('dy', 0.0) or 0.0)
024562:         prev_dx = float(self.prev_motion.get('dx', 0.0) or 0.0)
024563:         prev_dy = float(self.prev_motion.get('dy', 0.0) or 0.0)
```
