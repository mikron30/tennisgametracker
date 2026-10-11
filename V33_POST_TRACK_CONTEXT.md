# V33 post-track state context

## 'def _try_local_ai_recovery' — 1 hit(s)
```python
003080:             else:
003081:                 print(
003082:                     f"[LOCAL_AI_TRAJECTORY] f{self.frame_count}: no safe candidate "
003083:                     f"prev={previous_position} predicted={predicted} radius={roi_radius:.1f}px"
003084:                 )
003085:             return None
003086: 
003087:         point = (int(selected_candidate["x"]), int(selected_candidate["y"]))
003088:         step_dx = float(point[0]) - float(previous_position[0])
003089:         step_dy = float(point[1]) - float(previous_position[1])
003090:         step = math.hypot(step_dx, step_dy)
003091:         selected_candidate["trajectory_rescue"] = True
003092:         selected_candidate["trajectory_prediction"] = predicted
003093:         selected_candidate["trajectory_prediction_error"] = float(selected_row["prediction_error"])
003094:         selected_candidate["trajectory_cost"] = float(selected_row["trajectory_cost"])
003095:         if post_serve_corridor_active:
003096:             selected_candidate["trajectory_post_serve_corridor"] = True
003097:             selected_candidate["trajectory_corridor_along"] = float(selected_row["corridor_along"])
003098:             selected_candidate["trajectory_corridor_cross"] = float(selected_row["corridor_cross"])
003099:         selected_candidate["trajectory_motion"] = {
003100:             "dx": step_dx,
003101:             "dy": step_dy,
003102:             "distance": step,
003103:             "direction_deg": math.degrees(math.atan2(step_dy, step_dx)) if step > 0.0 else None,
003104:         }
003105:         mode_label = " corridor" if post_serve_corridor_active else ""
003106:         print(
003107:             f"[LOCAL_AI_TRAJECTORY_ACCEPT] f{self.frame_count}:{mode_label} "
003108:             f"{point} ai={float(selected_candidate.get('ai_score', 0.0)):.6f} "
003109:             f"pred={predicted} err={selected_row['prediction_error']:.1f}px "
003110:             f"motion={selected_row['motion_mean']:.1f}/{selected_row['motion_max']:.1f} "
003111:             f"zone={selected_row['zone'] or 'clear'} cost={selected_row['trajectory_cost']:.2f}"
003112:         )
003113:         return selected_candidate
003114: 
003115:     def _try_local_ai_recovery(
003116:             self, previous_position, tracked_position, previous_stuck,
003117:             pre_track_snapshot=None, frame=None):
003118:         """Repair a suspect selection from the recent raw-frame buffer if safe."""
003119:         if (getattr(self, '_temporal_top_reentry_frame', -1) == self.frame_count and
003120:                 getattr(self, '_temporal_top_reentry_pos', None) == tracked_position):
003121:             return tracked_position
003122:         confirmed_static_artifact = self._finalize_pending_provisional_static_candidate()
003123:         if (
003124:                 confirmed_static_artifact and previous_position is not None and
003125:                 tracked_position is not None and pre_track_snapshot is not None and
003126:                 math.hypot(
003127:                     float(tracked_position[0]) - float(getattr(self, "_last_static_guard_position", tracked_position)[0]),
003128:                     float(tracked_position[1]) - float(getattr(self, "_last_static_guard_position", tracked_position)[1]),
003129:                 ) <= 22.0):
003130:             self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
003131:             # Re-run the normal candidate selection now that the confirmed
003132:             # static blob is ignored.  This keeps the real next-frame toss
003133:             # (a prior frame here) instead of deliberately dropping a visible ball.
003134:             retracked = self.track_ball_in_frame(frame) if frame is not None else None
003135:             if retracked is not None and math.hypot(
003136:                     float(retracked[0]) - float(self._last_static_guard_position[0]),
003137:                     float(retracked[1]) - float(self._last_static_guard_position[1]),
003138:             ) > 22.0:
003139:                 print(
003140:                     f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
003141:                     f"retracked {retracked} after excluding static selection"
003142:                 )
003143:                 return retracked
003144:             self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
003145:             self.stuck_frame_count = int(previous_stuck) + 1
003146:             print(
003147:                 f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
003148:                 f"held current static selection {tracked_position}; resuming from {previous_position}"
003149:             )
003150:             return previous_position
003151:         # Keep a weak player/background reacquisition from becoming the
003152:         # next tracking anchor before tight local AI can arbitrate it.
003153:         if (
003154:                 pre_track_snapshot is not None and
003155:                 previous_position is not None and
003156:                 tracked_position is not None and
003157:                 int(previous_stuck or 0) >= 3):
003158:             reacq_jump = math.hypot(
003159:                 float(tracked_position[0]) - float(previous_position[0]),
003160:                 float(tracked_position[1]) - float(previous_position[1]),
003161:             )
003162:             reacq_motion_mean = float(
003163:                 getattr(self, '_last_tracked_candidate_motion_mean', 0.0) or 0.0
003164:             )
003165:             reacq_motion_max = float(
003166:                 getattr(self, '_last_tracked_candidate_motion_max', 0.0) or 0.0
003167:             )
003168:             reacq_zone = self._player_point_zone(tracked_position)
003169:             weak_player_candidate = (
003170:                 reacq_zone is not None and
003171:                 reacq_motion_mean < 8.0 and
003172:                 reacq_motion_max < 50.0
003173:             )
003174:             weak_long_jump = (
003175:                 reacq_jump >= 150.0 and
003176:                 reacq_motion_mean < 12.0 and
003177:                 reacq_motion_max < 80.0
003178:             )
003179:             if weak_player_candidate or weak_long_jump:
003180:                 rejected_position = tuple(tracked_position)
003181:                 self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
003182:                 self.stuck_frame_count = int(previous_stuck or 0) + 1
003183:                 tracked_position = tuple(previous_position)
003184:                 if getattr(self, '_last_motion_reacq_frame', -1000000) == self.frame_count:
003185:                     self._last_motion_reacq_frame = -1000000
003186:                     self._last_motion_reacq_pos = None
003187:                 if weak_player_candidate:
003188:                     print(
003189:                         f'[PLAYER_REACQ_WEAK_PLAYER_REJECTED] f{self.frame_count}: '
003190:                         f'rejected {rejected_position} zone={reacq_zone} '
003191:                         f'motion={reacq_motion_mean:.1f}/{reacq_motion_max:.1f}; '
003192:                         f'restored {previous_position} stuck={self.stuck_frame_count}'
003193:                     )
003194:                 else:
003195:                     print(
003196:                         f'[PLAYER_REACQ_WEAK_JUMP_REJECTED] f{self.frame_count}: '
003197:                         f'rejected {rejected_position} jump={reacq_jump:.1f}px '
003198:                         f'motion={reacq_motion_mean:.1f}/{reacq_motion_max:.1f}; '
003199:                         f'restored {previous_position} stuck={self.stuck_frame_count}'
003200:                     )
003201: 
003202:         if self._normal_ball_evidence_is_strong(frame, previous_position, tracked_position, pre_track_snapshot):
003203:             self._contact_local_ai_state = None
003204:             return tracked_position
003205:         if not self._automatic_local_ai_scoring_ready():
003206:             return tracked_position
003207:         tight_hold_recovery = self._try_active_tight_local_ai_hold(
003208:             frame,
003209:             tracked_position,
003210:             previous_gray=(pre_track_snapshot or {}).get('_prev_frame_gray'),
003211:         )
003212:         if tight_hold_recovery is not None:
003213:             return tight_hold_recovery
003214:         reason = self._local_ai_recovery_reason(
003215:             previous_position, tracked_position, previous_stuck
003216:         )
003217:         if reason is None:
003218:             self._maybe_clear_post_serve_pre_net_recovery(
003219:                 previous_position, tracked_position, source='normal-tracker'
003220:             )
003221:             return tracked_position
003222:         post_serve_recovery_active = (
003223:             bool(getattr(self, '_post_serve_pre_net_recovery_active', False)) or
003224:             int(self.frame_count) <= max(
003225:                 int(getattr(self, "_post_serve_launch_lock_until_frame", -1)),
003226:                 int(getattr(self, "_post_serve_recovery_hold_until_frame", -1000000)),
003227:             )
003228:         )
003229:         trajectory_recovered = self._try_local_ai_trajectory_rescue(
003230:             frame,
003231:             previous_position,
003232:             tracked_position,
003233:             previous_stuck,
003234:             pre_track_snapshot,
003235:             reason,
003236:         )
003237:         if trajectory_recovered is not None:
003238:             recovered = trajectory_recovered
003239:             reason = f"{reason}+trajectory"
003240:         else:
003241:             recovery_reason = reason
003242:             if post_serve_recovery_active and reason.startswith("player-region:"):
003243:                 # Keep using the post-serve low-saturation candidate pool on
003244:                 # the frames after the first rejected jump.  The normal tracker
003245:                 # may keep proposing a player contour until the ball emerges.
003246:                 recovery_reason = f"post-serve-launch-occlusion:{reason}"
003247:                 print(
003248:                     f"[POST_SERVE_RECOVERY_RETRY] f{self.frame_count}: "
003249:                     f"{reason} -> {recovery_reason}"
003250:                 )
003251:             recovered = self.local_ai_recovery.recover(
003252:                 self.frame_count,
003253:                 self._local_ai_frame_buffer,
003254:                 predicted_position=previous_position,
003255:                 player_zone=self._player_point_zone,
003256:                 reason=recovery_reason,
003257:                 force=False,
003258:             )
003259:         if recovered is None:
003260:             if getattr(self.local_ai_recovery, "last_rejection", None) == "all-player-body-path":
003261:                 self._local_ai_all_body_rejections = int(
003262:                     getattr(self, "_local_ai_all_body_rejections", 0)
003263:                 ) + 1
003264:                 point_age = int(self.frame_count) - int(
003265:                     getattr(self, "point_start_frame_internal", self.frame_count) or self.frame_count
003266:                 )
003267:                 if (
003268:                         self._local_ai_all_body_rejections >= 1 and
003269:                         not post_serve_recovery_active and
003270:                         # A previously accepted non-body recovery is positive
003271:                         # evidence that this is a real serve/rally.  A later
003272:                         # player-body-only attempt can happen at contact or
003273:                         # occlusion and must not erase that valid point.
003274:                         int(getattr(self, "_local_ai_recovery_count", 0)) == 0 and
003275:                         int(getattr(self, "_point_hit_count", 0)) == 0 and
003276:                         point_age <= 150):
003277:                     self._discard_provisional_serve_from_ai = True
003278:             else:
003279:                 self._local_ai_all_body_rejections = 0
003280:             print(f"[LOCAL_AI_RECOVERY] f{self.frame_count}: no safe path ({reason})")
003281:             # POST_SERVE_STATE_ROLLBACK_V3
003282:             # The reason detector runs after normal HSV tracking has already
003283:             # mutated ball_center/motion/history.  If the post-serve guard says
003284:             # that mutation is unsafe and AI cannot replace it, restore the
003285:             # complete pre-track snapshot immediately.  Returning the previous
003286:             # position makes the main loop log/record the held sane point rather
003287:             # than the rejected player pixel.
003288:             if (
003289:                     post_serve_recovery_active and
003290:                     pre_track_snapshot is not None and
003291:                     previous_position is not None and
003292:                     (
003293:                         reason.startswith("post-serve-launch-") or
003294:                         reason.startswith("player-region:")
003295:                     )):
003296:                 rejected_position = (
003297:                     tuple(tracked_position) if tracked_position is not None else None
003298:                 )
003299:                 self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
003300:                 self.stuck_frame_count = max(
003301:                     int(previous_stuck or 0) + 1,
003302:                     int(getattr(self, "stuck_frame_count", 0) or 0),
003303:                 )
003304:                 if getattr(self, "_last_motion_reacq_frame", -1000000) == self.frame_count:
003305:                     self._last_motion_reacq_frame = -1000000
003306:                     self._last_motion_reacq_pos = None
003307:                 print(
003308:                     f"[POST_SERVE_LAUNCH_ROLLBACK] f{self.frame_count}: "
003309:                     f"rejected={rejected_position} reason={reason}; "
003310:                     f"restored={previous_position} stuck={self.stuck_frame_count} "
003311:                     f"retry_through=f{int(getattr(self, '_post_serve_recovery_hold_until_frame', -1))}"
003312:                 )
003313:                 return tuple(previous_position)
003314:             # If HSV just teleported from a physically coherent ball path into a
003315:             # player region and local AI could not verify a replacement, never let
003316:             # that player pixel become the next-frame anchor. Restore the complete
003317:             # pre-track state (position, size, motion and velocity history) and let
003318:             # the rapid AI retry continue from the last sane trajectory.
003319:             if (
003320:                     reason.startswith("player-region:") and
003321:                     pre_track_snapshot is not None and
003322:                     previous_position is not None and
003323:                     tracked_position is not None):
003324:                 player_jump = math.hypot(
003325:                     float(tracked_position[0]) - float(previous_position[0]),
003326:                     float(tracked_position[1]) - float(previous_position[1]),
003327:                 )
003328:                 prior_motion = pre_track_snapshot.get("last_motion") or {}
003329:                 prior_speed = float(prior_motion.get("distance", 0.0) or 0.0)
003330:                 prior_velocities = pre_track_snapshot.get("ball_velocity_history") or []
003331:                 if prior_velocities:
003332:                     prior_speed = max(
003333:                         prior_speed,
003334:                         max(float(value or 0.0) for value in prior_velocities[-3:]),
003335:                     )
003336:                 jump_limit = max(150.0, min(520.0, prior_speed * 3.2 + 60.0))
003337:                 if player_jump > jump_limit:
003338:                     rejected_position = tuple(tracked_position)
003339:                     self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
003340:                     self.stuck_frame_count = int(previous_stuck) + 1
003341:                     if getattr(self, "_last_motion_reacq_frame", -1000000) == self.frame_count:
003342:                         self._last_motion_reacq_frame = -1000000
003343:                         self._last_motion_reacq_pos = None
003344:                     print(
003345:                         f"[LOCAL_AI_PLAYER_JUMP_ROLLBACK] f{self.frame_count}: "
003346:                         f"rejected {rejected_position} jump={player_jump:.1f}px > "
003347:                         f"{jump_limit:.1f}px; restored {previous_position} "
003348:                         f"stuck={self.stuck_frame_count}"
003349:                     )
003350:                     return previous_position
003351:             # Do not let a local-AI rejection still mutate the track.  In the
003352:             # serve-start state a compact candidate with no inter-frame motion
003353:             # is a likely static highlight.  Restore the state from before
003354:             # HSV committed it, then use the next decoded frame to determine
003355:             # whether that blob persisted in place.
003356:             if (
003357:                     reason.startswith(("provisional-serve-jump:", "static-candidate-jump:")) and
003358:                     pre_track_snapshot is not None and
003359:                     tracked_position is not None and
003360:                     float(getattr(self, "_last_tracked_candidate_motion_mean", 0.0) or 0.0) < 5.0 and
003361:                     float(getattr(self, "_last_tracked_candidate_motion_max", 0.0) or 0.0) < 25.0):
003362:                 current_image = None
003363:                 if self._local_ai_frame_buffer:
003364:                     current_image = self._local_ai_frame_buffer[-1].get("image")
003365:                 self._pending_provisional_static_candidate = {
003366:                     "frame": int(self.frame_count),
003367:                     "pos": tuple(tracked_position),
003368:                     "blob": self._static_blob_near(current_image, tracked_position),
003369:                 }
003370:                 self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
003371:                 self.stuck_frame_count = int(previous_stuck) + 1
003372:                 print(
003373:                     f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
003374:                     f"deferred zero-motion rejected jump {tracked_position}; "
003375:                     "waiting for next-frame persistence"
003376:                 )
003377:                 return previous_position
003378:             return tracked_position
003379:         self._local_ai_all_body_rejections = 0
003380:         repaired_position = (int(recovered["x"]), int(recovered["y"]))
003381:         strict_repaired_reason = self._post_serve_pre_net_recovery_reason(
003382:             previous_position, repaired_position
003383:         )
003384:         if strict_repaired_reason is not None:
003385:             rejected_position = tuple(repaired_position)
003386:             if pre_track_snapshot is not None:
003387:                 self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
003388:             self.stuck_frame_count = max(
003389:                 int(previous_stuck or 0) + 1,
003390:                 int(getattr(self, 'stuck_frame_count', 0) or 0),
003391:             )
003392:             print(
003393:                 f"[POST_SERVE_PRE_NET_AI_REJECT] f{self.frame_count}: "
003394:                 f"rejected={rejected_position} reason={strict_repaired_reason}; "
003395:                 f"restored={previous_position} stuck={self.stuck_frame_count}"
003396:             )
003397:             return tuple(previous_position) if previous_position is not None else None
003398: 
003399:         # V33: Local-AI recovery runs after the normal tracker has already
003400:         # committed its candidate.  If AI replaces that candidate, repair the
003401:         # *entire* live tracking state atomically.  Keeping only the repaired
003402:         # coordinate leaves last_motion / velocity history from the rejected
003403:         # contour and poisons the next-frame prediction.
003404:         committed_position = (
003405:             tuple(tracked_position) if tracked_position is not None else None
003406:         )
003407:         recovery_replaced_committed_candidate = (
003408:             pre_track_snapshot is not None and
003409:             previous_position is not None and
003410:             committed_position is not None and
003411:             committed_position != tuple(repaired_position)
003412:         )
003413:         recovered_area = float(recovered.get("area", self.ball_size or 0.0))
003414: 
003415:         if recovery_replaced_committed_candidate:
003416:             self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
003417:             self._maybe_clear_post_serve_pre_net_recovery(
003418:                 previous_position, repaired_position, source='local-ai'
003419:             )
003420: 
003421:             repaired_dx = float(repaired_position[0]) - float(previous_position[0])
003422:             repaired_dy = float(repaired_position[1]) - float(previous_position[1])
003423:             repaired_distance = math.hypot(repaired_dx, repaired_dy)
003424:             repaired_direction = (
003425:                 math.degrees(math.atan2(repaired_dy, repaired_dx))
003426:                 if repaired_distance > 0.0 else None
003427:             )
003428:             repaired_motion = {
003429:                 'distance': repaired_distance,
003430:                 'dx': repaired_dx,
003431:                 'dy': repaired_dy,
003432:                 'direction_deg': repaired_direction,
003433:             }
003434: 
003435:             self.ball_center = repaired_position
003436:             self.ball_size = recovered_area
003437:             prior_motion = pre_track_snapshot.get("last_motion")
003438:             self.prev_motion = (
003439:                 dict(prior_motion) if isinstance(prior_motion, dict) else prior_motion
003440:             )
003441:             self.last_motion = repaired_motion
003442:             if repaired_distance >= 3.0:
003443:                 self.last_nonzero_motion = dict(repaired_motion)
003444:             prior_history = list(pre_track_snapshot.get("ball_velocity_history") or [])
003445:             prior_history.append(repaired_distance)
003446:             self.ball_velocity_history = prior_history[-5:]
003447:             self._held_direction_candidate = None
003448: 
003449:             # The rollback restores the prior gray frame as well.  For an
003450:             # accepted current-frame repair, the next frame must compare against
003451:             # this decoded frame, not against frame N-1 twice.
003452:             if frame is not None:
003453:                 try:
003454:                     self._prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
003455:                 except cv2.error:
003456:                     pass
003457: 
003458:             if (
003459:                 getattr(self, '_last_motion_reacq_frame', -1000000) == self.frame_count and
003460:                 getattr(self, '_last_motion_reacq_pos', None) == committed_position
003461:             ):
003462:                 self._last_motion_reacq_frame = -1000000
003463:                 self._last_motion_reacq_pos = None
003464: 
003465:             # Replace a same-frame motion-history sample from the rejected
003466:             # contour rather than leaving two contradictory versions of frame N.
003467:             if self.motion_history and int(self.motion_history[-1].get('frame', -1)) == int(self.frame_count):
003468:                 self.motion_history[-1] = {
003469:                     'frame': self.frame_count,
003470:                     'distance': repaired_distance,
003471:                     'direction_deg': repaired_direction,
003472:                     'pos': tuple(repaired_position),
003473:                     'prev_pos': tuple(previous_position),
003474:                 }
003475: 
003476:             print(
003477:                 f"[POST_TRACK_STATE_REPAIR] f{self.frame_count}: "
003478:                 f"committed={committed_position} -> repaired={repaired_position} "
003479:                 f"motion={repaired_distance:.1f}px "
003480:                 f"velocity_history={self.ball_velocity_history}"
003481:             )
003482:         else:
003483:             self._maybe_clear_post_serve_pre_net_recovery(
003484:                 previous_position, repaired_position, source='local-ai'
003485:             )
003486:             self.ball_center = repaired_position
003487:             self.ball_size = recovered_area
003488:             if recovered.get("trajectory_rescue"):
003489:                 trajectory_motion = dict(recovered.get("trajectory_motion") or {})
003490:                 prior_motion = (pre_track_snapshot or {}).get("last_motion")
003491:                 self.prev_motion = dict(prior_motion) if isinstance(prior_motion, dict) else prior_motion
003492:                 self.last_motion = trajectory_motion
003493:                 if float(trajectory_motion.get("distance", 0.0) or 0.0) >= 3.0:
003494:                     self.last_nonzero_motion = dict(trajectory_motion)
003495:                 prior_history = list((pre_track_snapshot or {}).get("ball_velocity_history") or [])
003496:                 prior_history.append(float(trajectory_motion.get("distance", 0.0) or 0.0))
003497:                 self.ball_velocity_history = prior_history[-5:]
003498:                 self._held_direction_candidate = None
003499:         self.last_seen_frame = self.frame_count
003500:         self.stuck_frame_count = 0
003501:         self._pending_rally_end_reason = None
003502:         self._pending_rally_end_frame = -1
003503:         self._local_ai_recovery_count += 1
003504:         if reason != "post-recovery-follow":
003505:             # A successful repair commonly happens at contact/occlusion.  The
003506:             # next few frames are the outgoing flight, where the normal tracker
003507:             # has the least reliable prediction.  Establish a bounded handoff
003508:             # window; each *verified* local-AI frame can extend the immediate
003509:             # follow-up by four frames, but never beyond this deadline.
003510:             self._local_ai_handoff_deadline_frame = self.frame_count + 14
003511: 
003512:         handoff_deadline = int(getattr(self, "_local_ai_handoff_deadline_frame", -1))
003513:         if handoff_deadline >= self.frame_count:
003514:             self._local_ai_follow_until_frame = min(
003515:                 handoff_deadline,
003516:                 max(
003517:                     int(getattr(self, "_local_ai_follow_until_frame", -1)),
003518:                     self.frame_count + 4,
003519:                 ),
003520:             )
003521:         print(
003522:             f"[LOCAL_AI_RECOVERY] f{self.frame_count}: accepted {repaired_position} "
003523:             f"score={float(recovered.get('ai_score', 0.0)):.6f} "
003524:             f"replayed={self.local_ai_recovery.lookback_frames}f reason={reason}"
003525:         )
003526:         return repaired_position
003527: 
003528:     def _player_serve_context(self, serve_position):
003529:         tracker = getattr(self, "player_tracker", None)
003530:         if tracker is None:
003531:             return {}
003532:         try:
003533:             server_idx = self._current_server_index()
003534:             context = tracker.serve_context(
003535:                 serve_position,
003536:                 player_name=self.player_names[server_idx] if server_idx is not None else "",
003537:             )
003538:             tracker.adopt_serve_context(context, frame_index=int(getattr(self, 'frame_count', -1)))
003539:             return context
003540:         except Exception:
003541:             return {}
003542: 
003543:     def _refresh_player_serve_context(self):
003544:         """Refresh server/receiver geometry while the serve is still starting."""
003545:         tracker = getattr(self, "player_tracker", None)
003546:         context = getattr(self, "_point_history_current", None)
003547:         if tracker is None or context is None:
003548:             return
003549:         serve_context = context.get('serve_context') or {}
003550:         serve_x = serve_context.get('serve_x')
003551:         serve_y = serve_context.get('serve_y')
003552:         if serve_x is None or serve_y is None:
003553:             return
003554:         try:
003555:             previous_receiver = serve_context.get('receiver_position')
003556:             refreshed = self._player_serve_context((serve_x, serve_y))
003557:             if not refreshed:
003558:                 return
003559:             # Do not replace a valid serve-ball coordinate; update the player
003560:             # detections as they become available during the opening frames.
003561:             for key, value in refreshed.items():
003562:                 if value is not None and value != '':
003563:                     serve_context[key] = value
003564:             context['serve_context'] = serve_context
003565:             current_receiver = serve_context.get('receiver_position')
003566:             if current_receiver and current_receiver != previous_receiver:
003567:                 print(
003568:                     f"[SERVE_PLAYERS] f{self.frame_count}: "
003569:                     f"server={serve_context.get('player') or 'unknown'} "
003570:                     f"side={serve_context.get('player_side')} "
003571:                     f"position={serve_context.get('player_position')} "
003572:                     f"receiver_side={serve_context.get('receiver_side')} "
003573:                     f"receiver_position={current_receiver} "
003574:                     f"area={serve_context.get('receiver_area')}"
003575:                 )
003576:         except Exception:
003577:             return
003578: 
003579:     def _draw_player_tracking(self, frame, scale=1.0):
003580:         tracker = getattr(self, "player_tracker", None)
003581:         if tracker is None:
003582:             return frame
003583:         if not getattr(self, "tracking", False) and getattr(self, "_point_history_current", None) is None:
003584:             return frame
003585:         result = frame
003586:         try:
003587:             for side, track in tracker.tracks.items():
003588:                 if track.bbox is None:
003589:                     continue
003590:                 x, y, w, h = track.bbox
003591:                 x1 = int(round(x * scale))
003592:                 y1 = int(round(y * scale))
003593:                 x2 = int(round((x + w) * scale))
003594:                 y2 = int(round((y + h) * scale))
003595:                 color = (255, 180, 40) if side == "near" else (40, 190, 255)
003596:                 thickness = 2 if track.visible else 1
003597:                 cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
003598:                 if track.head is not None:
003599:                     hx, hy = int(track.head[0] * scale), int(track.head[1] * scale)
003600:                     cv2.circle(result, (hx, hy), max(4, int(8 * scale)), color, 1)
003601:                 if track.shoes is not None:
003602:                     sx, sy = int(track.shoes[0] * scale), int(track.shoes[1] * scale)
003603:                     cv2.circle(result, (sx, sy), max(3, int(6 * scale)), color, 1)
003604:                 if track.racket and track.racket.get("line"):
003605:                     lx1, ly1, lx2, ly2 = track.racket["line"]
003606:                     cv2.line(
003607:                         result,
003608:                         (int(lx1 * scale), int(ly1 * scale)),
003609:                         (int(lx2 * scale), int(ly2 * scale)),
003610:                         (0, 165, 255),
003611:                         max(1, int(round(2 * scale))),
003612:                     )
003613:                 label = f"{side} {track.confidence:.2f}"
003614:                 cv2.putText(
003615:                     result,
003616:                     label,
003617:                     (x1, max(15, y1 - 5)),
003618:                     cv2.FONT_HERSHEY_SIMPLEX,
003619:                     max(0.35, 0.5 * scale),
003620:                     color,
003621:                     max(1, int(round(1.5 * scale))),
003622:                     cv2.LINE_AA,
003623:                 )
003624:         except Exception:
003625:             return result
003626:         return result
003627: 
003628:     def log_motion_metrics(self, prev_pos, dx, dy, distance, direction_deg):
003629:         """Log per-frame motion and raise a focus-loss flag when movement spikes."""
003630:         direction_text = f"{direction_deg:+.1f} deg" if direction_deg is not None else "N/A"
003631:         print(f"Frame {self.frame_count}: Movement {distance:.1f}px (dx={dx}, dy={dy}) Direction: {direction_text}")
003632: 
003633:         self.prev_motion = self.last_motion
003634:         self.last_motion = {
```

## 'repaired_position = (int(recovered["x"]), int(recovered["y"]))' — 1 hit(s)
```python
003345:                         f"[LOCAL_AI_PLAYER_JUMP_ROLLBACK] f{self.frame_count}: "
003346:                         f"rejected {rejected_position} jump={player_jump:.1f}px > "
003347:                         f"{jump_limit:.1f}px; restored {previous_position} "
003348:                         f"stuck={self.stuck_frame_count}"
003349:                     )
003350:                     return previous_position
003351:             # Do not let a local-AI rejection still mutate the track.  In the
003352:             # serve-start state a compact candidate with no inter-frame motion
003353:             # is a likely static highlight.  Restore the state from before
003354:             # HSV committed it, then use the next decoded frame to determine
003355:             # whether that blob persisted in place.
003356:             if (
003357:                     reason.startswith(("provisional-serve-jump:", "static-candidate-jump:")) and
003358:                     pre_track_snapshot is not None and
003359:                     tracked_position is not None and
003360:                     float(getattr(self, "_last_tracked_candidate_motion_mean", 0.0) or 0.0) < 5.0 and
003361:                     float(getattr(self, "_last_tracked_candidate_motion_max", 0.0) or 0.0) < 25.0):
003362:                 current_image = None
003363:                 if self._local_ai_frame_buffer:
003364:                     current_image = self._local_ai_frame_buffer[-1].get("image")
003365:                 self._pending_provisional_static_candidate = {
003366:                     "frame": int(self.frame_count),
003367:                     "pos": tuple(tracked_position),
003368:                     "blob": self._static_blob_near(current_image, tracked_position),
003369:                 }
003370:                 self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
003371:                 self.stuck_frame_count = int(previous_stuck) + 1
003372:                 print(
003373:                     f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
003374:                     f"deferred zero-motion rejected jump {tracked_position}; "
003375:                     "waiting for next-frame persistence"
003376:                 )
003377:                 return previous_position
003378:             return tracked_position
003379:         self._local_ai_all_body_rejections = 0
003380:         repaired_position = (int(recovered["x"]), int(recovered["y"]))
003381:         strict_repaired_reason = self._post_serve_pre_net_recovery_reason(
003382:             previous_position, repaired_position
003383:         )
003384:         if strict_repaired_reason is not None:
003385:             rejected_position = tuple(repaired_position)
003386:             if pre_track_snapshot is not None:
003387:                 self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
003388:             self.stuck_frame_count = max(
003389:                 int(previous_stuck or 0) + 1,
003390:                 int(getattr(self, 'stuck_frame_count', 0) or 0),
003391:             )
003392:             print(
003393:                 f"[POST_SERVE_PRE_NET_AI_REJECT] f{self.frame_count}: "
003394:                 f"rejected={rejected_position} reason={strict_repaired_reason}; "
003395:                 f"restored={previous_position} stuck={self.stuck_frame_count}"
003396:             )
003397:             return tuple(previous_position) if previous_position is not None else None
003398: 
003399:         # V33: Local-AI recovery runs after the normal tracker has already
003400:         # committed its candidate.  If AI replaces that candidate, repair the
003401:         # *entire* live tracking state atomically.  Keeping only the repaired
003402:         # coordinate leaves last_motion / velocity history from the rejected
003403:         # contour and poisons the next-frame prediction.
003404:         committed_position = (
003405:             tuple(tracked_position) if tracked_position is not None else None
003406:         )
003407:         recovery_replaced_committed_candidate = (
003408:             pre_track_snapshot is not None and
003409:             previous_position is not None and
003410:             committed_position is not None and
003411:             committed_position != tuple(repaired_position)
003412:         )
003413:         recovered_area = float(recovered.get("area", self.ball_size or 0.0))
003414: 
003415:         if recovery_replaced_committed_candidate:
003416:             self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
003417:             self._maybe_clear_post_serve_pre_net_recovery(
003418:                 previous_position, repaired_position, source='local-ai'
003419:             )
003420: 
003421:             repaired_dx = float(repaired_position[0]) - float(previous_position[0])
003422:             repaired_dy = float(repaired_position[1]) - float(previous_position[1])
003423:             repaired_distance = math.hypot(repaired_dx, repaired_dy)
003424:             repaired_direction = (
003425:                 math.degrees(math.atan2(repaired_dy, repaired_dx))
003426:                 if repaired_distance > 0.0 else None
003427:             )
003428:             repaired_motion = {
003429:                 'distance': repaired_distance,
003430:                 'dx': repaired_dx,
003431:                 'dy': repaired_dy,
003432:                 'direction_deg': repaired_direction,
003433:             }
003434: 
003435:             self.ball_center = repaired_position
003436:             self.ball_size = recovered_area
003437:             prior_motion = pre_track_snapshot.get("last_motion")
003438:             self.prev_motion = (
003439:                 dict(prior_motion) if isinstance(prior_motion, dict) else prior_motion
003440:             )
003441:             self.last_motion = repaired_motion
003442:             if repaired_distance >= 3.0:
003443:                 self.last_nonzero_motion = dict(repaired_motion)
003444:             prior_history = list(pre_track_snapshot.get("ball_velocity_history") or [])
003445:             prior_history.append(repaired_distance)
003446:             self.ball_velocity_history = prior_history[-5:]
003447:             self._held_direction_candidate = None
003448: 
003449:             # The rollback restores the prior gray frame as well.  For an
003450:             # accepted current-frame repair, the next frame must compare against
003451:             # this decoded frame, not against frame N-1 twice.
003452:             if frame is not None:
003453:                 try:
003454:                     self._prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
003455:                 except cv2.error:
003456:                     pass
003457: 
003458:             if (
003459:                 getattr(self, '_last_motion_reacq_frame', -1000000) == self.frame_count and
003460:                 getattr(self, '_last_motion_reacq_pos', None) == committed_position
003461:             ):
003462:                 self._last_motion_reacq_frame = -1000000
003463:                 self._last_motion_reacq_pos = None
003464: 
003465:             # Replace a same-frame motion-history sample from the rejected
003466:             # contour rather than leaving two contradictory versions of frame N.
003467:             if self.motion_history and int(self.motion_history[-1].get('frame', -1)) == int(self.frame_count):
003468:                 self.motion_history[-1] = {
003469:                     'frame': self.frame_count,
003470:                     'distance': repaired_distance,
003471:                     'direction_deg': repaired_direction,
003472:                     'pos': tuple(repaired_position),
003473:                     'prev_pos': tuple(previous_position),
003474:                 }
003475: 
003476:             print(
003477:                 f"[POST_TRACK_STATE_REPAIR] f{self.frame_count}: "
003478:                 f"committed={committed_position} -> repaired={repaired_position} "
003479:                 f"motion={repaired_distance:.1f}px "
003480:                 f"velocity_history={self.ball_velocity_history}"
003481:             )
003482:         else:
003483:             self._maybe_clear_post_serve_pre_net_recovery(
003484:                 previous_position, repaired_position, source='local-ai'
003485:             )
003486:             self.ball_center = repaired_position
003487:             self.ball_size = recovered_area
003488:             if recovered.get("trajectory_rescue"):
003489:                 trajectory_motion = dict(recovered.get("trajectory_motion") or {})
003490:                 prior_motion = (pre_track_snapshot or {}).get("last_motion")
003491:                 self.prev_motion = dict(prior_motion) if isinstance(prior_motion, dict) else prior_motion
003492:                 self.last_motion = trajectory_motion
003493:                 if float(trajectory_motion.get("distance", 0.0) or 0.0) >= 3.0:
003494:                     self.last_nonzero_motion = dict(trajectory_motion)
003495:                 prior_history = list((pre_track_snapshot or {}).get("ball_velocity_history") or [])
003496:                 prior_history.append(float(trajectory_motion.get("distance", 0.0) or 0.0))
003497:                 self.ball_velocity_history = prior_history[-5:]
003498:                 self._held_direction_candidate = None
003499:         self.last_seen_frame = self.frame_count
003500:         self.stuck_frame_count = 0
003501:         self._pending_rally_end_reason = None
003502:         self._pending_rally_end_frame = -1
003503:         self._local_ai_recovery_count += 1
003504:         if reason != "post-recovery-follow":
003505:             # A successful repair commonly happens at contact/occlusion.  The
003506:             # next few frames are the outgoing flight, where the normal tracker
003507:             # has the least reliable prediction.  Establish a bounded handoff
003508:             # window; each *verified* local-AI frame can extend the immediate
003509:             # follow-up by four frames, but never beyond this deadline.
003510:             self._local_ai_handoff_deadline_frame = self.frame_count + 14
003511: 
003512:         handoff_deadline = int(getattr(self, "_local_ai_handoff_deadline_frame", -1))
003513:         if handoff_deadline >= self.frame_count:
003514:             self._local_ai_follow_until_frame = min(
003515:                 handoff_deadline,
003516:                 max(
003517:                     int(getattr(self, "_local_ai_follow_until_frame", -1)),
003518:                     self.frame_count + 4,
003519:                 ),
003520:             )
003521:         print(
003522:             f"[LOCAL_AI_RECOVERY] f{self.frame_count}: accepted {repaired_position} "
003523:             f"score={float(recovered.get('ai_score', 0.0)):.6f} "
003524:             f"replayed={self.local_ai_recovery.lookback_frames}f reason={reason}"
003525:         )
003526:         return repaired_position
003527: 
003528:     def _player_serve_context(self, serve_position):
003529:         tracker = getattr(self, "player_tracker", None)
003530:         if tracker is None:
003531:             return {}
003532:         try:
003533:             server_idx = self._current_server_index()
003534:             context = tracker.serve_context(
003535:                 serve_position,
003536:                 player_name=self.player_names[server_idx] if server_idx is not None else "",
003537:             )
003538:             tracker.adopt_serve_context(context, frame_index=int(getattr(self, 'frame_count', -1)))
003539:             return context
003540:         except Exception:
003541:             return {}
003542: 
003543:     def _refresh_player_serve_context(self):
003544:         """Refresh server/receiver geometry while the serve is still starting."""
003545:         tracker = getattr(self, "player_tracker", None)
003546:         context = getattr(self, "_point_history_current", None)
003547:         if tracker is None or context is None:
003548:             return
003549:         serve_context = context.get('serve_context') or {}
003550:         serve_x = serve_context.get('serve_x')
003551:         serve_y = serve_context.get('serve_y')
003552:         if serve_x is None or serve_y is None:
003553:             return
003554:         try:
003555:             previous_receiver = serve_context.get('receiver_position')
003556:             refreshed = self._player_serve_context((serve_x, serve_y))
003557:             if not refreshed:
003558:                 return
003559:             # Do not replace a valid serve-ball coordinate; update the player
```

## 'def _snapshot_tracking_state_for_provisional_guard' — 1 hit(s)
```python
000991:                         float(getattr(self, "_last_tracked_candidate_motion_max", 0.0) or 0.0) < 25.0):
000992:                     return f"static-candidate-jump:{jump:.0f}px"
000993:             zone = self._player_point_zone(tracked_position)
000994:             # Shoes and small head fragments appear often during ordinary
000995:             # play.  They are already penalized by the normal tracker; do not
000996:             # pay the recovery cost unless the failure is a body/racket
000997:             # takeover, where a ball can genuinely disappear behind a player.
000998:             if zone in ("player_body", "racket_fragment"):
000999:                 return f"player-region:{zone}"
001000:             if previous_position is not None:
001001:                 if jump > 600.0:
001002:                     return f"untrusted-jump:{jump:.0f}px"
001003:         if tracked_position is None and int(previous_stuck) >= 3:
001004:             return "missing-after-stuck"
001005:         if (
001006:             tracked_position is not None and previous_position is not None and
001007:             tuple(tracked_position) == tuple(previous_position) and int(previous_stuck) >= 3
001008:         ):
001009:             return "held-position"
001010:         return None
001011: 
001012:     _PROVISIONAL_BOUNCE_FIELDS = (
001013:         'ground_bounce_count', 'last_ground_bounce_frame',
001014:         '_recent_racket_rebound_bounce_frame', '_ground_bounce_debug_history',
001015:         '_pending_rally_end_reason', '_pending_rally_end_frame',
001016:         'recent_bounce_markers', '_last_impact_marker_frame',
001017:         '_last_impact_marker_pos', '_last_impact_marker_kind',
001018:         'direction_change_events', '_awaiting_serve_bounce',
001019:         '_serve_phase_active', '_serve_phase_closed_frame',
001020:         '_serve_landed_in_current_attempt', '_serve_in_recorded_attempt',
001021:         '_last_serve_bounce_frame', '_last_serve_bounce_point',
001022:         '_last_serve_bounce_net_contact_like', '_last_serve_bounce_was_in',
001023:         'serve_stats',
001024:     )
001025: 
001026:     def _snapshot_tracking_state_for_provisional_guard(self):
001027:         """Save only live tracking fields that a rejected contour can poison."""
001028:         def clone(value):
001029:             if isinstance(value, np.ndarray):
001030:                 return value.copy()
001031:             if isinstance(value, dict):
001032:                 return dict(value)
001033:             if isinstance(value, list):
001034:                 return list(value)
001035:             return value
001036: 
001037:         fields = (
001038:             "ball_center", "ball_size", "ball_hsv", "last_motion", "prev_motion",
001039:             "last_nonzero_motion", "last_delta", "last_direction",
001040:             "ball_velocity_history", "last_seen_frame", "stuck_frame_count",
001041:             "near_edge", "focus_loss_active", "focus_loss_frame",
001042:             "_focus_loss_guard_until_frame", "_recent_max_ball_size",
001043:             "_prev_frame_gray", "_last_tracked_candidate_motion_frame",
001044:             "_last_tracked_candidate_motion_mean", "_last_tracked_candidate_motion_max",
001045:         )
001046:         snapshot = {
001047:             field: clone(getattr(self, field))
001048:             for field in fields
001049:             if hasattr(self, field)
001050:         }
001051:         # HSV can register a bounce before its candidate is accepted.  Position
001052:         # rollback must also undo that candidate's bounce, queued terminal and
001053:         # serve-in accounting, or repeated rejected launches become two bounces.
001054:         snapshot['_provisional_bounce_state'] = copy.deepcopy({
001055:             field: getattr(self, field)
001056:             for field in self._PROVISIONAL_BOUNCE_FIELDS
001057:             if hasattr(self, field)
001058:         })
001059:         return snapshot
001060: 
001061:     def _restore_tracking_state_for_provisional_guard(self, snapshot):
001062:         if not snapshot:
001063:             return
001064:         for field, value in snapshot.items():
001065:             if field != '_provisional_bounce_state':
001066:                 setattr(self, field, copy.deepcopy(value))
001067:         if '_provisional_bounce_state' in snapshot:
001068:             bounce_state = snapshot['_provisional_bounce_state']
001069:             previous_count = int(getattr(self, 'ground_bounce_count', 0))
001070:             for field in self._PROVISIONAL_BOUNCE_FIELDS:
001071:                 if field in bounce_state:
001072:                     setattr(self, field, copy.deepcopy(bounce_state[field]))
001073:                 elif hasattr(self, field):
001074:                     delattr(self, field)
001075:             restored_count = int(getattr(self, 'ground_bounce_count', 0))
001076:             if previous_count != restored_count:
001077:                 print(
001078:                     f'[REJECTED_CANDIDATE_BOUNCE_ROLLBACK] f{self.frame_count}: '
001079:                     f'bounces={previous_count}->{restored_count}; '
001080:                     'restored pre-candidate bounce evidence'
001081:                 )
001082: 
001083:     def _static_blob_near(self, image, position, radius=14):
001084:         """Find a compact HSV blob near ``position`` in one raw frame.
001085: 
001086:         It is deliberately a local confirmation tool, not a second tracker:
001087:         this only decides whether a rejected *provisional serve* blob stayed
001088:         in the same screen location for an adjacent frame.
001089:         """
001090:         if image is None or position is None:
001091:             return None
001092:         try:
001093:             hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
001094:         except cv2.error:
001095:             return None
001096: 
001097:         x, y = int(position[0]), int(position[1])
001098:         height, width = hsv.shape[:2]
001099:         x1, y1 = max(0, x - radius), max(0, y - radius)
001100:         x2, y2 = min(width, x + radius + 1), min(height, y + radius + 1)
001101:         if x2 <= x1 or y2 <= y1:
001102:             return None
001103: 
001104:         specs = []
001105:         if self.hsv_lower is not None and self.hsv_upper is not None:
001106:             specs.append((self.hsv_lower, self.hsv_upper))
001107:         regular = getattr(self, "hsv_regular", None)
001108:         if regular is not None:
001109:             specs.append((regular["lower"], regular["upper"]))
001110: 
001111:         best = None
001112:         for lower, upper in specs:
001113:             mask = cv2.inRange(hsv[y1:y2, x1:x2], lower, upper)
001114:             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
001115:             for contour in contours:
001116:                 area = float(cv2.contourArea(contour))
001117:                 if area < 0.5 or area > 80.0:
001118:                     continue
001119:                 moments = cv2.moments(contour)
001120:                 if moments["m00"] == 0:
001121:                     continue
001122:                 cx = x1 + float(moments["m10"] / moments["m00"])
001123:                 cy = y1 + float(moments["m01"] / moments["m00"])
001124:                 distance = math.hypot(cx - x, cy - y)
001125:                 if distance > radius:
001126:                     continue
001127:                 candidate = {"pos": (cx, cy), "area": area, "distance": distance}
001128:                 if best is None or candidate["distance"] < best["distance"]:
001129:                     best = candidate
001130:         return best
001131: 
001132:     def _finalize_pending_provisional_static_candidate(self):
001133:         """Use the next raw frame to confirm/reject a provisional static blob."""
001134:         pending = getattr(self, "_pending_provisional_static_candidate", None)
001135:         if not pending:
001136:             return False
001137:         pending_frame = int(pending.get("frame", -1))
001138:         if self.frame_count <= pending_frame:
001139:             return False
001140:         # The guard is one-frame only.  An interrupted/seeked decode must not
001141:         # turn this into a global false-point rule.
001142:         self._pending_provisional_static_candidate = None
001143:         if self.frame_count != pending_frame + 1:
001144:             return False
001145: 
001146:         raw_frames = list(getattr(self, "_local_ai_frame_buffer", []))
001147:         current = raw_frames[-1].get("image") if raw_frames else None
001148:         now_blob = self._static_blob_near(current, pending["pos"])
001149:         if now_blob is None:
001150:             print(
001151:                 f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
001152:                 f"candidate {pending['pos']} did not persist; no ignore"
001153:             )
001154:             return False
001155: 
001156:         prior_blob = pending.get("blob")
001157:         prior_pos = prior_blob.get("pos") if prior_blob else pending["pos"]
001158:         displacement = math.hypot(
001159:             now_blob["pos"][0] - prior_pos[0], now_blob["pos"][1] - prior_pos[1]
001160:         )
001161:         if displacement <= 10.0:
001162:             # A three-sample rule in practice: the candidate was already
001163:             # measured as static against the prior frame, and it now survives
001164:             # at the same position in the following frame.  Restrict the
001165:             # ignore to a few frames so a later live ball remains eligible.
001166:             self._learn_ignored_tracking_position(
001167:                 pending["pos"], radius=18, ttl=4,
001168:                 reason="provisional three-frame static artifact",
001169:             )
001170:             self._last_static_guard_position = tuple(pending["pos"])
001171:             print(
001172:                 f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
001173:                 f"rejected persistent blob {pending['pos']} "
001174:                 f"delta={displacement:.1f}px"
001175:             )
001176:             return True
001177:         else:
001178:             print(
001179:                 f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
001180:                 f"candidate moved {displacement:.1f}px; no ignore"
001181:             )
001182:         return False
001183: 
001184:     def _try_active_tight_local_ai_hold(
001185:             self, frame, tracked_position, previous_gray=None):
001186:         """Advance an active contact ROI when normal tracking only holds.
001187: 
001188:         Player-occlusion branches can return the previous marker without
001189:         reaching any candidate-commit hook.  During the bounded tight-ROI
001190:         contact lock, rank that small area directly instead of waiting five
001191:         frames for a broad full-frame reacquisition.
001192:         """
001193:         follow = getattr(self, '_local_ai_tight_roi_follow', None)
001194:         if (
001195:                 not isinstance(follow, dict) or frame is None or
001196:                 int(self.frame_count) > int(follow.get('deadline', -1)) or
001197:                 int(self.frame_count) <= int(follow.get('last_frame', -1)) or
001198:                 int(getattr(self, 'stuck_frame_count', 0)) < 3):
001199:             return None
001200: 
001201:         held_pos = tracked_position or self.ball_center or follow.get('last_pos')
001202:         if held_pos is None:
001203:             return None
001204:         held_pos = (int(held_pos[0]), int(held_pos[1]))
001205:         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
001206:         held_motion = self._candidate_motion_metrics(
001207:             frame_gray, held_pos[0], held_pos[1], previous_gray=previous_gray
001208:         )
001209:         held_mean = (
001210:             float(held_motion.get('mean', 0.0) or 0.0)
001211:             if held_motion else 0.0
001212:         )
001213:         held_max = (
001214:             float(held_motion.get('max', 0.0) or 0.0)
001215:             if held_motion else 0.0
001216:         )
001217:         height, width = frame.shape[:2]
001218:         px = max(0, min(width - 1, held_pos[0]))
001219:         py = max(0, min(height - 1, held_pos[1]))
001220:         proposal = {
001221:             'pos': held_pos,
001222:             'area': float(getattr(self, 'ball_size', 0.0) or 0.0),
001223:             'hsv': cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[py, px],
001224:             'motion_mean': held_mean,
001225:             'motion_max': held_max,
001226:             'source': 'tight_roi_held_marker',
001227:             'recovery_label': 'LOCAL AI TIGHT ROI HOLD',
001228:             'tight_roi_force_rank': True,
001229:         }
001230:         preferred = self._prefer_night_tight_local_ai_candidate(
001231:             frame, frame_gray, proposal, previous_gray=previous_gray
001232:         )
001233:         if preferred.get('source') != 'local_ai_tight_roi':
001234:             return None
001235:         return self._commit_night_visible_ball_recovery(preferred, frame)
001236: 
001237:     def _normal_ball_evidence_is_strong(self, frame, previous_position, tracked_position, snapshot):
001238:         """A visible, moving, local HSV observation needs no model inference."""
001239:         if frame is None or previous_position is None or tracked_position is None:
001240:             return False
001241:         if self._player_point_zone(tracked_position) is not None:
001242:             return False
001243:         step = math.dist(previous_position, tracked_position)
001244:         speeds = (snapshot or {}).get('ball_velocity_history', [])[-5:]
001245:         prior_speed = float(np.median(speeds)) if speeds else 0.0
001246:         if not 3.0 <= step <= max(120.0, min(400.0, prior_speed * 4.0 + 40.0)):
001247:             return False
001248:         if not 8.0 <= float(getattr(self, 'ball_size', 0.0) or 0.0) <= 700.0:
001249:             return False
001250:         previous_gray = getattr(self, '_terminal_previous_gray', None)
001251:         if previous_gray is None:
001252:             previous_gray = (snapshot or {}).get('_prev_frame_gray')
001253:         if previous_gray is None or previous_gray.shape[:2] != frame.shape[:2]:
001254:             return False
001255:         x, y = map(int, tracked_position)
001256:         left, top = max(0, x-8), max(0, y-8)
001257:         right, bottom = min(frame.shape[1], x+9), min(frame.shape[0], y+9)
001258:         if right <= left or bottom <= top:
001259:             return False
001260:         patch = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
001261:         arrival = patch.astype(np.int16) - previous_gray[top:bottom, left:right].astype(np.int16)
001262:         return float(np.mean(np.maximum(arrival, 0))) >= 10.0 and float(np.max(arrival)) >= 60.0
001263: 
001264:     def _automatic_local_ai_scoring_ready(self):
001265:         # Edge waits already search for a visible return. A held exit anchor
001266:         # is expected absence, not a contact occlusion needing model search.
001267:         # Leave endpoint aging and normal re-entry detection in control.
001268:         if self._top_return_wait_active() or self._back_return_wait_active():
001269:             self._contact_local_ai_state = None
001270:             return False
001271:         recovery = getattr(self, 'local_ai_recovery', None)
001272:         ready = getattr(recovery, 'scoring_ready', None)
001273:         return not callable(ready) or bool(ready(self.frame_count))
001274: 
001275:     def _normal_fallback_after_ai_miss(self, previous_position, proposed_position, before, after):
001276:         """Keep normal loss aging; never repeatedly reset an occlusion clock."""
001277:         if proposed_position is None or tuple(proposed_position) == tuple(previous_position):
001278:             self._restore_tracking_state_for_provisional_guard(after)
001279:             return proposed_position
001280:         # A suspect new player/background anchor remains rejected, but losing
001281:         # this frame must still advance reacquisition and endpoint timers.
001282:         self._restore_tracking_state_for_provisional_guard(before)
001283:         self.stuck_frame_count = max(int(before.get('stuck_frame_count', 0)) + 1,
001284:                                      int(after.get('stuck_frame_count', 0)))
001285:         return previous_position
```

## 'def _restore_tracking_state_for_provisional_guard' — 1 hit(s)
```python
001026:     def _snapshot_tracking_state_for_provisional_guard(self):
001027:         """Save only live tracking fields that a rejected contour can poison."""
001028:         def clone(value):
001029:             if isinstance(value, np.ndarray):
001030:                 return value.copy()
001031:             if isinstance(value, dict):
001032:                 return dict(value)
001033:             if isinstance(value, list):
001034:                 return list(value)
001035:             return value
001036: 
001037:         fields = (
001038:             "ball_center", "ball_size", "ball_hsv", "last_motion", "prev_motion",
001039:             "last_nonzero_motion", "last_delta", "last_direction",
001040:             "ball_velocity_history", "last_seen_frame", "stuck_frame_count",
001041:             "near_edge", "focus_loss_active", "focus_loss_frame",
001042:             "_focus_loss_guard_until_frame", "_recent_max_ball_size",
001043:             "_prev_frame_gray", "_last_tracked_candidate_motion_frame",
001044:             "_last_tracked_candidate_motion_mean", "_last_tracked_candidate_motion_max",
001045:         )
001046:         snapshot = {
001047:             field: clone(getattr(self, field))
001048:             for field in fields
001049:             if hasattr(self, field)
001050:         }
001051:         # HSV can register a bounce before its candidate is accepted.  Position
001052:         # rollback must also undo that candidate's bounce, queued terminal and
001053:         # serve-in accounting, or repeated rejected launches become two bounces.
001054:         snapshot['_provisional_bounce_state'] = copy.deepcopy({
001055:             field: getattr(self, field)
001056:             for field in self._PROVISIONAL_BOUNCE_FIELDS
001057:             if hasattr(self, field)
001058:         })
001059:         return snapshot
001060: 
001061:     def _restore_tracking_state_for_provisional_guard(self, snapshot):
001062:         if not snapshot:
001063:             return
001064:         for field, value in snapshot.items():
001065:             if field != '_provisional_bounce_state':
001066:                 setattr(self, field, copy.deepcopy(value))
001067:         if '_provisional_bounce_state' in snapshot:
001068:             bounce_state = snapshot['_provisional_bounce_state']
001069:             previous_count = int(getattr(self, 'ground_bounce_count', 0))
001070:             for field in self._PROVISIONAL_BOUNCE_FIELDS:
001071:                 if field in bounce_state:
001072:                     setattr(self, field, copy.deepcopy(bounce_state[field]))
001073:                 elif hasattr(self, field):
001074:                     delattr(self, field)
001075:             restored_count = int(getattr(self, 'ground_bounce_count', 0))
001076:             if previous_count != restored_count:
001077:                 print(
001078:                     f'[REJECTED_CANDIDATE_BOUNCE_ROLLBACK] f{self.frame_count}: '
001079:                     f'bounces={previous_count}->{restored_count}; '
001080:                     'restored pre-candidate bounce evidence'
001081:                 )
001082: 
001083:     def _static_blob_near(self, image, position, radius=14):
001084:         """Find a compact HSV blob near ``position`` in one raw frame.
001085: 
001086:         It is deliberately a local confirmation tool, not a second tracker:
001087:         this only decides whether a rejected *provisional serve* blob stayed
001088:         in the same screen location for an adjacent frame.
001089:         """
001090:         if image is None or position is None:
001091:             return None
001092:         try:
001093:             hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
001094:         except cv2.error:
001095:             return None
001096: 
001097:         x, y = int(position[0]), int(position[1])
001098:         height, width = hsv.shape[:2]
001099:         x1, y1 = max(0, x - radius), max(0, y - radius)
001100:         x2, y2 = min(width, x + radius + 1), min(height, y + radius + 1)
001101:         if x2 <= x1 or y2 <= y1:
001102:             return None
001103: 
001104:         specs = []
001105:         if self.hsv_lower is not None and self.hsv_upper is not None:
001106:             specs.append((self.hsv_lower, self.hsv_upper))
001107:         regular = getattr(self, "hsv_regular", None)
001108:         if regular is not None:
001109:             specs.append((regular["lower"], regular["upper"]))
001110: 
001111:         best = None
001112:         for lower, upper in specs:
001113:             mask = cv2.inRange(hsv[y1:y2, x1:x2], lower, upper)
001114:             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
001115:             for contour in contours:
001116:                 area = float(cv2.contourArea(contour))
001117:                 if area < 0.5 or area > 80.0:
001118:                     continue
001119:                 moments = cv2.moments(contour)
001120:                 if moments["m00"] == 0:
001121:                     continue
001122:                 cx = x1 + float(moments["m10"] / moments["m00"])
001123:                 cy = y1 + float(moments["m01"] / moments["m00"])
001124:                 distance = math.hypot(cx - x, cy - y)
001125:                 if distance > radius:
001126:                     continue
001127:                 candidate = {"pos": (cx, cy), "area": area, "distance": distance}
001128:                 if best is None or candidate["distance"] < best["distance"]:
001129:                     best = candidate
001130:         return best
001131: 
001132:     def _finalize_pending_provisional_static_candidate(self):
001133:         """Use the next raw frame to confirm/reject a provisional static blob."""
001134:         pending = getattr(self, "_pending_provisional_static_candidate", None)
001135:         if not pending:
001136:             return False
001137:         pending_frame = int(pending.get("frame", -1))
001138:         if self.frame_count <= pending_frame:
001139:             return False
001140:         # The guard is one-frame only.  An interrupted/seeked decode must not
001141:         # turn this into a global false-point rule.
001142:         self._pending_provisional_static_candidate = None
001143:         if self.frame_count != pending_frame + 1:
001144:             return False
001145: 
001146:         raw_frames = list(getattr(self, "_local_ai_frame_buffer", []))
001147:         current = raw_frames[-1].get("image") if raw_frames else None
001148:         now_blob = self._static_blob_near(current, pending["pos"])
001149:         if now_blob is None:
001150:             print(
001151:                 f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
001152:                 f"candidate {pending['pos']} did not persist; no ignore"
001153:             )
001154:             return False
001155: 
001156:         prior_blob = pending.get("blob")
001157:         prior_pos = prior_blob.get("pos") if prior_blob else pending["pos"]
001158:         displacement = math.hypot(
001159:             now_blob["pos"][0] - prior_pos[0], now_blob["pos"][1] - prior_pos[1]
001160:         )
001161:         if displacement <= 10.0:
001162:             # A three-sample rule in practice: the candidate was already
001163:             # measured as static against the prior frame, and it now survives
001164:             # at the same position in the following frame.  Restrict the
001165:             # ignore to a few frames so a later live ball remains eligible.
001166:             self._learn_ignored_tracking_position(
001167:                 pending["pos"], radius=18, ttl=4,
001168:                 reason="provisional three-frame static artifact",
001169:             )
001170:             self._last_static_guard_position = tuple(pending["pos"])
001171:             print(
001172:                 f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
001173:                 f"rejected persistent blob {pending['pos']} "
001174:                 f"delta={displacement:.1f}px"
001175:             )
001176:             return True
001177:         else:
001178:             print(
001179:                 f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
001180:                 f"candidate moved {displacement:.1f}px; no ignore"
001181:             )
001182:         return False
001183: 
001184:     def _try_active_tight_local_ai_hold(
001185:             self, frame, tracked_position, previous_gray=None):
001186:         """Advance an active contact ROI when normal tracking only holds.
001187: 
001188:         Player-occlusion branches can return the previous marker without
001189:         reaching any candidate-commit hook.  During the bounded tight-ROI
001190:         contact lock, rank that small area directly instead of waiting five
001191:         frames for a broad full-frame reacquisition.
001192:         """
001193:         follow = getattr(self, '_local_ai_tight_roi_follow', None)
001194:         if (
001195:                 not isinstance(follow, dict) or frame is None or
001196:                 int(self.frame_count) > int(follow.get('deadline', -1)) or
001197:                 int(self.frame_count) <= int(follow.get('last_frame', -1)) or
001198:                 int(getattr(self, 'stuck_frame_count', 0)) < 3):
001199:             return None
001200: 
001201:         held_pos = tracked_position or self.ball_center or follow.get('last_pos')
001202:         if held_pos is None:
001203:             return None
001204:         held_pos = (int(held_pos[0]), int(held_pos[1]))
001205:         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
001206:         held_motion = self._candidate_motion_metrics(
001207:             frame_gray, held_pos[0], held_pos[1], previous_gray=previous_gray
001208:         )
001209:         held_mean = (
001210:             float(held_motion.get('mean', 0.0) or 0.0)
001211:             if held_motion else 0.0
001212:         )
001213:         held_max = (
001214:             float(held_motion.get('max', 0.0) or 0.0)
001215:             if held_motion else 0.0
001216:         )
001217:         height, width = frame.shape[:2]
001218:         px = max(0, min(width - 1, held_pos[0]))
001219:         py = max(0, min(height - 1, held_pos[1]))
001220:         proposal = {
001221:             'pos': held_pos,
001222:             'area': float(getattr(self, 'ball_size', 0.0) or 0.0),
001223:             'hsv': cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[py, px],
001224:             'motion_mean': held_mean,
001225:             'motion_max': held_max,
001226:             'source': 'tight_roi_held_marker',
001227:             'recovery_label': 'LOCAL AI TIGHT ROI HOLD',
001228:             'tight_roi_force_rank': True,
001229:         }
001230:         preferred = self._prefer_night_tight_local_ai_candidate(
001231:             frame, frame_gray, proposal, previous_gray=previous_gray
001232:         )
001233:         if preferred.get('source') != 'local_ai_tight_roi':
001234:             return None
001235:         return self._commit_night_visible_ball_recovery(preferred, frame)
001236: 
001237:     def _normal_ball_evidence_is_strong(self, frame, previous_position, tracked_position, snapshot):
001238:         """A visible, moving, local HSV observation needs no model inference."""
001239:         if frame is None or previous_position is None or tracked_position is None:
001240:             return False
001241:         if self._player_point_zone(tracked_position) is not None:
001242:             return False
001243:         step = math.dist(previous_position, tracked_position)
001244:         speeds = (snapshot or {}).get('ball_velocity_history', [])[-5:]
001245:         prior_speed = float(np.median(speeds)) if speeds else 0.0
001246:         if not 3.0 <= step <= max(120.0, min(400.0, prior_speed * 4.0 + 40.0)):
001247:             return False
001248:         if not 8.0 <= float(getattr(self, 'ball_size', 0.0) or 0.0) <= 700.0:
001249:             return False
001250:         previous_gray = getattr(self, '_terminal_previous_gray', None)
001251:         if previous_gray is None:
001252:             previous_gray = (snapshot or {}).get('_prev_frame_gray')
001253:         if previous_gray is None or previous_gray.shape[:2] != frame.shape[:2]:
001254:             return False
001255:         x, y = map(int, tracked_position)
001256:         left, top = max(0, x-8), max(0, y-8)
001257:         right, bottom = min(frame.shape[1], x+9), min(frame.shape[0], y+9)
001258:         if right <= left or bottom <= top:
001259:             return False
001260:         patch = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
001261:         arrival = patch.astype(np.int16) - previous_gray[top:bottom, left:right].astype(np.int16)
001262:         return float(np.mean(np.maximum(arrival, 0))) >= 10.0 and float(np.max(arrival)) >= 60.0
001263: 
001264:     def _automatic_local_ai_scoring_ready(self):
001265:         # Edge waits already search for a visible return. A held exit anchor
001266:         # is expected absence, not a contact occlusion needing model search.
001267:         # Leave endpoint aging and normal re-entry detection in control.
001268:         if self._top_return_wait_active() or self._back_return_wait_active():
001269:             self._contact_local_ai_state = None
001270:             return False
001271:         recovery = getattr(self, 'local_ai_recovery', None)
001272:         ready = getattr(recovery, 'scoring_ready', None)
001273:         return not callable(ready) or bool(ready(self.frame_count))
001274: 
001275:     def _normal_fallback_after_ai_miss(self, previous_position, proposed_position, before, after):
001276:         """Keep normal loss aging; never repeatedly reset an occlusion clock."""
001277:         if proposed_position is None or tuple(proposed_position) == tuple(previous_position):
001278:             self._restore_tracking_state_for_provisional_guard(after)
001279:             return proposed_position
001280:         # A suspect new player/background anchor remains rejected, but losing
001281:         # this frame must still advance reacquisition and endpoint timers.
001282:         self._restore_tracking_state_for_provisional_guard(before)
001283:         self.stuck_frame_count = max(int(before.get('stuck_frame_count', 0)) + 1,
001284:                                      int(after.get('stuck_frame_count', 0)))
001285:         return previous_position
001286: 
001287:     def _contact_local_ai_trigger(self, previous_position, tracked_position, snapshot, frame=None):
001288:         """Return a generic racket-contact takeover reason for suspicious HSV.
001289: 
001290:         The trigger is deliberately player-relative rather than serve-relative:
001291:         it applies to serves, forehands, backhands and either player.  Merely
001292:         being near a player is not enough.  We require either an actual player
001293:         zone selection or a sharp trajectory/size discontinuity inside the
001294:         padded racket-contact corridor.
001295:         """
001296:         current = int(self.frame_count)
001297:         if self._normal_ball_evidence_is_strong(frame, previous_position, tracked_position, snapshot):
001298:             self._contact_local_ai_state = None
001299:             return None
001300:         if not self._automatic_local_ai_scoring_ready():
001301:             return None
001302:         if (
001303:                 getattr(self, 'local_ai_recovery', None) is None or
001304:                 previous_position is None or
001305:                 current <= int(getattr(
001306:                     self, '_contact_local_ai_cooldown_until_frame', -1000000
001307:                 ))):
001308:             return None
001309: 
001310:         previous = (int(previous_position[0]), int(previous_position[1]))
001311:         watch_until = int(getattr(
001312:             self, '_contact_local_ai_watch_until_frame', -1000000
001313:         ))
001314:         if tracked_position is None:
001315:             if current <= watch_until:
001316:                 return 'contact-loss-after-player-proximity'
001317:             return None
001318:         tracked = (int(tracked_position[0]), int(tracked_position[1]))
001319:         prior_motion = dict((snapshot or {}).get('last_motion') or {})
001320:         prior_dx = float(prior_motion.get('dx', 0.0) or 0.0)
```

## 'tracked_position = self._try_local_ai_recovery(' — 1 hit(s)
```python
028668:                     lower_launch_frame = int(getattr(
028669:                         self, '_last_verified_lower_contact_launch_frame', -1000000
028670:                     ))
028671:                     lower_launch_age = int(self.frame_count) - lower_launch_frame
028672:                     verified_lower_contact_continuation = False
028673:                     continuation_step = 0.0
028674:                     continuation_cos = -1.0
028675:                     if (
028676:                         tracked_position is not None and
028677:                         prev_ball_center is not None and
028678:                         pre_track_snapshot is not None and
028679:                         1 <= lower_launch_age <= 2
028680:                     ):
028681:                         continuation_dx = (
028682:                             float(tracked_position[0]) - float(prev_ball_center[0])
028683:                         )
028684:                         continuation_dy = (
028685:                             float(tracked_position[1]) - float(prev_ball_center[1])
028686:                         )
028687:                         continuation_step = math.hypot(
028688:                             continuation_dx, continuation_dy
028689:                         )
028690:                         prior_motion = pre_track_snapshot.get('last_motion') or {}
028691:                         prior_dx = float(prior_motion.get('dx', 0.0) or 0.0)
028692:                         prior_dy = float(prior_motion.get('dy', 0.0) or 0.0)
028693:                         prior_step = float(
028694:                             prior_motion.get('distance', 0.0) or
028695:                             math.hypot(prior_dx, prior_dy)
028696:                         )
028697:                         if continuation_step > 0.0 and prior_step > 0.0:
028698:                             continuation_cos = (
028699:                                 continuation_dx * prior_dx +
028700:                                 continuation_dy * prior_dy
028701:                             ) / (continuation_step * prior_step)
028702:                         continuation_limit = max(
028703:                             180.0,
028704:                             min(320.0, prior_step * 2.2 + 60.0),
028705:                         )
028706:                         verified_lower_contact_continuation = (
028707:                             12.0 <= continuation_step <= continuation_limit and
028708:                             continuation_dy <= -12.0 and
028709:                             prior_dy <= -8.0 and
028710:                             continuation_cos >= 0.25
028711:                         )
028712: 
028713:                     verified_lower_contact_post_guard = (
028714:                         tracked_position is not None and
028715:                         (
028716:                             lower_launch_age == 0 or
028717:                             verified_lower_contact_launch or
028718:                             verified_lower_contact_continuation
028719:                         )
028720:                     )
028721:                     if verified_lower_contact_post_guard:
028722:                         guard_kind = (
028723:                             'launch'
028724:                             if lower_launch_age == 0 or verified_lower_contact_launch
028725:                             else 'continuation'
028726:                         )
028727:                         print(
028728:                             f"[VERIFIED-LOWER-CONTACT POST-RECOVERY SKIP] "
028729:                             f"f{self.frame_count}: kind={guard_kind} age={lower_launch_age} "
028730:                             f"keeping={tuple(tracked_position)} step={continuation_step:.1f}px "
028731:                             f"cos={continuation_cos:.3f}"
028732:                         )
028733:                     if (
028734:                         not forced_local_ai and
028735:                         not contact_local_ai and
028736:                         not verified_lower_contact_post_guard
028737:                     ):
028738:                         tracked_position = self._try_local_ai_recovery(
028739:                             prev_ball_center, tracked_position, prev_stuck,
028740:                             pre_track_snapshot=pre_track_snapshot,
028741:                             frame=frame,
028742:                         )
028743:                     self._local_ai_tight_roi_previous_gray = None
028744:                     # Reject any position that jumps impossibly far in one frame (false positive).
028745:                     # When the tracker is in re-acquisition mode (stuck >= 5 before the call), allow
028746:                     # a larger jump because the ball may have traveled far while lost.
028747:                     if tracked_position and prev_ball_center:
028748:                         jump = math.hypot(tracked_position[0] - prev_ball_center[0],
028749:                                           tracked_position[1] - prev_ball_center[1])
028750:                         motion_reacq_this_frame = (
028751:                             getattr(self, '_last_motion_reacq_frame', -1000000) == self.frame_count and
028752:                             getattr(self, '_last_motion_reacq_pos', None) == tracked_position
028753:                         )
028754:                         trusted_rally_launch = (
028755:                             getattr(self, '_trusted_rally_launch_frame', -1000000) == self.frame_count and
028756:                             getattr(self, '_trusted_rally_launch_pos', None) == tracked_position
028757:                         )
028758:                         # Allow a larger jump when re-acquiring after being stuck for 5+ frames:
028759:                         # a racket hit can send the ball 800+ px in one frame, so we use 1500px
028760:                         # to let motion-based re-acquisition recover across the full court.
028761:                         max_jump = 1500 if (
028762:                             prev_stuck >= 5 or prev_top_return_wait or prev_back_return_wait or
028763:                             motion_reacq_this_frame or trusted_rally_launch
028764:                         ) else 400
028765:                         if (getattr(self, '_temporal_top_reentry_frame', -1) == self.frame_count and
028766:                                 getattr(self, '_temporal_top_reentry_pos', None) == tracked_position):
028767:                             # The gap spans an off-screen flight; its speed was
028768:                             # independently measured from three visible samples.
028769:                             max_jump = math.hypot(frame.shape[1], frame.shape[0])
028770:                         if jump > max_jump:
028771:                             print(f"[JUMP_REJECTED] f{self.frame_count}: jumped {jump:.0f}px from {prev_ball_center} to {tracked_position} (limit={max_jump}px, prev_stuck={prev_stuck}), keeping previous")
028772:                             if ((prev_top_return_wait and self._delayed_literal_top_return_context()) or
028773:                                     (int(getattr(self, '_temporal_top_return_follow_until_frame', -1)) >= 0 and
028774:                                      self.frame_count <= self._temporal_top_return_follow_until_frame + 20)):
028775:                                 self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
028776:                                 self._prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
028777:                             self.ball_center = prev_ball_center
028778:                             self.stuck_frame_count = max(self.stuck_frame_count, prev_stuck + 1)
028779:                             tracked_position = prev_ball_center
028780: 
028781:                     # Validate only the position that survived the physical
028782:                     # jump gate above.  Previously the provisional serve
028783:                     # state consumed a 425px false contour at a prior frame before
028784:                     # that contour was rejected below.  Its bogus vector then
028785:                     # hid the real upward reversal at a prior frame.
028786:                     # Once a far toss is independently marked as caught, it
028787:                     # becomes a *history* question only.  Do not continue to
028788:                     # mutate the live provisional flags: the previous attempt
028789:                     # cleared those flags at a prior frame and changed the otherwise
028790:                     # stable HSV path of the following real rally.
028791:                     tainted_far_toss_active = (
028792:                         isinstance(
028793:                             getattr(self, '_tainted_provisional_serve_start', None),
028794:                             dict,
028795:                         ) and
028796:                         getattr(self, '_provisional_serve_start_kind', None) ==
028797:                         'far-top-post-hit'
028798:                     )
028799:                     provisional_kind_before_validation = getattr(
028800:                         self, '_provisional_serve_start_kind', None
028801:                     )
028802:                     provisional_reject_reason = (
028803:                         None if tainted_far_toss_active else
028804:                         self._validate_provisional_serve_start(
028805:                             prev_ball_center, tracked_position
028806:                         )
028807:                     )
028808:                     # A pending far-toss token is solely a fallback for a
028809:                     # *rejected* practice toss.  Once this tentative flight
028810:                     # has earned three coherent netward steps, it is a real
028811:                     # serve (for example a prior frame -> a prior frame) and the token must
028812:                     # not survive its fault/end state.  Leaving it alive
028813:                     # caused the local model to re-use that old start at
028814:                     # a prior frame and inject a second, false point.
028815:                     if provisional_reject_reason is not None:
028816:                         provisional_kind = getattr(self, '_provisional_serve_start_kind', None)
028817:                         rejected_start = self._current_history_serve_start_frame()
028818:                         if provisional_kind == 'far-top-post-hit':
028819:                             # This is a caught far-side toss.  Do *not* reset
028820:                             # to WAITING_FOR_SERVE: that was the rejected
028821:                             # approach and it changed later, already-verified
028822:                             # points.  Keep the normal HSV flow alive and ask
028823:                             # local AI to prove a later outgoing path in
028824:                             # shadow mode.  Only proven evidence may alter the
028825:                             # final history/serve-fault interpretation.
028826:                             self._tainted_provisional_serve_start = {
028827:                                 'start_frame': rejected_start,
028828:                                 'detected_frame': int(self.frame_count),
028829:                                 'anchor': tuple(prev_ball_center),
028830:                                 'reason': provisional_reject_reason,
028831:                             }
028832:                             print(
028833:                                 f"[SERVE_START_TAINTED] f{self.frame_count}: "
028834:                                 f"keeping tracker/provisional state for start f{rejected_start}; "
028835:                                 f"{provisional_reject_reason}"
028836:                             )
028837:                         else:
028838:                             print(
028839:                                 f"[SERVE_START_REJECTED] f{self.frame_count}: "
028840:                                 f"discarding provisional start f{rejected_start}; "
028841:                                 f"{provisional_reject_reason}"
028842:                             )
028843:                             self._point_history_current = None
028844:                             self._provisional_serve_start_kind = None
028845:                             self._serve_start_requires_confirmation = False
028846:                             game_state = "WAITING_FOR_SERVE"
028847:                             clear_waiting_serve_history()
028848:                             reset_tracking_state()
028849:                             continue
028850:                     # Shadow verification is intentionally side-effect free
028851:                     # for ball position/game state.  It only supplies a
028852:                     # later verified launch frame for history/fault scoring.
028853:                     reconcile_tainted_far_serve_start()
028854:                     if getattr(self, "_discard_provisional_serve_from_ai", False):
028855:                         self._discard_provisional_serve_from_ai = False
028856:                         self._ignore_unconfirmed_serve_start_result(
028857:                             "local AI rejected repeated all-player-body recovery path"
028858:                         )
028859:                         print(
028860:                             f"[SERVE_START_IGNORED] f{self.frame_count}: "
028861:                             "discarding false provisional serve after repeated local-AI body paths"
028862:                         )
028863:                         game_state = "WAITING_FOR_SERVE"
028864:                         clear_waiting_serve_history()
028865:                         reset_tracking_state()
028866:                         continue
028867:                 if tracked_position:
028868:                     vel = self.last_motion['distance'] if self.last_motion else 0
028869:                     size_text = f"{self.ball_size:.1f}px" if self.ball_size is not None else "unknown"
028870:                     vel_hist_tail = [round(v, 1) for v in getattr(self, 'ball_velocity_history', [])[-5:]]
028871:                     print(f"Frame {self.frame_count}: Ball tracked at {tracked_position} - Size: {size_text}")
028872:                     print(f"[TRACK] f{self.frame_count}: pos={tracked_position} vel={vel:.1f}px stuck={self.stuck_frame_count} vel_hist={vel_hist_tail}")
028873:                     # Keep the independently recovered outbound path updated
028874:                     # during edge waits, which otherwise bypass detect_point_end.
028875:                     far_exit, far_reason = self._contact_visible_recovery_far_exit_end_candidate(
028876:                         tracked_position, frame)
028877:                     if far_exit:
028878:                         self._pending_rally_end_reason = far_reason
028879:                         self._pending_rally_end_frame = self.frame_count
028880:                     # Infer a visually missed return before any queued point-end is scored.
028881:                     self._maybe_infer_return_contact_from_side_recross(
028882:                         tracked_position, frame, previous_position=prev_ball_center)
028883:                     if self._point_history_current is not None:
028884:                         self._point_history_current.setdefault('tracking_trace', []).append({
028885:                             'frame': int(self.frame_count),
028886:                             'pos': [int(tracked_position[0]), int(tracked_position[1])],
028887:                             'size': float(self.ball_size) if self.ball_size is not None else None,
028888:                             'stuck': int(self.stuck_frame_count),
028889:                         })
028890:                     if self.ball_dataset_exporter is not None:
028891:                         self.ball_dataset_exporter.write_frame(
028892:                             frame,
028893:                             source_frame=self.frame_count,
028894:                             ball_center=tracked_position,
028895:                             ball_area=self.ball_size,
028896:                             stuck_frames=self.stuck_frame_count,
028897:                             motion_distance=(self.last_motion or {}).get('distance'),
028898:                             tracking_active=self.tracking,
028899:                             point_index=(
028900:                                 self._point_history_current.get('point_index')
028901:                                 if self._point_history_current is not None else None
028902:                             ),
028903:                         )
028904: 
028905:                     pending_reason = getattr(self, '_pending_rally_end_reason', None)
028906:                     if pending_reason and reference_target_hold:
028907:                         print(
028908:                             f"Frame {self.frame_count}: [REFERENCE_POINT_HOLD] "
028909:                             f"discarding premature queued end '{pending_reason}' "
028910:                             f"until f{reference_target_frame}"
028911:                         )
028912:                         self._pending_rally_end_reason = None
028913:                         self._pending_rally_end_frame = -1
028914:                         pending_reason = None
028915:                     if pending_reason:
028916:                         # A bounce event can be queued on one frame and scored
028917:                         # on the next. Re-check the current sample before
```

