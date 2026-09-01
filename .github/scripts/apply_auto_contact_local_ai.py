from pathlib import Path

path = Path('interactive_ball_analyzer.py')
text = path.read_text(encoding='utf-8')


def replace_once(old: str, new: str, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{label}: expected 1 match, found {count}')
    text = text.replace(old, new, 1)


if '[CONTACT_LOCAL_AI_ARM]' in text:
    print('automatic contact Local AI already installed')
    raise SystemExit(0)

# Runtime state. Manual --force-local-ai-range remains independent and has priority.
replace_once(
    '''        self._force_local_ai_history = []\n        self._local_ai_frame_buffer = frame_buffer(12)\n''',
    '''        self._force_local_ai_history = []\n        # Generic racket-contact AI ownership.  This is armed automatically only\n        # when the normal HSV path becomes suspicious inside a tracked player's\n        # contact corridor.  It reuses the proven forced-AI tracker, but keeps a\n        # separate history so the debug flag and automatic mode never interfere.\n        self._contact_local_ai_state = None\n        self._contact_local_ai_cooldown_until_frame = -1000000\n        self._contact_local_ai_radius = 450.0\n        self._contact_local_ai_min_score = 0.985\n        self._local_ai_frame_buffer = frame_buffer(12)\n''',
    'contact state',
)

method_anchor = '''    def _force_local_ai_frame(self, frame, previous_position):\n'''
if text.count(method_anchor) != 1:
    raise RuntimeError(f'force method anchor: expected 1 match, found {text.count(method_anchor)}')

methods = r'''    def _contact_local_ai_trigger(self, previous_position, tracked_position, snapshot):
        """Return a generic racket-contact takeover reason for suspicious HSV.

        The trigger is deliberately player-relative rather than serve-relative:
        it applies to serves, forehands, backhands and either player.  Merely
        being near a player is not enough.  We require either an actual player
        zone selection or a sharp trajectory/size discontinuity inside the
        padded racket-contact corridor.
        """
        if (
                getattr(self, 'local_ai_recovery', None) is None or
                previous_position is None or tracked_position is None or
                int(self.frame_count) <= int(getattr(
                    self, '_contact_local_ai_cooldown_until_frame', -1000000
                ))):
            return None

        previous = (int(previous_position[0]), int(previous_position[1]))
        tracked = (int(tracked_position[0]), int(tracked_position[1]))
        prior_motion = dict((snapshot or {}).get('last_motion') or {})
        prior_dx = float(prior_motion.get('dx', 0.0) or 0.0)
        prior_dy = float(prior_motion.get('dy', 0.0) or 0.0)
        prior_speed = float(
            prior_motion.get('distance', math.hypot(prior_dx, prior_dy)) or 0.0
        )
        proposed_dx = float(tracked[0] - previous[0])
        proposed_dy = float(tracked[1] - previous[1])
        proposed_speed = math.hypot(proposed_dx, proposed_dy)

        predicted = (
            int(round(float(previous[0]) + prior_dx)),
            int(round(float(previous[1]) + prior_dy)),
        )
        prediction_error = math.hypot(
            float(tracked[0] - predicted[0]), float(tracked[1] - predicted[1])
        )

        angle_delta = 0.0
        if prior_speed >= 4.0 and proposed_speed >= 4.0:
            prior_angle = math.degrees(math.atan2(prior_dy, prior_dx))
            proposed_angle = math.degrees(math.atan2(proposed_dy, proposed_dx))
            raw_delta = abs(proposed_angle - prior_angle) % 360.0
            angle_delta = min(raw_delta, 360.0 - raw_delta)

        zone = self._player_point_zone(tracked)
        contact_near = any(
            self._point_in_player_contact_corridor(point)
            for point in (previous, tracked, predicted)
        )
        if zone is None and not contact_near:
            return None

        recent_speeds = [
            float(value) for value in (snapshot or {}).get('ball_velocity_history', [])[-5:]
            if value is not None
        ]
        recent_median = float(np.median(recent_speeds)) if recent_speeds else prior_speed
        jump_floor = max(70.0, recent_median * 1.8, prior_speed * 1.55)

        previous_size = float((snapshot or {}).get('ball_size', 0.0) or 0.0)
        proposed_size = float(getattr(self, 'ball_size', 0.0) or 0.0)
        size_ratio = (
            max(previous_size, proposed_size) / max(1.0, min(previous_size, proposed_size))
            if previous_size > 0.0 and proposed_size > 0.0 else 1.0
        )

        player_zones = (
            'player_head_hat', 'player_body', 'player_shoes', 'racket_fragment'
        )
        if zone in player_zones and max(prior_speed, proposed_speed) >= 12.0:
            return (
                f'player-zone:{zone}/step={proposed_speed:.0f}px/'
                f'angle={angle_delta:.0f}deg'
            )

        if (
                contact_near and proposed_speed >= jump_floor and
                (angle_delta >= 35.0 or prediction_error >= 50.0)):
            return (
                f'contact-jump:{proposed_speed:.0f}px>={jump_floor:.0f}/'
                f'angle={angle_delta:.0f}deg/pred={prediction_error:.0f}px'
            )

        if (
                contact_near and prior_speed >= 18.0 and proposed_speed >= 18.0 and
                angle_delta >= 70.0 and prediction_error >= 30.0):
            return (
                f'contact-turn:{angle_delta:.0f}deg/'
                f'pred={prediction_error:.0f}px'
            )

        if (
                contact_near and proposed_speed >= 35.0 and size_ratio >= 2.6 and
                prediction_error >= 35.0):
            return (
                f'contact-size-jump:{size_ratio:.1f}x/'
                f'pred={prediction_error:.0f}px'
            )
        return None

    def _run_contact_local_ai_frame(self, frame, previous_position, arm_reason=None):
        """Let Local AI own a short generic racket-contact sequence.

        The implementation intentionally reuses the already validated
        --force-local-ai-range tracker: player/head/shoe/racket candidates are
        excluded before scoring, the AI-selected point becomes the next anchor,
        and trajectory prediction starts after two accepted AI positions.
        """
        recovery = getattr(self, 'local_ai_recovery', None)
        if recovery is None or frame is None or previous_position is None:
            return None

        current = int(self.frame_count)
        state = getattr(self, '_contact_local_ai_state', None)
        if arm_reason is not None and not isinstance(state, dict):
            state = {
                'start_frame': current,
                'deadline': current + 22,
                'history': [],
                'outside_count': 0,
                'reason': str(arm_reason),
            }
            self._contact_local_ai_state = state
            # Do not let an older tight-ROI experiment compete with contact
            # ownership.  The automatic owner is the stronger, player-excluding
            # path and will hand control back explicitly.
            self._local_ai_tight_roi_follow = None
            print(
                f"[CONTACT_LOCAL_AI_ARM] f{current}: anchor={tuple(previous_position)} "
                f"reason={arm_reason} radius={self._contact_local_ai_radius:.0f}px"
            )

        state = getattr(self, '_contact_local_ai_state', None)
        if not isinstance(state, dict):
            return None
        if current > int(state.get('deadline', -1)):
            print(
                f"[CONTACT_LOCAL_AI_EXPIRE] f{current}: "
                f"started=f{state.get('start_frame')} reason={state.get('reason')}"
            )
            self._contact_local_ai_state = None
            self._contact_local_ai_cooldown_until_frame = current + 3
            return None

        # Reuse the proven forced-AI implementation without exposing its state
        # to the user's manual debug range.
        saved_range = getattr(self, '_force_local_ai_range', None)
        saved_history = list(getattr(self, '_force_local_ai_history', []) or [])
        saved_radius = float(getattr(self, '_force_local_ai_radius', 450.0))
        saved_min_score = float(getattr(self, '_force_local_ai_min_score', 0.985))
        try:
            self._force_local_ai_range = (
                int(state['start_frame']), int(state['deadline'])
            )
            self._force_local_ai_history = list(state.get('history', []) or [])
            self._force_local_ai_radius = float(
                getattr(self, '_contact_local_ai_radius', 450.0)
            )
            self._force_local_ai_min_score = float(
                getattr(self, '_contact_local_ai_min_score', 0.985)
            )
            selected = self._force_local_ai_frame(frame, previous_position)
            state['history'] = list(getattr(self, '_force_local_ai_history', []) or [])
        finally:
            self._force_local_ai_range = saved_range
            self._force_local_ai_history = saved_history
            self._force_local_ai_radius = saved_radius
            self._force_local_ai_min_score = saved_min_score

        if selected is None:
            return None

        selected = (int(selected[0]), int(selected[1]))
        zone = self._player_point_zone(selected)
        inside_contact = self._point_in_player_contact_corridor(selected)
        age = current - int(state['start_frame'])
        if zone is None and not inside_contact:
            state['outside_count'] = int(state.get('outside_count', 0)) + 1
        else:
            state['outside_count'] = 0

        print(
            f"[CONTACT_LOCAL_AI] f{current}: selected={selected} age={age}f "
            f"outside={state['outside_count']} zone={zone}"
        )

        # Keep AI for several frames after contact.  This prevents an immediate
        # handback to HSV while its search anchor is still the racket/player.
        if age >= 5 and int(state.get('outside_count', 0)) >= 3:
            print(
                f"[CONTACT_LOCAL_AI_HANDOFF] f{current}: selected={selected} "
                f"clear of player/contact corridor for {state['outside_count']}f"
            )
            self._contact_local_ai_state = None
            self._contact_local_ai_cooldown_until_frame = current + 3
        else:
            self._contact_local_ai_state = state
        return selected

'''
text = text.replace(method_anchor, methods + method_anchor, 1)

# Manual forced-AI has first priority.  If no manual range owns the frame, an
# already-armed generic contact owner bypasses HSV.  Otherwise HSV gets one
# normal proposal; only a suspicious near-player contact proposal is rolled
# back and re-run through Local AI in the SAME frame.
old_flow = '''                    forced_local_ai = False\n                    tracked_position = self._force_local_ai_frame(frame, prev_ball_center)\n                    if tracked_position is not None:\n                        forced_local_ai = True\n                    else:\n                        tracked_position = self.track_ball_in_frame(frame)\n                    if self.local_ai_recovery is not None and self._local_ai_frame_buffer:\n                        self._local_ai_frame_buffer[-1]["normal_position"] = (\n                            tuple(tracked_position) if tracked_position is not None else None\n                        )\n                    self._debug_local_ai_shadow_frame(frame, prev_ball_center, tracked_position)\n                    if not forced_local_ai:\n                        tracked_position = self._try_local_ai_recovery(\n                            prev_ball_center, tracked_position, prev_stuck,\n                            pre_track_snapshot=pre_track_snapshot,\n                            frame=frame,\n                        )\n'''
new_flow = '''                    forced_local_ai = False\n                    contact_local_ai = False\n                    tracked_position = self._force_local_ai_frame(frame, prev_ball_center)\n                    if tracked_position is not None:\n                        forced_local_ai = True\n                    else:\n                        # Once armed, contact Local AI owns the next few frames\n                        # before HSV is allowed to mutate the anchor again.\n                        tracked_position = self._run_contact_local_ai_frame(\n                            frame, prev_ball_center\n                        )\n                        if tracked_position is not None:\n                            contact_local_ai = True\n                        else:\n                            tracked_position = self.track_ball_in_frame(frame)\n                            contact_reason = self._contact_local_ai_trigger(\n                                prev_ball_center, tracked_position, pre_track_snapshot\n                            )\n                            if contact_reason is not None:\n                                rejected_hsv = (\n                                    tuple(tracked_position)\n                                    if tracked_position is not None else None\n                                )\n                                self._restore_tracking_state_for_provisional_guard(\n                                    pre_track_snapshot\n                                )\n                                self._local_ai_tight_roi_previous_gray = (\n                                    pre_track_snapshot.get('_prev_frame_gray')\n                                )\n                                tracked_position = self._run_contact_local_ai_frame(\n                                    frame, prev_ball_center, arm_reason=contact_reason\n                                )\n                                if tracked_position is not None:\n                                    contact_local_ai = True\n                                    print(\n                                        f"[CONTACT_LOCAL_AI_REPLACE] f{self.frame_count}: "\n                                        f"HSV={rejected_hsv} -> AI={tracked_position}"\n                                    )\n                                else:\n                                    # A failed AI arm must not leave state partially\n                                    # rolled back.  Hold the last trusted ball and retry\n                                    # on the next frame rather than accepting player HSV.\n                                    self.ball_center = tuple(prev_ball_center)\n                                    self.stuck_frame_count = max(\n                                        1, int(prev_stuck) + 1\n                                    )\n                                    tracked_position = tuple(prev_ball_center)\n                                    contact_local_ai = True\n                    if self.local_ai_recovery is not None and self._local_ai_frame_buffer:\n                        self._local_ai_frame_buffer[-1]["normal_position"] = (\n                            tuple(tracked_position) if tracked_position is not None else None\n                        )\n                    self._debug_local_ai_shadow_frame(frame, prev_ball_center, tracked_position)\n                    if not forced_local_ai and not contact_local_ai:\n                        tracked_position = self._try_local_ai_recovery(\n                            prev_ball_center, tracked_position, prev_stuck,\n                            pre_track_snapshot=pre_track_snapshot,\n                            frame=frame,\n                        )\n'''
replace_once(old_flow, new_flow, 'main contact ownership flow')

for marker in (
    '[CONTACT_LOCAL_AI_ARM]',
    '[CONTACT_LOCAL_AI]',
    '[CONTACT_LOCAL_AI_HANDOFF]',
    '[CONTACT_LOCAL_AI_REPLACE]',
    'contact-jump:',
):
    if marker not in text:
        raise RuntimeError(f'missing marker {marker}')

path.write_text(text, encoding='utf-8')
print('automatic generic contact Local AI patch applied')
