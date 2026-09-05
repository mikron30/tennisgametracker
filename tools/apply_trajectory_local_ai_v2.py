from pathlib import Path

path = Path('interactive_ball_analyzer.py')
text = path.read_text(encoding='utf-8')

old = """        trajectory_hotspot = getattr(self, '_trajectory_local_ai_hotspot', None)
        if (
                zone is None and not contact_near and not watch_active and
                isinstance(trajectory_hotspot, dict) and
                int(trajectory_hotspot.get('frame', -1)) == current):
            hotspot_motion_mean = float(trajectory_hotspot.get('motion_mean', 999.0) or 0.0)
            hotspot_motion_max = float(trajectory_hotspot.get('motion_max', 999.0) or 0.0)
            physical_event_grace = any(
                int(getattr(self, attr, 0) or 0) > 0
                for attr in (
                    '_serve_contact_grace_frames',
                    '_rally_contact_grace_frames',
                    '_ground_bounce_grace_frames',
                )
            )
            if (
                    not physical_event_grace and
                    prior_speed >= 14.0 and proposed_speed >= 14.0 and
                    angle_delta >= 50.0 and prediction_error >= 12.0 and
                    hotspot_motion_mean <= 3.0 and hotspot_motion_max <= 12.0):
                reason = (
                    f'trajectory-hotspot-turn:{angle_delta:.0f}deg/'
                    f'pred={prediction_error:.0f}px/'
                    f'motion={hotspot_motion_mean:.1f}/{hotspot_motion_max:.1f}'
                )
                print(
                    f'[TRAJECTORY_LOCAL_AI_TRIGGER] f{current}: '
                    f'previous={previous} normal={tracked} predicted={predicted} '
                    f'angle={angle_delta:.1f}deg pred_error={prediction_error:.1f}px '
                    f'motion={hotspot_motion_mean:.1f}/{hotspot_motion_max:.1f}'
                )
                return reason
"""

new = """        trajectory_hotspot = getattr(self, '_trajectory_local_ai_hotspot', None)
        if (
                isinstance(trajectory_hotspot, dict) and
                int(trajectory_hotspot.get('frame', -1)) == current):
            hotspot_motion_mean = float(trajectory_hotspot.get('motion_mean', 999.0) or 0.0)
            hotspot_motion_max = float(trajectory_hotspot.get('motion_max', 999.0) or 0.0)

            # A predicted-path candidate that is nearly static in the image and
            # simultaneously creates a large trajectory break is suspicious on
            # its own. Do not suppress this because a player/contact watch is
            # active: that is exactly when Local AI can protect the trajectory.
            strong_hotspot_break = (
                prior_speed >= 14.0 and proposed_speed >= 14.0 and
                angle_delta >= 50.0 and prediction_error >= 12.0 and
                hotspot_motion_mean <= 3.0 and hotspot_motion_max <= 12.0
            )
            print(
                f'[TRAJECTORY_LOCAL_AI_GATE] f{current}: '
                f'previous={previous} normal={tracked} predicted={predicted} '
                f'prior_speed={prior_speed:.1f} proposed_speed={proposed_speed:.1f} '
                f'angle={angle_delta:.1f}deg pred_error={prediction_error:.1f}px '
                f'motion={hotspot_motion_mean:.1f}/{hotspot_motion_max:.1f} '
                f'zone={zone} contact_near={contact_near} watch_active={watch_active} '
                f'pass={strong_hotspot_break}'
            )
            if strong_hotspot_break:
                reason = (
                    f'trajectory-hotspot-turn:{angle_delta:.0f}deg/'
                    f'pred={prediction_error:.0f}px/'
                    f'motion={hotspot_motion_mean:.1f}/{hotspot_motion_max:.1f}'
                )
                print(
                    f'[TRAJECTORY_LOCAL_AI_TRIGGER] f{current}: '
                    f'previous={previous} normal={tracked} predicted={predicted} '
                    f'angle={angle_delta:.1f}deg pred_error={prediction_error:.1f}px '
                    f'motion={hotspot_motion_mean:.1f}/{hotspot_motion_max:.1f}'
                )
                return reason
"""

if old not in text:
    if '[TRAJECTORY_LOCAL_AI_GATE]' in text:
        print('v2 patch already applied')
    else:
        raise SystemExit('target hotspot trigger block not found')
else:
    text = text.replace(old, new, 1)
    path.write_text(text, encoding='utf-8')
    print('applied trajectory Local AI v2 patch')
