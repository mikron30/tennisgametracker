from pathlib import Path


TARGET = Path("interactive_ball_analyzer.py")

OLD = '''                coherent_rebound = (
                    pending_dy >= 6.0 and
                    current_dy <= -12.0 and
                    turn_angle >= 120.0 and
                    24.0 <= current_speed <= max_rebound_step and
                    50.0 <= displacement <= max_rebound_displacement
                )
                if coherent_rebound:
'''

NEW = '''                coherent_rebound = (
                    pending_dy >= 6.0 and
                    current_dy <= -12.0 and
                    turn_angle >= 120.0 and
                    24.0 <= current_speed <= max_rebound_step and
                    50.0 <= displacement <= max_rebound_displacement
                )

                # V20: a suppressed night-side hotspot is only allowed to come
                # back as a confirmed OUT when the later rebound is independent
                # ball evidence.  In the reviewed f4284->f4287 failure, the
                # original point was correctly suppressed as a static side
                # artifact, but ALT4 then selected a tiny player-body fragment
                # (H=97, 10.5px).  Its 142px upward jump looked like a perfect
                # geometric rebound and incorrectly promoted the old hotspot to
                # an OUT.  Player/racket overlap is therefore not sufficient
                # evidence to resurrect a suppressed static endpoint.  Keep the
                # pending candidate alive for its final bounded frame so a real,
                # clear rebound can still confirm it.
                rebound_player_zone = self._player_point_zone(ball_position)
                rebound_overlaps_player = rebound_player_zone in (
                    'player_head_hat',
                    'player_shoes',
                    'racket_fragment',
                    'player_body',
                )
                if coherent_rebound and rebound_overlaps_player:
                    print(
                        f"Frame {self.frame_count}: [OUT-BOUNCE PENDING REBOUND REJECT] "
                        f"source_f={pending_frame} point={pending_pos} "
                        f"rebound={tuple(ball_position)} zone={rebound_player_zone} "
                        f"speed={current_speed:.1f}px turn={turn_angle:.1f}deg "
                        f"displacement={displacement:.1f}px"
                    )
                    coherent_rebound = False

                if coherent_rebound:
'''


def main():
    text = TARGET.read_text(encoding="utf-8")
    if "OUT-BOUNCE PENDING REBOUND REJECT" in text:
        raise SystemExit("V20 guard already present")
    count = text.count(OLD)
    if count != 1:
        raise SystemExit(f"expected exactly one V20 target block, found {count}")
    updated = text.replace(OLD, NEW, 1)
    compile(updated, str(TARGET), "exec")
    TARGET.write_text(updated, encoding="utf-8")
    print("Applied V20 pending-OUT player-rebound guard")


if __name__ == "__main__":
    main()
