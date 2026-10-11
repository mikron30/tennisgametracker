from pathlib import Path

path = Path("interactive_ball_analyzer.py")
text = path.read_text(encoding="utf-8")

old = '''                            net_geometry.get('near_polygon', False) and
                            last_change_geometry.get('near_polygon', False) and
                            not direct_contact_zone and
                            prev_dist <= 5.0 and
'''

new = '''                            net_geometry.get('near_polygon', False) and
                            last_change_geometry.get('near_polygon', False) and
                            prev_dist <= 5.0 and
'''

if old not in text:
    if '[V39 SLOW NEAR-NET TURN]' in text:
        print('V39 already applied')
        compile(text, str(path), 'exec')
        raise SystemExit(0)
    raise SystemExit('V39 target block not found')

text = text.replace(old, new, 1)
text = text.replace(
    '[V38 SLOW NEAR-NET TURN]',
    '[V39 SLOW NEAR-NET TURN]',
    1,
)
text = text.replace(
    "# V38: V37's repaired trajectory reaches the net correctly, but",
    "# V39: V37's repaired trajectory reaches the net correctly, but",
    1,
)
text = text.replace(
    "# its low-speed f322 turn can sit just outside the strict net",
    "# its low-speed f322 turn can straddle the strict net boundary:",
    1,
)
text = text.replace(
    "# polygon/top-tape gate.  Accept only this very narrow physical",
    "# the previous turn point may be just outside while the live ball is",
    1,
)
text = text.replace(
    "# signature: the ball was crawling immediately before a sharp",
    "# already inside. Accept only this very narrow physical signature:",
    1,
)
text = text.replace(
    "# downward turn, both the live point and the direction-change",
    "# the ball was crawling immediately before a sharp downward turn,",
    1,
)
text = text.replace(
    "# point are still within the calibrated near-net polygon, and",
    "# both points are within the calibrated near-net polygon margin, and",
    1,
)
text = text.replace(
    "# the current step remains slow.  The surrounding branch already",
    "# the current step remains slow. The surrounding branch already",
    1,
)

path.write_text(text, encoding="utf-8")
compile(text, str(path), 'exec')
print('Applied V39 direct-or-near net turn guard')
