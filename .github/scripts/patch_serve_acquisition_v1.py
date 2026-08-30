from pathlib import Path

path = Path("interactive_ball_analyzer.py")
text = path.read_text(encoding="utf-8")


def replace_once(old, new, label):
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly 1 match, found {count}")
    text = text.replace(old, new, 1)
    print(f"patched: {label}")


# 1) A player-body blob must never become the first serve sample merely because
# the current court profile is not classified as a night session. Once a real
# toss path is locked the existing code still allows body/racket crossings.
replace_once(
'''        if lock_active or not self._is_night_session_config():
            return False
        return self._player_point_zone(point) == "player_body"
''',
'''        if lock_active:
            return False
        return self._player_point_zone(point) == "player_body"
''',
"general pre-lock serve body rejection",
)

# 2) The historical startup bias was tied to the first ten decoded frames and
# could force a large high contour to win. Keep the useful high-ball preference
# only for inactive serve scans and only for compact-sized, non-body seeds.
replace_once(
'''        # Early-serve bias: when starting and no previous ball, favor the highest (smallest y) valid contour
        if self.ball_center is None and self.frame_count <= self.start_frame + 10 and candidates:
            highest = min(candidates, key=lambda c: (c[3], c[4]))  # prioritize lowest y (higher on screen), then smaller area
            _, best_source, _, _, _, _, _, _ = highest
            best_contour = contours[highest[0]][1]
            print(f"  DEBUG: Early-serve bias -> picking highest contour (y={highest[3]}, area={highest[4]:.1f})")
''',
'''        # Startup serve bias: a high contour is useful only when it still has
        # a plausible seed size.  Never let a large player/background blob win
        # solely because it happens to be highest in the first decoded frames.
        if (allow_inactive and self.ball_center is None and
                self.frame_count <= self.start_frame + 10 and candidates):
            seed_area_cap = max(
                18.0,
                min(float(self.serve_ball_size_max), float(self.serve_ball_size_min) * 2.5),
            )
            seed_candidates = [
                candidate for candidate in candidates
                if candidate[4] <= seed_area_cap and
                self._player_point_zone((candidate[2], candidate[3])) != "player_body"
            ]
            if seed_candidates:
                highest = min(seed_candidates, key=lambda c: (c[3], c[4]))
                _, best_source, _, _, _, _, _, _ = highest
                best_contour = contours[highest[0]][1]
                print(
                    f"  DEBUG: [SERVE_SEED_BIAS] highest compact contour "
                    f"y={highest[3]} area={highest[4]:.1f}px cap={seed_area_cap:.1f}px"
                )
            else:
                print(
                    f"  DEBUG: [SERVE_SEED_BIAS] no compact seed candidate "
                    f"under {seed_area_cap:.1f}px; keeping normal ranking"
                )
''',
"safe startup serve bias",
)

# 3) Persist generic shape evidence with every serve candidate.  This lets the
# temporal toss validator distinguish a compact moving ball from a torso/racket
# colour fragment without using any frame number or fixed location.
replace_once(
'''                            local_hsv = hsv_frame[cy, cx]
                            self._last_detected_serve_candidate = {
                                'pos': (cx, cy),
                                'frame': int(self.frame_count),
                                'area': float(area),
                                'hsv': (int(local_hsv[0]), int(local_hsv[1]), int(local_hsv[2])),
                                'motion_delta': motion_delta,
                            }
''',
'''                            local_hsv = hsv_frame[cy, cx]
                            box_x, box_y, box_w, box_h = cv2.boundingRect(contour)
                            short_side = max(1, min(box_w, box_h))
                            aspect_ratio = float(max(box_w, box_h)) / float(short_side)
                            perimeter = float(cv2.arcLength(contour, True))
                            circularity = (
                                float(4.0 * math.pi * area / (perimeter * perimeter))
                                if perimeter > 0.0 else 0.0
                            )
                            self._last_detected_serve_candidate = {
                                'pos': (cx, cy),
                                'frame': int(self.frame_count),
                                'area': float(area),
                                'hsv': (int(local_hsv[0]), int(local_hsv[1]), int(local_hsv[2])),
                                'motion_delta': motion_delta,
                                'bbox': (int(box_x), int(box_y), int(box_w), int(box_h)),
                                'aspect_ratio': aspect_ratio,
                                'circularity': circularity,
                            }
''',
"serve candidate shape metadata",
)

# 4) A valid serve start already requires three upward steps. Add independent
# geometry/shape/motion evidence to those same steps. The thresholds scale with
# the configured serve region and deliberately remain permissive for blur.
replace_once(
'''            if len(up_steps) < 3:
                return None
            if (
                    night_recent_point_end_start and
''',
'''            if len(up_steps) < 3:
                return None

            toss_segment = history[low_index:min_index + 1]
            serve_width = max(
                1,
                int(getattr(self, 'serve_area_x_max', 1)) -
                int(getattr(self, 'serve_area_x_min', 0)),
            )
            lateral_span = float(max(p[0] for p in toss_segment) - min(p[0] for p in toss_segment))
            max_lateral_span = max(110.0, serve_width * 0.07, rise * 0.75)
            if lateral_span > max_lateral_span:
                print(
                    f"[SERVE_TOSS_REJECT] f{self.frame_count}: lateral-span "
                    f"{lateral_span:.0f}px > {max_lateral_span:.0f}px"
                )
                return None

            if details is not None:
                segment_details = [
                    details[idx] for idx in range(low_index, min_index + 1)
                    if idx < len(details) and isinstance(details[idx], dict)
                ]
                shape_details = [
                    detail for detail in segment_details
                    if detail.get('aspect_ratio') is not None and
                    detail.get('circularity') is not None
                ]
                if len(shape_details) >= 3:
                    compact_votes = sum(
                        1 for detail in shape_details
                        if float(detail['aspect_ratio']) <= 3.2 and
                        float(detail['circularity']) >= 0.12
                    )
                    required_compact = max(2, int(math.ceil(len(shape_details) * 0.60)))
                    if compact_votes < required_compact:
                        print(
                            f"[SERVE_TOSS_REJECT] f{self.frame_count}: shape votes "
                            f"{compact_votes}/{len(shape_details)} < {required_compact}"
                        )
                        return None

                motion_values = [
                    float(detail['motion_delta']) for detail in segment_details
                    if detail.get('motion_delta') is not None
                ]
                if len(motion_values) >= 3:
                    moving_votes = sum(1 for value in motion_values if value >= 6.0)
                    if moving_votes < 2:
                        print(
                            f"[SERVE_TOSS_REJECT] f{self.frame_count}: motion votes "
                            f"{moving_votes}/{len(motion_values)} < 2"
                        )
                        return None

            if (
                    night_recent_point_end_start and
''',
"serve toss multi-evidence confirmation",
)

# 5) Do not poison the first tracking frames with the size of a clearly
# non-ball-shaped seed.  A trusted compact seed keeps the old behaviour; an
# untrusted one starts with size=None so trajectory/motion can acquire the ball.
replace_once(
'''            area = best_detail.get('area')
            hsv = best_detail.get('hsv')
            if area is not None:
                self.ball_size = float(area)
            if hsv is not None:
                self.ball_hsv = np.array(hsv, dtype=np.uint8)
''',
'''            area = best_detail.get('area')
            hsv = best_detail.get('hsv')
            aspect_ratio = best_detail.get('aspect_ratio')
            circularity = best_detail.get('circularity')
            trusted_shape = (
                aspect_ratio is None or circularity is None or
                (float(aspect_ratio) <= 3.2 and float(circularity) >= 0.12)
            )
            if area is not None and trusted_shape:
                self.ball_size = float(area)
                print(
                    f"[SERVE_SEED_SIZE] f{self.frame_count}: trusted area={self.ball_size:.1f}px "
                    f"aspect={float(aspect_ratio):.2f} circ={float(circularity):.2f}"
                    if aspect_ratio is not None and circularity is not None else
                    f"[SERVE_SEED_SIZE] f{self.frame_count}: legacy area={self.ball_size:.1f}px"
                )
            elif area is not None:
                self.ball_size = None
                print(
                    f"[SERVE_SEED_SIZE] f{self.frame_count}: ignored untrusted area={float(area):.1f}px "
                    f"aspect={float(aspect_ratio):.2f} circ={float(circularity):.2f}"
                )
            if hsv is not None:
                self.ball_hsv = np.array(hsv, dtype=np.uint8)
''',
"safe serve tracking size seed",
)

path.write_text(text, encoding="utf-8")
print("patch complete")
