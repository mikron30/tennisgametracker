"""Position video decoders without losing HEVC reference pictures."""
import cv2


def position_capture(capture, frame_index):
    """Leave the next decoded picture at the requested zero-based frame.

    OpenCV's random HEVC seek can return successful reads with missing
    reference pictures until the next clean keyframe. Decode intervening
    pictures instead; grab avoids the unnecessary BGR conversion while
    retaining the decoder's reference state. Other codecs keep native seeks.
    """
    target = max(0, int(frame_index))
    current = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    if current == target:
        return True
    code = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = ''.join(chr((code >> (8 * i)) & 255) for i in range(4)).lower()
    if codec not in ('hevc', 'hvc1', 'hev1', 'h265'):
        return bool(capture.set(cv2.CAP_PROP_POS_FRAMES, target))
    if current > target:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
            return False
        current = 0
    distance = target - current
    if distance >= 250:
        print(f'[VIDEO_PREROLL] Decoding HEVC references from {current} to {target}', flush=True)
    for offset in range(distance):
        if not capture.grab():
            return False
        if distance >= 250 and (offset + 1) % 1000 == 0:
            print(f'[VIDEO_PREROLL] Decoded {current + offset + 1}/{target}', flush=True)
    return True
