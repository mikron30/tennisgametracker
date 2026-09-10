"""Build an UNLABELLED visual review queue from one video's tracker log.

Logs nominate candidates; they never verify labels. Output keeps source-frame
provenance and separates point-audit verdicts from ball-patch reviews.
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


def parse_log(text):
    # Stop before another video resets the frame counter.
    text = text.split('[VIDEO_CONTINUE]')[0]
    tracks = {}
    for match in re.finditer(r'\[TRACK\] f(\d+): pos=\((\d+),\s*(\d+)\) vel=([\d.]+)px stuck=(\d+)', text):
        f, x, y, speed, stuck = match.groups()
        tracks[int(f)] = dict(tracker_frame=int(f), source_frame=int(f)-1,
                              x=int(x), y=int(y), speed=float(speed), stuck=int(stuck))
    points = [dict(point=int(p), start=int(s), end=int(e), log_verdict=v)
              for p, s, e, v in re.findall(
                  r'\[POINT_AUDIT\] point=(\d+) start=(\d+) end=(\d+) verdict=(\w+)', text)]
    return tracks, points


def prepare(log, video, output, per_point=6):
    raw = Path(log).read_bytes()
    encoding = 'utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
    tracks, points = parse_log(raw.decode(encoding, errors='replace'))
    if not tracks or not points:
        raise ValueError('Log must contain TRACK rows and POINT_AUDIT intervals')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = []
    for point in points:
        entries = [row for f, row in sorted(tracks.items()) if point['start'] <= f <= point['end']]
        if not entries:
            continue
        for index in sorted(set(np.linspace(0, len(entries)-1, per_point, dtype=int))):
            selected.append(dict(entries[index], point=point['point'], log_verdict=point['log_verdict']))
    cap = cv2.VideoCapture(str(video))
    tiles = []
    try:
        for index, row in enumerate(selected):
            target = row['source_frame']
            # Decode a preroll rather than using the first HEVC picture after a seek.
            begin = max(0, target-60)
            cap.set(cv2.CAP_PROP_POS_FRAMES, begin)
            for _ in range(target-begin+1):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'Could not decode source frame {target}')
            image_path = output / f'source_{target}.jpg'
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f'Could not write {image_path}')
            row.update(image_path=str(image_path), video_path=str(Path(video).resolve()),
                       review_status='unreviewed', index=index+1)
            x, y = row['x'], row['y']
            tile = np.zeros((152, 152, 3), dtype=np.uint8)
            crop = frame[max(0,y-64):y+64, max(0,x-64):x+64]
            tile[24:24+min(128,len(crop)), :min(128,crop.shape[1])] = crop[:128,:128]
            cv2.putText(tile, f'{index+1} p{row["point"]} f{row["tracker_frame"]}',
                        (2,17), cv2.FONT_HERSHEY_SIMPLEX, .38, (255,255,255), 1)
            tiles.append(tile)
    finally:
        cap.release()
    for page, offset in enumerate(range(0,len(tiles),24),1):
        batch=tiles[offset:offset+24]
        batch += [np.zeros_like(tiles[0])] * (24-len(batch))
        cv2.imwrite(str(output/f'review_{page}.jpg'), np.vstack([
            np.hstack(batch[i:i+6]) for i in range(0,24,6)]))
    (output/'queue.json').write_text(json.dumps(selected,indent=2))
    (output/'point_audits.json').write_text(json.dumps(points,indent=2))
    return len(selected)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log',required=True)
    parser.add_argument('--video',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--per-point',type=int,default=6)
    args=parser.parse_args()
    print('Unreviewed candidates:',prepare(args.log,args.video,args.output,max(1,args.per_point)))
