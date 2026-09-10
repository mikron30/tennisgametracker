"""Train and run the optional local ball-candidate AI.

This module is deliberately a *candidate scorer*, not a replacement tracker.
The regular tracker still generates colour/motion candidates.  During a loss,
the scorer ranks those candidate image patches and the caller is expected to
apply its own temporal and geometry checks before accepting an answer.

Run it with the dedicated Python 3.10 environment created in ``.tools``::

    .tools\\ball-ai-venv310\\Scripts\\python.exe ball_local_ai.py train \
        --database metadata\\ball_dataset\\ball_ai.sqlite \
        --model metadata\\ball_dataset\\ball_patch_model.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


PATCH_SIZE = 96


def _torch():
    """Import the optional runtime only for train/inference commands."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:  # pragma: no cover - depends on local runtime
        raise RuntimeError(
            "Local AI runtime is not installed. Use .tools\\ball-ai-venv310 "
            "and install torch-directml first."
        ) from error
    return torch, nn, functional, DataLoader, Dataset


def _device(torch):
    """Prefer DirectML but retain a deterministic CPU fallback."""
    try:  # pragma: no cover - hardware dependent
        import torch_directml
        return torch_directml.device(), "directml"
    except Exception:
        return torch.device("cpu"), "cpu"


def _crop_rgb(image: Image.Image, center: Sequence[float], size: int = PATCH_SIZE) -> np.ndarray:
    """Return a fixed patch, black padding rather than silently shifting it."""
    x, y = (int(round(float(value))) for value in center)
    half = size // 2
    canvas = Image.new("RGB", (size, size))
    source = image.crop((x - half, y - half, x + half, y + half))
    source_left = max(0, half - x)
    source_top = max(0, half - y)
    canvas.paste(source, (source_left, source_top))
    return np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0


def _model_class():
    torch, nn, _, _, _ = _torch()

    class BallPatchCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 20, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(20, 36, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                # Keep spatial layout.  Candidate scoring must distinguish a
                # ball *at the candidate centre* from the same ball near a
                # crop edge; global average pooling erased that distinction.
                nn.Conv2d(36, 48, 3, padding=1), nn.ReLU(),
            )
            self.classifier = nn.Linear(48 * (PATCH_SIZE // 4) * (PATCH_SIZE // 4), 1)

        def forward(self, batch):
            return self.classifier(self.features(batch).flatten(1)).squeeze(1)

    return BallPatchCNN


def _legacy_model_class():
    """Architecture used by early 48-feature local-AI checkpoints."""
    torch, nn, _, _, _ = _torch()

    class LegacyBallPatchCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 20, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(20, 36, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(36, 48, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(48, 1)

        def forward(self, batch):
            return self.classifier(self.features(batch).flatten(1)).squeeze(1)

    return LegacyBallPatchCNN


def _checkpoint_architecture(checkpoint):
    """Identify supported patch-CNN generations from classifier shape."""
    state_dict = checkpoint.get('state_dict') or {}
    weight = state_dict.get('classifier.weight')
    shape = getattr(weight, 'shape', None)
    if shape is None or len(shape) != 2:
        raise RuntimeError('Unsupported local-AI checkpoint: missing classifier.weight shape')
    in_features = int(shape[1])
    if in_features == 48:
        return 'legacy-global-average-pool'
    current_features = 48 * (PATCH_SIZE // 4) * (PATCH_SIZE // 4)
    if in_features == current_features:
        return 'spatial-center-aware'
    raise RuntimeError(
        f'Unsupported local-AI checkpoint classifier width: {in_features}'
    )


@dataclass(frozen=True)
class TrainingRow:
    image_path: str
    x: float
    y: float
    group: str = ""


def _training_rows(database: Path, limit: int, seed: int) -> list[TrainingRow]:
    connection = sqlite3.connect(database)
    try:
        rows = [TrainingRow(*row) for row in connection.execute(
            "SELECT f.image_path, "
            "CASE WHEN r.review_status='corrected' THEN r.corrected_x ELSE f.ball_x END, "
            "CASE WHEN r.review_status='corrected' THEN r.corrected_y ELSE f.ball_y END, "
            "COALESCE(f.video_path, f.video_id, f.run_id) "
            "FROM ball_frames f JOIN label_reviews r USING(run_id, source_frame) "
            "WHERE r.review_status IN ('accepted','corrected') "
            "AND (r.review_status != 'corrected' OR "
            "(r.corrected_x IS NOT NULL AND r.corrected_y IS NOT NULL)) "
            "ORDER BY f.run_id, f.source_frame"
        )]
    finally:
        connection.close()
    random.Random(seed).shuffle(rows)
    return rows[: max(1, limit)]


def _reviewed_patch_rows(database: Path, table: str) -> list[TrainingRow]:
    if table not in {"hard_negative_patches", "hard_positive_patches"}:
        raise ValueError(f"Unsupported reviewed-patch table: {table}")
    connection = sqlite3.connect(database)
    try:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            return []
        return [TrainingRow(*row) for row in connection.execute(
            f"SELECT p.image_path, p.candidate_x, p.candidate_y, "
            "COALESCE((SELECT f.video_path FROM ball_frames f WHERE f.image_path=p.image_path LIMIT 1), "
            "(SELECT g.video_path FROM reviewed_image_sources g WHERE g.image_path=p.image_path)) "
            f"FROM {table} p ORDER BY p.added_at, p.source_frame"
        )]
    finally:
        connection.close()


def reviewed_split(database: Path, samples: int, seed: int):
    """Keep entire source videos together across labels and hard patches.

    Fail closed for patches without source provenance. Row limits and random
    augmentation never decide held-out membership.
    """
    parts = [_training_rows(database, 10**9, seed),
             _reviewed_patch_rows(database, "hard_negative_patches"),
             _reviewed_patch_rows(database, "hard_positive_patches")]
    key = lambda row: (row.image_path, row.x, row.y)
    negative_keys = {key(row) for row in parts[1]}
    if any(key(row) in negative_keys for row in parts[0] + parts[2]):
        raise RuntimeError("Conflicting positive/negative reviews must be corrected before training")
    difficult_keys = {key(row) for row in parts[2]}
    parts[0] = [row for row in parts[0] if key(row) not in difficult_keys]
    if any(not row.group for part in parts for row in part):
        raise RuntimeError("Reviewed patches need video provenance in reviewed_image_sources")
    groups = sorted({row.group for part in parts for row in part})
    if len(groups) < 2:
        raise RuntimeError("Need reviewed examples from at least two source videos for held-out evaluation")
    random.Random(seed).shuffle(groups)
    held_out = set(groups[max(1, int(len(groups) * .8)):])
    train = [[row for row in part if row.group not in held_out] for part in parts]
    validation = [[row for row in part if row.group in held_out] for part in parts]
    train[0] = train[0][:max(1, samples)]
    return train, validation


def _make_dataset(rows: list[TrainingRow], hard_negatives: list[TrainingRow],
                  hard_positives: list[TrainingRow], seed: int):
    torch, _, _, _, Dataset = _torch()
    rng = random.Random(seed)
    examples: list[tuple[TrainingRow, tuple[float, float], float]] = []
    for row in rows:
        # Jitter means the network learns the ball itself, not one exact pixel.
        examples.append((row, (row.x + rng.uniform(-8, 8), row.y + rng.uniform(-8, 8)), 1.0))
        angle = rng.random() * 6.2831853
        # Most recovery false points are close to the true flight path.  Train
        # on a centre-miss just outside the patch as well as easy background.
        distance = rng.uniform(62, 180)
        examples.append((row, (row.x + np.cos(angle) * distance,
                               row.y + np.sin(angle) * distance), 0.0))

    # A manually/visually reviewed false candidate is much more valuable than
    # an arbitrary background crop.  Repeat it with a tiny jitter so a few
    # corrections have enough influence among the reviewed examples.
    for row in hard_negatives:
        for _ in range(24):
            examples.append((
                row,
                (row.x + rng.uniform(-6, 6), row.y + rng.uniform(-6, 6)),
                0.0,
            ))
    for row in hard_positives:
        for _ in range(24):
            examples.append((
                row,
                (row.x + rng.uniform(-6, 6), row.y + rng.uniform(-6, 6)),
                1.0,
            ))
        angle = rng.random() * 6.2831853
        distance = rng.uniform(200, 440)
        examples.append((row, (row.x + np.cos(angle) * distance,
                               row.y + np.sin(angle) * distance), 0.0))

    class PatchDataset(Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, index):
            row, center, label = examples[index]
            with Image.open(row.image_path) as image:
                patch = _crop_rgb(image.convert("RGB"), center)
            return torch.from_numpy(patch), torch.tensor(label, dtype=torch.float32)

    return PatchDataset()


def train(database: Path, model_path: Path, *, epochs: int, samples: int,
          batch_size: int, seed: int) -> dict:
    if model_path.exists():
        raise FileExistsError("Refusing to overwrite an existing model; choose a candidate path")
    train_parts, validation_parts = reviewed_split(database, samples, seed)
    train_rows, train_hard_negatives, train_hard_positives = train_parts
    validation_rows, validation_hard_negatives, validation_hard_positives = validation_parts
    rows = train_rows + validation_rows
    hard_negatives = train_hard_negatives + validation_hard_negatives
    hard_positives = train_hard_positives + validation_hard_positives
    if len(train_rows) + len(train_hard_positives) < 100:
        raise RuntimeError("Need at least 100 reviewed training positives; tracker labels are not ground truth")
    if not validation_rows + validation_hard_positives or not validation_hard_negatives:
        raise RuntimeError("Held-out evaluation requires reviewed positives and hard negatives")
    torch, _, functional, DataLoader, _ = _torch()
    device, device_name = _device(torch)
    model = _model_class()().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(_make_dataset(
        train_rows, train_hard_negatives, train_hard_positives, seed
    ), batch_size=batch_size,
                              shuffle=True, num_workers=0)
    validation_loader = DataLoader(_make_dataset(
        validation_rows, validation_hard_negatives, validation_hard_positives, seed + 1
    ), batch_size=batch_size,
                                   shuffle=False, num_workers=0)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = functional.binary_cross_entropy_with_logits(model(patches), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(labels)
            count += len(labels)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for patches, labels in validation_loader:
                predicted = torch.sigmoid(model(patches.to(device))).cpu() >= 0.5
                correct += int((predicted == (labels >= 0.5)).sum())
                total += len(labels)
        history.append({"epoch": epoch, "loss": total_loss / max(1, count),
                        "validation_accuracy": correct / max(1, total)})
        print(json.dumps(history[-1], sort_keys=True))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"state_dict": model.cpu().state_dict(), "patch_size": PATCH_SIZE,
                "device_trained": device_name, "rows": len(rows),
                "hard_negatives": len(hard_negatives), "hard_positives": len(hard_positives),
                "review_policy": "explicit-review-only/source-video-split",
                "split_seed": seed,
                "held_out_videos": sorted({r.group for part in validation_parts for r in part}),
                "history": history}
    with model_path.open('xb') as output:
        torch.save(checkpoint, output)
    return {"model": str(model_path), "device": device_name, "rows": len(rows),
            "hard_negatives": len(hard_negatives), "hard_positives": len(hard_positives),
            "history": history}


def score(model_path: Path, image_path: Path, candidates: Iterable[dict], batch_size: int) -> list[dict]:
    torch, _, _, _, _ = _torch()
    device, model = _load_model(model_path, torch)
    candidates = list(candidates)
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        patches = [_crop_rgb(rgb, (candidate["x"], candidate["y"])) for candidate in candidates]
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(patches), max(1, batch_size)):
            batch = torch.from_numpy(np.stack(patches[start:start + batch_size])).to(device)
            scores.extend(torch.sigmoid(model(batch)).cpu().tolist())
    return [{**candidate, "ai_score": round(float(value), 6)}
            for candidate, value in zip(candidates, scores)]


def score_batch(model_path: Path, requests: Iterable[dict], batch_size: int) -> list[list[dict]]:
    """Score several frame candidate lists while loading the model once.

    Local recovery normally needs a short temporal path.  Starting Python and
    DirectML separately for each frame made a four-frame check far slower than
    the HSV tracker itself.  Requests stay separate in the result so callers
    retain their existing per-frame continuity logic.
    """
    torch, _, _, _, _ = _torch()
    device, model = _load_model(model_path, torch)
    prepared: list[tuple[list[dict], list[np.ndarray]]] = []
    for request in requests:
        candidates = list(request.get("candidates") or [])
        image_path = Path(request["image"])
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            patches = [_crop_rgb(rgb, (candidate["x"], candidate["y"])) for candidate in candidates]
        prepared.append((candidates, patches))

    flat_patches = [patch for _, patches in prepared for patch in patches]
    flat_scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(flat_patches), max(1, batch_size)):
            batch = torch.from_numpy(np.stack(flat_patches[start:start + batch_size])).to(device)
            flat_scores.extend(torch.sigmoid(model(batch)).cpu().tolist())

    results: list[list[dict]] = []
    offset = 0
    for candidates, patches in prepared:
        scores = flat_scores[offset:offset + len(patches)]
        offset += len(patches)
        results.append([
            {**candidate, "ai_score": round(float(value), 6)}
            for candidate, value in zip(candidates, scores)
        ])
    return results


def _load_model(model_path: Path, torch, *, return_metadata=False):
    """Load once for scoring/evaluation; never run an untrusted checkpoint."""
    device, _ = _device(torch)
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:  # torch versions before weights_only support
        checkpoint = torch.load(model_path, map_location="cpu")
    architecture = _checkpoint_architecture(checkpoint)
    model_factory = (
        _legacy_model_class()
        if architecture == 'legacy-global-average-pool'
        else _model_class()
    )
    model = model_factory()
    model.load_state_dict(checkpoint["state_dict"])
    checkpoint = dict(checkpoint)
    checkpoint.setdefault('model_architecture', architecture)
    loaded = (device, model.to(device).eval())
    return (*loaded, checkpoint) if return_metadata else loaded


def evaluate(database: Path, model_path: Path, *, samples: int, seed: int,
             batch_size: int) -> dict:
    """Measure whether the true centre beats near and far false candidates.

    This uses the same deterministic source-video split that ``train`` reserves,
    but evaluates the real recovery decision: choose the highest score among
    a true candidate and four nearby alternatives.
    """
    torch, _, _, _, _ = _torch()
    _, validation = reviewed_split(database, samples, seed)
    positive_rows, negative_rows, difficult_rows = validation
    validation_rows = positive_rows + difficult_rows
    if not validation_rows or not negative_rows:
        raise RuntimeError("Held-out evaluation requires reviewed positives and hard negatives")
    device, model, checkpoint = _load_model(model_path, torch, return_metadata=True)
    held_out_videos = sorted({row.group for part in validation for row in part})
    recorded_videos = checkpoint.get('held_out_videos')
    if recorded_videos is not None and (
            checkpoint.get('split_seed') != seed or recorded_videos != held_out_videos):
        raise RuntimeError('Evaluation split differs from checkpoint; use its seed and frozen database')
    offsets = ((0, 0), (-64, 0), (64, 0), (0, -64), (0, 64))
    wins = 0
    correct = total = false_positives = recovery_false_positives = difficult_wins = 0
    margins: list[float] = []
    with torch.no_grad():
        for row in validation_rows:
            with Image.open(row.image_path) as image:
                rgb = image.convert("RGB")
                patches = np.stack([
                    _crop_rgb(rgb, (row.x + dx, row.y + dy))
                    for dx, dy in offsets
                ])
            scores = torch.sigmoid(model(torch.from_numpy(patches).to(device))).cpu().tolist()
            win = int(int(np.argmax(scores)) == 0)
            wins += win
            if row in difficult_rows:
                difficult_wins += win
            correct += int(scores[0] >= 0.5)
            total += 1
            margins.append(float(scores[0] - max(scores[1:])))
        for row in negative_rows:
            with Image.open(row.image_path) as image:
                patch = _crop_rgb(image.convert("RGB"), (row.x, row.y))
            value = float(torch.sigmoid(model(torch.from_numpy(patch[None]).to(device))).cpu()[0])
            false_positives += int(value >= 0.5)
            recovery_false_positives += int(value >= 0.985)
            correct += int(value < 0.5)
            total += 1
    return {
        "validation_accuracy": correct / total,
        "held_out_provenance_verified": recorded_videos is not None,
        "held_out_videos": held_out_videos,
        "hard_negative_false_positive_rate": false_positives / len(negative_rows),
        "hard_negative_false_positive_rate_at_0_985": recovery_false_positives / len(negative_rows),
        "hard_negative_count": len(negative_rows),
        "reviewed_difficult_top1_accuracy": difficult_wins / len(difficult_rows) if difficult_rows else None,
        "reviewed_difficult_count": len(difficult_rows),
        "rows": len(validation_rows),
        "top1_true_center_accuracy": wins / len(validation_rows),
        "mean_true_center_margin": float(np.mean(margins)),
        "minimum_true_center_margin": float(np.min(margins)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local candidate-scoring AI for TennisGameTracker")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--database", required=True)
    train_parser.add_argument("--model", required=True)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--samples", type=int, default=6000)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--seed", type=int, default=1337)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--model", required=True)
    score_parser.add_argument("--image", required=True)
    score_parser.add_argument("--candidates", required=True,
                              help="JSON file containing [{x, y, ...}, ...]")
    score_parser.add_argument("--batch-size", type=int, default=64)
    score_batch_parser = commands.add_parser("score-batch")
    score_batch_parser.add_argument("--model", required=True)
    score_batch_parser.add_argument("--requests", required=True,
                                    help="JSON file containing [{image, candidates}, ...]")
    score_batch_parser.add_argument("--batch-size", type=int, default=64)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--database", required=True)
    evaluate_parser.add_argument("--model", required=True)
    evaluate_parser.add_argument("--samples", type=int, default=1500)
    evaluate_parser.add_argument("--seed", type=int, default=1337)
    evaluate_parser.add_argument("--batch-size", type=int, default=64)
    compare_parser = commands.add_parser('compare')
    compare_parser.add_argument('--database', required=True, help='Frozen reviewed SQLite snapshot')
    compare_parser.add_argument('--old-model', required=True)
    compare_parser.add_argument('--candidate-model', required=True)
    compare_parser.add_argument('--report', required=True)
    compare_parser.add_argument('--seed', type=int, default=1337)
    compare_parser.add_argument('--samples', type=int, default=6000)
    compare_parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(train(Path(args.database).resolve(), Path(args.model).resolve(),
                               epochs=max(1, args.epochs), samples=max(100, args.samples),
                               batch_size=max(1, args.batch_size), seed=args.seed), indent=2))
    elif args.command == "score":
        candidates = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
        print(json.dumps(score(Path(args.model).resolve(), Path(args.image).resolve(),
                               candidates, max(1, args.batch_size))))
    elif args.command == "score-batch":
        requests = json.loads(Path(args.requests).read_text(encoding="utf-8"))
        print(json.dumps(score_batch(Path(args.model).resolve(), requests,
                                     max(1, args.batch_size))))
    elif args.command == 'compare':
        results = {
            name: evaluate(Path(args.database).resolve(), Path(path).resolve(),
                           samples=args.samples, seed=args.seed, batch_size=args.batch_size)
            for name, path in [('old', args.old_model), ('candidate', args.candidate_model)]
        }
        higher = ['validation_accuracy', 'top1_true_center_accuracy', 'reviewed_difficult_top1_accuracy']
        lower = ['hard_negative_false_positive_rate', 'hard_negative_false_positive_rate_at_0_985']
        old, candidate = results['old'], results['candidate']
        complete = all(old[key] is not None and candidate[key] is not None for key in higher + lower)
        no_regression = complete and all(candidate[k] >= old[k] for k in higher) and all(
            candidate[k] <= old[k] for k in lower)
        results['measured_improvement_without_regression'] = bool(no_regression and (
            any(candidate[k] > old[k] for k in higher) or any(candidate[k] < old[k] for k in lower)))
        results['models_replaced'] = False
        results['note'] = ('Comparison only. Promotion requires independent held-out provenance for both models, '
                           'adequate reviewed sample coverage, and the complete tracker regression.')
        Path(args.report).write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(json.dumps(results, indent=2))
    else:
        print(json.dumps(evaluate(Path(args.database).resolve(), Path(args.model).resolve(),
                                  samples=max(100, args.samples), seed=args.seed,
                                  batch_size=max(1, args.batch_size)), indent=2))


if __name__ == "__main__":
    main()
