"""Train a compact radar segmentation model and export the held-out MMFlood replay."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
import torch
from torch import nn

SIZE = 128
HOLDOUT = "EMSR265"


def load_tiles(directory):
    examples = []
    for label in sorted(directory.glob("*_Label.tif")):
        name = label.name.removesuffix("_Label.tif")
        with rasterio.open(label) as raster:
            y = raster.read(1, out_shape=(SIZE, SIZE), resampling=Resampling.nearest)
            bounds = list(raster.bounds)
        with rasterio.open(directory / f"{name}_Radar.tif") as raster:
            x = raster.read((1, 2), out_shape=(2, SIZE, SIZE), resampling=Resampling.bilinear)
        valid = np.isin(y, (0, 1)) & np.isfinite(x).all(axis=0) & (x > 0).all(axis=0)
        db = 10 * np.log10(np.maximum(np.nan_to_num(x), 1e-6))
        x = np.clip((db + 35) / 40, 0, 1)
        examples.append((name, x.astype(np.float32), y, valid, bounds))
    return examples


def scores(prediction, label, valid):
    tp = int((prediction & (label == 1) & valid).sum())
    fp = int((prediction & (label == 0) & valid).sum())
    fn = int((~prediction & (label == 1) & valid).sum())
    return np.array([tp, fp, fn])


def report(counts):
    tp, fp, fn = map(int, counts)
    return {"iou": tp / max(1, tp + fp + fn), "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn), "tp": tp, "fp": fp, "fn": fn}


def model():
    return nn.Sequential(nn.Conv2d(2, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(24, 12, 3, padding=1), nn.ReLU(),
                         nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                         nn.Conv2d(12, 1, 1))


def train(examples, epochs):
    torch.manual_seed(17)
    torch.set_num_threads(3)
    x = torch.tensor(np.stack([item[1] for item in examples]))
    labels = torch.tensor(np.stack([item[2] == 1 for item in examples]))
    valid = torch.tensor(np.stack([item[3] for item in examples]))
    detector = model()
    optimizer = torch.optim.Adam(detector.parameters(), lr=.003)
    positives = (labels & valid).sum().item()
    negatives = (~labels & valid).sum().item()
    positive_weight = min(10.0, max(1.0, negatives / max(1, positives)))
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    for _ in range(epochs):
        for indexes in torch.randperm(len(examples)).split(5):
            mask = valid[indexes]
            predictions = detector(x[indexes])[:, 0]
            optimizer.zero_grad()
            error = loss(predictions[mask], labels[indexes][mask].float())
            error.backward()
            optimizer.step()
    return detector.eval()


def save_view(path, image, mask, valid, color):
    gray = np.repeat((image * 228).astype("uint8")[:, :, None], 3, axis=2)
    if mask is not None:
        gray[mask & valid] = color
    gray[~valid] = (25, 30, 38)
    Image.fromarray(gray).save(path)


def export_replay(examples, output, detector, cutoff, probability_cutoff):
    (output / "tiles").mkdir(parents=True, exist_ok=True)
    totals = {"model": np.zeros(3, dtype=int), "threshold": np.zeros(3, dtype=int)}
    tiles = []
    for name, x, labels, valid, bounds in examples:
        with torch.inference_mode():
            predicted = detector(torch.from_numpy(x[None]))[0, 0].sigmoid().numpy() >= probability_cutoff
        threshold = x[0] < cutoff
        for key, mask in (("model", predicted), ("threshold", threshold)):
            totals[key] += scores(mask, labels, valid)
        for key, mask, color in (("radar", None, None), ("model", predicted, (65,213,224)),
                                 ("reference", labels == 1, (75,198,158)),
                                 ("errors", None, None), ("threshold", threshold, (65,213,224))):
            path = output / "tiles" / f"{name}-{key}.png"
            save_view(path, x[0], mask, valid, color)
            if key == "errors":
                canvas = np.array(Image.open(path))
                canvas[predicted & (labels == 0) & valid] = (247, 162, 101)
                canvas[~predicted & (labels == 1) & valid] = (221, 144, 235)
                Image.fromarray(canvas).save(path)
        tiles.append({"id": name, "bounds": bounds, "model": report(scores(predicted, labels, valid)),
                      "threshold": report(scores(threshold, labels, valid)),
                      "water_fraction": float(((labels == 1) & valid).sum() / max(1, valid.sum()))})
    return {key: report(value) for key, value in totals.items()}, tiles


def run(directory, output, epochs):
    examples = load_tiles(directory)
    earlier = [item for item in examples if not item[0].startswith(HOLDOUT + "-")]
    later = [item for item in examples if item[0].startswith(HOLDOUT + "-")]
    if not earlier or not later:
        raise ValueError("Training and held-out events must both be present.")
    cutoff = max(np.arange(.15, .76, .025),
                 key=lambda t: report(sum(scores(x[0] < t, y, v) for _, x, y, v, _ in earlier))["iou"])
    detector = train(earlier, epochs)
    with torch.inference_mode():
        fitted = detector(torch.tensor(np.stack([item[1] for item in earlier])))[:, 0].sigmoid().numpy()
    probability_cutoff = max(np.arange(.05, .96, .025), key=lambda t: report(sum(
        scores(p >= t, y, valid) for p, (_, _, y, valid, _) in zip(fitted, earlier)))["iou"])
    (output / "tiles").mkdir(parents=True, exist_ok=True)
    torch.save(detector.state_dict(), output.parent / "model.pt")
    metrics, tiles = export_replay(later, output, detector, cutoff, probability_cutoff)
    result = {"event": HOLDOUT, "training": len(earlier), "test": len(later), "epochs": epochs,
              "threshold_cutoff": float(cutoff), "model_cutoff": float(probability_cutoff), "metrics": metrics,
              "tiles": tiles}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("site"))
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    run(args.directory, args.output, args.epochs)
