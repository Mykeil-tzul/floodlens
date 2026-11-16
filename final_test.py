"""Score the prespecified Sweden sample with the original released checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from train import export_replay, load_tiles, model

ROOT = Path(__file__).resolve().parent


def frozen_model(plan):
    checkpoint = ROOT / "model.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the frozen test plan.")
    detector = model()
    detector.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    return detector.eval()


def evaluate(directory, output):
    plan = json.loads((ROOT / "final-test-plan.json").read_text())
    detector = frozen_model(plan)
    examples = load_tiles(directory)
    if len(examples) != plan["tiles"] or any(not item[0].startswith(plan["event"] + "-") for item in examples):
        raise ValueError("Use the exact activation and sample size specified in the final-test plan.")
    metrics, tiles = export_replay(examples, output, detector, plan["threshold_cutoff"], plan["model_cutoff"])
    result = {**plan, "test": len(tiles), "metrics": metrics, "tiles": tiles}
    (output / "final-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", default=ROOT / "site", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(3)
    evaluate(args.directory, args.output)
