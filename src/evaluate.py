"""
Quantitative evaluation -- the piece that was completely missing from
the original project. Computes BLEU-1..4 on the held-out Flickr8k test
split and saves a results table + a handful of qualitative examples.

Run this AFTER train.py, once you have a checkpoint.
"""

import os
import json
import random
from collections import defaultdict

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import config
import data_prep
from predict import CaptionPredictor

smoothie = SmoothingFunction().method4


def build_references(test_descriptions):
    """{image: [[tokenized ref caption, ...], ...]} with <start>/<end> stripped."""
    refs = defaultdict(list)
    for img, caps in test_descriptions.items():
        for cap in caps:
            words = [
                w for w in cap.split()
                if w not in (config.START_TOKEN, config.END_TOKEN)
            ]
            refs[img].append(words)
    return refs


def evaluate(method="beam", sample_size=None, seed=42):
    _, test_descriptions, _, _ = data_prep.prepare_all()
    references = build_references(test_descriptions)

    image_ids = list(references.keys())
    if sample_size:
        random.seed(seed)
        image_ids = random.sample(image_ids, min(sample_size, len(image_ids)))

    predictor = CaptionPredictor()

    all_refs, all_hyps = [], []
    qualitative = []

    for img in image_ids:
        img_path = os.path.join(config.DATASET_IMAGES_DIR, img)
        hyp_text = predictor.caption(img_path, method=method)
        hyp_tokens = hyp_text.split()

        all_refs.append(references[img])
        all_hyps.append(hyp_tokens)

        if len(qualitative) < 10:
            qualitative.append({
                "image": img,
                "generated": hyp_text,
                "references": [" ".join(r) for r in references[img]],
            })

    scores = {}
    for n in range(1, 5):
        weights = tuple(1.0 / n for _ in range(n)) + tuple(0 for _ in range(4 - n))
        scores[f"BLEU-{n}"] = corpus_bleu(
            all_refs, all_hyps, weights=weights, smoothing_function=smoothie
        )

    results = {"method": method, "num_images": len(image_ids), "scores": scores,
               "examples": qualitative}

    with open(config.RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluated on {len(image_ids)} test images ({method} decoding)")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nFull results + qualitative examples saved to {config.RESULTS_PATH}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["greedy", "beam"], default="beam")
    parser.add_argument("--sample_size", type=int, default=None,
                         help="Evaluate on a random subset for speed; omit for full test set")
    args = parser.parse_args()

    evaluate(method=args.method, sample_size=args.sample_size)
