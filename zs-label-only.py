"""Zero-shot multi-task animal behaviour classification using VideoPrism.

Predicts three label types for each video:
  1. Species  (single-label) — which animal species is in the video
  2. Activity (single-label) — high-level behavioural category
  3. Actions  (multi-label)  — fine-grained actions (semicolon-separated in CSV)

The CSV must have columns: video_path, activity, actions, species.
The 'actions' column uses semicolons to separate multiple concurrent actions.

The set of candidate labels for each task is derived automatically from all
unique values in the CSV.

Usage:
    conda activate videoprism
    python zs-label-only.py --csv /path/to/test-mod-B.csv --video_dir /data/clips

    # Custom prompt templates:
    python zs-label-only.py --csv /path/to/test-mod-B.csv \
        --video_dir /data/clips \
        --species_template "a video of a {}." \
        --activity_template "a video of a {} {}." \
        --action_template "a video of a {} {}."

Example CSV (test-mod-B.csv):
    video_path,activity,actions,species
    clip01.mp4,foraging,walking;none,red_deer
    clip02.mp4,courtship,standing_head_up;vocalizing,red_deer
    clip03.mp4,camera_reaction,sniffing;none,hare
"""

import argparse
import collections
import csv
import datetime
import os
import sys

import jax
import jax.numpy as jnp
import mediapy
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score, f1_score

from videoprism import models as vp

tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")

NUM_FRAMES = 16
FRAME_SIZE = 288
DEFAULT_SPECIES_TEMPLATE = "a video of a {}."
DEFAULT_ACTIVITY_TEMPLATE = "a video of a {} {}."
DEFAULT_ACTION_TEMPLATE = "a video of a {} {}."


def read_and_preprocess_video(filename: str) -> np.ndarray:
    """Reads a video file and returns float32 [T, H, W, 3] in [0, 1]."""
    frames = mediapy.read_video(filename)

    frame_indices = np.linspace(
        0, len(frames), num=NUM_FRAMES, endpoint=False, dtype=np.int32
    )
    frames = np.array([frames[i] for i in frame_indices])

    h, w = frames.shape[1], frames.shape[2]
    scale = max(FRAME_SIZE / h, FRAME_SIZE / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    if (new_h, new_w) != (h, w):
        frames = mediapy.resize_video(frames, shape=(new_h, new_w))

    top = (frames.shape[1] - FRAME_SIZE) // 2
    left = (frames.shape[2] - FRAME_SIZE) // 2
    frames = frames[:, top : top + FRAME_SIZE, left : left + FRAME_SIZE]

    return mediapy.to_float01(frames)


def load_csv(csv_path: str, video_dir: str) -> list[dict]:
    """Loads the CSV manifest and returns a list of row dicts.

    Each dict has keys: video_path, activity (str), actions (list[str]), species (str).
    Relative video paths are resolved against video_dir.
    """
    entries = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = [col.strip().lower() for col in next(reader)]

        expected = ["video_path", "activity", "actions", "species"]
        if header != expected:
            raise ValueError(
                f"CSV header must be {expected}, got {header}"
            )

        for row_num, row in enumerate(reader, start=2):
            if len(row) < 4:
                raise ValueError(
                    f"Row {row_num}: expected 4 columns, got {len(row)}"
                )
            video_path = row[0].strip()
            activity = row[1].strip()
            actions_raw = row[2].strip()
            species = row[3].strip()

            actions = [a.strip() for a in actions_raw.split(";") if a.strip()]

            if not os.path.isabs(video_path):
                video_path = os.path.join(video_dir, video_path)

            if not os.path.isfile(video_path):
                raise FileNotFoundError(
                    f"Row {row_num}: video not found: {video_path}"
                )
            if not activity:
                raise ValueError(f"Row {row_num}: no activity specified")
            if not species:
                raise ValueError(f"Row {row_num}: no species specified")
            if not actions:
                raise ValueError(f"Row {row_num}: no actions specified")

            entries.append({
                "video_path": video_path,
                "activity": activity,
                "actions": actions,
                "species": species,
            })

    if not entries:
        raise ValueError(f"CSV file is empty (no data rows): {csv_path}")

    return entries


def collect_label_sets(entries: list[dict]) -> tuple[list[str], list[str], list[str]]:
    """Extracts sorted unique labels for species, activities, and actions."""
    species_set: dict[str, None] = {}
    activity_set: dict[str, None] = {}
    action_set: dict[str, None] = {}

    for entry in entries:
        species_set[entry["species"]] = None
        activity_set[entry["activity"]] = None
        for act in entry["actions"]:
            action_set[act] = None

    return list(species_set.keys()), list(activity_set.keys()), list(action_set.keys())


def get_video_embedding(video_path: str, forward_fn, dummy_text_ids, dummy_text_paddings) -> np.ndarray:
    """Computes the video embedding for a single clip."""
    frames = read_and_preprocess_video(video_path)
    video_input = jnp.asarray(frames[None, ...])
    video_embeddings, _, _ = forward_fn(video_input, dummy_text_ids, dummy_text_paddings)
    return np.array(video_embeddings).reshape(-1)


def classify_single_label(
    video_emb: np.ndarray,
    text_embeddings: np.ndarray,
    temperature: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-label classification via softmax over cosine similarities.

    Returns (probs, top_indices) where probs are softmax-normalized scores.
    """
    similarity = np.dot(text_embeddings, video_emb)
    logits = similarity / temperature
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    top_indices = np.argsort(probs)[::-1]
    return probs, top_indices


def classify_multi_label(
    video_emb: np.ndarray,
    text_embeddings: np.ndarray,
    temperature: float = 0.01,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-label classification via per-class sigmoid over cosine similarities.

    Returns (scores, predictions) where scores are sigmoid-normalized per-class
    and predictions is a binary array indicating which labels exceed threshold.
    """
    similarity = np.dot(text_embeddings, video_emb)
    logits = similarity / temperature
    scores = 1.0 / (1.0 + np.exp(-logits))
    predictions = (scores >= threshold).astype(np.int32)
    return scores, predictions


def compute_single_label_metrics(
    all_logits: np.ndarray,
    all_labels: np.ndarray,
) -> dict:
    """Computes mAP, top-1 accuracy, and mean class accuracy for single-label."""
    num_classes = all_logits.shape[-1]

    labels_onehot = np.zeros((len(all_labels), num_classes), dtype=np.float32)
    for i, lbl in enumerate(all_labels):
        labels_onehot[i, lbl] = 1.0

    present = labels_onehot.sum(axis=0) > 0
    if present.sum() > 1:
        map_score = float(average_precision_score(
            labels_onehot[:, present], all_logits[:, present], average="macro"
        ))
    else:
        map_score = float("nan")

    per_class_ap = {}
    for c in range(num_classes):
        if labels_onehot[:, c].sum() > 0:
            per_class_ap[c] = float(average_precision_score(
                labels_onehot[:, c], all_logits[:, c]
            ))

    preds_top1 = np.argmax(all_logits, axis=-1)
    top1 = float(np.mean(preds_top1 == all_labels))

    correct = np.zeros(num_classes, dtype=np.float32)
    total = np.zeros(num_classes, dtype=np.float32)
    for lbl, pred in zip(all_labels, preds_top1):
        total[lbl] += 1
        correct[lbl] += int(lbl == pred)
    classes_present = total > 0
    mean_class_acc = float(np.mean(correct[classes_present] / total[classes_present]))

    return {
        "map": map_score,
        "top1": top1,
        "mean_class_acc": mean_class_acc,
        "per_class_ap": per_class_ap,
    }


def compute_multi_label_metrics(
    all_scores: np.ndarray,
    all_labels: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Computes multi-label metrics: mAP, per-class AP, macro/micro F1.

    Args:
        all_scores: float32 [N, num_classes] — sigmoid scores per class.
        all_labels: int32 [N, num_classes] — binary ground-truth labels.
        threshold: threshold for binary predictions.
    """
    num_classes = all_scores.shape[-1]

    present = all_labels.sum(axis=0) > 0
    if present.sum() > 0:
        map_score = float(average_precision_score(
            all_labels[:, present], all_scores[:, present], average="macro"
        ))
    else:
        map_score = float("nan")

    per_class_ap = {}
    for c in range(num_classes):
        if all_labels[:, c].sum() > 0:
            per_class_ap[c] = float(average_precision_score(
                all_labels[:, c], all_scores[:, c]
            ))

    preds_binary = (all_scores >= threshold).astype(np.int32)
    macro_f1 = float(f1_score(all_labels, preds_binary, average="macro", zero_division=0))
    micro_f1 = float(f1_score(all_labels, preds_binary, average="micro", zero_division=0))

    per_class_f1 = f1_score(all_labels, preds_binary, average=None, zero_division=0)

    return {
        "map": map_score,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_class_ap": per_class_ap,
        "per_class_f1": per_class_f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot multi-task animal behaviour classification with VideoPrism"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV file with columns: video_path, activity, actions, species.",
    )
    parser.add_argument(
        "--video_dir",
        required=True,
        help="Directory where the video files listed in the CSV are stored.",
    )
    parser.add_argument(
        "--species_template",
        default=DEFAULT_SPECIES_TEMPLATE,
        help="Prompt template for species prediction with one placeholder (species).",
    )
    parser.add_argument(
        "--activity_template",
        default=DEFAULT_ACTIVITY_TEMPLATE,
        help="Prompt template for activity prediction with two placeholders (species, activity).",
    )
    parser.add_argument(
        "--action_template",
        default=DEFAULT_ACTION_TEMPLATE,
        help="Prompt template for action prediction with two placeholders (species, action).",
    )
    parser.add_argument(
        "--action_threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for multi-label action predictions (default: 0.5).",
    )
    parser.add_argument(
        "--model",
        default="videoprism_lvt_public_v1_base",
        choices=["videoprism_lvt_public_v1_base", "videoprism_lvt_public_v1_large"],
        help="Model variant to use",
    )
    parser.add_argument(
        "--log_dir",
        default="logs",
        help="Directory where the log file will be saved (default: logs)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    if not os.path.isdir(args.video_dir):
        raise FileNotFoundError(f"Video directory not found: {args.video_dir}")

    # Set up logging
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"zs_label_only_{timestamp}.log")

    class _Tee:
        def __init__(self, log_file, stream):
            self.log_file = log_file
            self.stream = stream

        def write(self, msg):
            self.stream.write(msg)
            self.log_file.write(msg)

        def flush(self):
            self.stream.flush()
            self.log_file.flush()

    log_file = open(log_path, "w")
    sys.stdout = _Tee(log_file, sys.__stdout__)
    print(f"Logging to: {log_path}")

    entries = load_csv(args.csv, args.video_dir)
    all_species, all_activities, all_actions = collect_label_sets(entries)

    print(f"JAX version:  {jax.__version__}")
    print(f"JAX platform: {jax.extend.backend.get_backend().platform}")
    print(f"JAX devices:  {jax.device_count()}")
    print(f"\nCSV file:   {args.csv}")
    print(f"Video dir:  {args.video_dir}")
    print(f"Videos:     {len(entries)}")
    print(f"\nLabel sets:")
    print(f"  Species    ({len(all_species)}): {all_species}")
    print(f"  Activities ({len(all_activities)}): {all_activities}")
    print(f"  Actions    ({len(all_actions)}): {all_actions}")

    # Load model
    print(f"\nLoading model: {args.model} ...")
    flax_model = vp.get_model(args.model)
    loaded_state = vp.load_pretrained_weights(args.model)
    text_tokenizer = vp.load_text_tokenizer("c4_en")

    @jax.jit
    def forward_fn(inputs, text_token_ids, text_paddings):
        return flax_model.apply(
            loaded_state, inputs, text_token_ids, text_paddings, train=False
        )

    # Build and tokenize text queries for species classification
    species_queries = [args.species_template.format(sp) for sp in all_species]
    species_text_ids, species_text_paddings = vp.tokenize_texts(text_tokenizer, species_queries)

    print(f"\nSpecies prompts:")
    for q in species_queries:
        print(f"  {q}")

    # Pre-compute text embeddings for species (single forward pass)
    # We need a dummy video to get text embeddings; use the first video
    dummy_frames = read_and_preprocess_video(entries[0]["video_path"])
    dummy_video = jnp.asarray(dummy_frames[None, ...])

    _, species_text_embs, _ = forward_fn(dummy_video, species_text_ids, species_text_paddings)
    species_text_embs = np.array(species_text_embs)

    # Pre-compute text embeddings for activities per species
    # (activity prompts are contextualized with species name)
    activity_text_embs_by_species = {}
    activity_queries_by_species = {}
    for sp in all_species:
        queries = [args.activity_template.format(sp, act) for act in all_activities]
        activity_queries_by_species[sp] = queries
        text_ids, text_paddings = vp.tokenize_texts(text_tokenizer, queries)
        _, t_embs, _ = forward_fn(dummy_video, text_ids, text_paddings)
        activity_text_embs_by_species[sp] = np.array(t_embs)

    print(f"\nActivity prompts (example for {all_species[0]}):")
    for q in activity_queries_by_species[all_species[0]]:
        print(f"  {q}")

    # Pre-compute text embeddings for actions per species
    action_text_embs_by_species = {}
    action_queries_by_species = {}
    for sp in all_species:
        queries = [args.action_template.format(sp, act) for act in all_actions]
        action_queries_by_species[sp] = queries
        text_ids, text_paddings = vp.tokenize_texts(text_tokenizer, queries)
        _, t_embs, _ = forward_fn(dummy_video, text_ids, text_paddings)
        action_text_embs_by_species[sp] = np.array(t_embs)

    print(f"\nAction prompts (example for {all_species[0]}):")
    for q in action_queries_by_species[all_species[0]]:
        print(f"  {q}")

    # Use the dummy text for video embedding extraction
    # (we only need text_ids/paddings to satisfy the model signature)
    dummy_text_ids = species_text_ids
    dummy_text_paddings = species_text_paddings

    # Classify each video
    print(f"\n{'='*70}")
    print(f"Classifying {len(entries)} videos (species + activity + actions)...")
    print(f"{'='*70}")

    # Storage for metrics computation
    all_species_logits = []
    all_species_labels = []
    all_activity_logits = []
    all_activity_labels = []
    all_action_scores = []
    all_action_labels = []
    results = []

    for i, entry in enumerate(entries):
        video_path = entry["video_path"]
        gt_species = entry["species"]
        gt_activity = entry["activity"]
        gt_actions = entry["actions"]

        species_idx = all_species.index(gt_species)
        activity_idx = all_activities.index(gt_activity)
        action_binary = np.zeros(len(all_actions), dtype=np.int32)
        for act in gt_actions:
            action_binary[all_actions.index(act)] = 1

        print(f"\n  [{i+1}/{len(entries)}] {os.path.basename(video_path)}")

        # Get video embedding
        video_emb = get_video_embedding(video_path, forward_fn, dummy_text_ids, dummy_text_paddings)

        # Species prediction (single-label)
        sp_probs, sp_top = classify_single_label(video_emb, species_text_embs)
        pred_species = all_species[sp_top[0]]

        # Activity prediction using ground-truth species context
        act_embs = activity_text_embs_by_species[gt_species]
        act_probs, act_top = classify_single_label(video_emb, act_embs)
        pred_activity = all_activities[act_top[0]]

        # Actions prediction using ground-truth species context (multi-label)
        action_embs = action_text_embs_by_species[gt_species]
        action_scores, action_preds = classify_multi_label(
            video_emb, action_embs, threshold=args.action_threshold
        )
        pred_actions = [all_actions[j] for j in range(len(all_actions)) if action_preds[j]]
        if not pred_actions:
            pred_actions = [all_actions[np.argmax(action_scores)]]

        # Store for metrics
        all_species_logits.append(sp_probs)
        all_species_labels.append(species_idx)
        all_activity_logits.append(act_probs)
        all_activity_labels.append(activity_idx)
        all_action_scores.append(action_scores)
        all_action_labels.append(action_binary)

        sp_ok = pred_species == gt_species
        act_ok = pred_activity == gt_activity
        actions_ok = set(pred_actions) == set(gt_actions)

        results.append({
            "video": video_path,
            "gt_species": gt_species,
            "pred_species": pred_species,
            "species_correct": sp_ok,
            "gt_activity": gt_activity,
            "pred_activity": pred_activity,
            "activity_correct": act_ok,
            "gt_actions": gt_actions,
            "pred_actions": pred_actions,
            "actions_correct": actions_ok,
            "species_conf": float(sp_probs[sp_top[0]]),
            "activity_conf": float(act_probs[act_top[0]]),
        })

        sp_status = "OK" if sp_ok else "MISS"
        act_status = "OK" if act_ok else "MISS"
        act_str_status = "OK" if actions_ok else "MISS"
        print(f"    Species:  {pred_species:<12} (gt: {gt_species}) [{sp_status}]")
        print(f"    Activity: {pred_activity:<16} (gt: {gt_activity}) [{act_status}]")
        print(f"    Actions:  {';'.join(pred_actions):<30} (gt: {';'.join(gt_actions)}) [{act_str_status}]")

    # Per-video summary table
    print(f"\n{'='*70}")
    print("PER-VIDEO RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Video':<40} {'Sp':>3} {'Act':>4} {'Actions':>7}")
    print(f"{'-'*40} {'-'*3} {'-'*4} {'-'*7}")
    for r in results:
        name = os.path.basename(r["video"])
        if len(name) > 38:
            name = name[:35] + "..."
        sp_mark = "OK" if r["species_correct"] else "X"
        act_mark = "OK" if r["activity_correct"] else "X"
        actions_mark = "OK" if r["actions_correct"] else "X"
        print(f"{name:<40} {sp_mark:>3} {act_mark:>4} {actions_mark:>7}")

    # Compute metrics
    print(f"\n{'='*70}")
    print("METRICS")
    print(f"{'='*70}")

    # Species metrics
    sp_logits_arr = np.stack(all_species_logits)
    sp_labels_arr = np.array(all_species_labels, dtype=np.int32)
    sp_metrics = compute_single_label_metrics(sp_logits_arr, sp_labels_arr)

    print(f"\n  SPECIES ({len(all_species)} classes, {len(entries)} samples)")
    print(f"    mAP            : {sp_metrics['map']*100:.2f}%")
    print(f"    Top-1 Acc      : {sp_metrics['top1']*100:.2f}%")
    print(f"    Mean Class Acc : {sp_metrics['mean_class_acc']*100:.2f}%")
    print(f"    Per-class AP:")
    for c_idx, ap in sp_metrics["per_class_ap"].items():
        print(f"      {all_species[c_idx]:<20} {ap*100:.2f}%")

    # Activity metrics
    act_logits_arr = np.stack(all_activity_logits)
    act_labels_arr = np.array(all_activity_labels, dtype=np.int32)
    act_metrics = compute_single_label_metrics(act_logits_arr, act_labels_arr)

    print(f"\n  ACTIVITY ({len(all_activities)} classes, {len(entries)} samples)")
    print(f"    mAP            : {act_metrics['map']*100:.2f}%")
    print(f"    Top-1 Acc      : {act_metrics['top1']*100:.2f}%")
    print(f"    Mean Class Acc : {act_metrics['mean_class_acc']*100:.2f}%")
    print(f"    Per-class AP:")
    for c_idx, ap in act_metrics["per_class_ap"].items():
        print(f"      {all_activities[c_idx]:<20} {ap*100:.2f}%")

    # Actions metrics (multi-label)
    action_scores_arr = np.stack(all_action_scores)
    action_labels_arr = np.stack(all_action_labels)
    action_metrics = compute_multi_label_metrics(
        action_scores_arr, action_labels_arr, threshold=args.action_threshold
    )

    print(f"\n  ACTIONS — multi-label ({len(all_actions)} classes, {len(entries)} samples)")
    print(f"    mAP      : {action_metrics['map']*100:.2f}%")
    print(f"    Macro F1 : {action_metrics['macro_f1']*100:.2f}%")
    print(f"    Micro F1 : {action_metrics['micro_f1']*100:.2f}%")
    print(f"    Per-class AP / F1:")
    for c_idx in range(len(all_actions)):
        ap_str = f"{action_metrics['per_class_ap'][c_idx]*100:.2f}%" if c_idx in action_metrics["per_class_ap"] else "  N/A"
        f1_val = action_metrics["per_class_f1"][c_idx]
        print(f"      {all_actions[c_idx]:<25} AP: {ap_str:<8}  F1: {f1_val*100:.2f}%")

    # Overall summary
    print(f"\n  {'─'*50}")
    print(f"  OVERALL SUMMARY")
    print(f"    Species  Top-1 Acc : {sp_metrics['top1']*100:.2f}%")
    print(f"    Activity Top-1 Acc : {act_metrics['top1']*100:.2f}%")
    print(f"    Actions  mAP       : {action_metrics['map']*100:.2f}%")
    print(f"    Actions  Macro F1  : {action_metrics['macro_f1']*100:.2f}%")

    # Close log file
    print(f"\nLog saved to: {log_path}")
    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == "__main__":
    main()
