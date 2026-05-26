"""Fine-tune VideoPrism for multi-task classification on MammalPS benchmark_1.

Predicts species, activity, and actions simultaneously using a shared
FactorizedEncoder backbone (frozen) with three separate classification heads
(attention-pooler + per-task projection).

Usage:
    python train_benchmark1.py train [OPTIONS]   # training + validation
    python train_benchmark1.py test  [OPTIONS]   # test-set evaluation

See --help for all flags.
"""

import argparse
import csv
import datetime
import functools
import glob
import json
import math
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapy
import numpy as np
import optax
import PIL.Image
import tensorflow as tf
from flax import linen as nn
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

from videoprism import encoders
from videoprism import layers as vp_layers
from videoprism import models as vp

# ---------------------------------------------------------------------------
# Prevent TF from claiming GPU/TPU devices — JAX owns them.
# ---------------------------------------------------------------------------
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_FRAMES = 16
FRAME_SIZE = 180

SPECIES_CLASSES = [
    "fox",
    "hare",
    "red_deer",
    "roe_deer",
    "wolf",
]
SPECIES_TO_IDX = {name: i for i, name in enumerate(SPECIES_CLASSES)}
NUM_SPECIES = len(SPECIES_CLASSES)

ACTIVITY_CLASSES = [
    "camera_reaction",
    "chasing",
    "courtship",
    "escaping",
    "foraging",
    "grooming",
    "marking",
    "playing",
    "resting",
    "unknown",
    "vigilance",
]
ACTIVITY_TO_IDX = {name: i for i, name in enumerate(ACTIVITY_CLASSES)}
NUM_ACTIVITIES = len(ACTIVITY_CLASSES)

ACTION_CLASSES = [
    "bathing",
    "defecating",
    "drinking",
    "grazing",
    "jumping",
    "laying",
    "looking_at_camera",
    "none",
    "running",
    "scratching_antlers",
    "scratching_body",
    "scratching_hoof",
    "shaking_fur",
    "sniffing",
    "standing_head_down",
    "standing_head_up",
    "unknown",
    "urinating",
    "vocalizing",
    "walking",
]
ACTION_TO_IDX = {name: i for i, name in enumerate(ACTION_CLASSES)}
NUM_ACTIONS = len(ACTION_CLASSES)


# ===================================================================
# Multi-task Flax module
# ===================================================================

class MultiTaskClassifier(vp_layers.Module):
    """Shared FactorizedEncoder with three classification heads.

    Attributes:
        encoder_params: Config dict for FactorizedEncoder.
        num_species: Number of species classes.
        num_activities: Number of activity classes.
        num_actions: Number of action classes (multi-label).
    """

    encoder_params: dict = None
    num_species: int = 0
    num_activities: int = 0
    num_actions: int = 0

    @nn.compact
    def __call__(self, inputs, train=False):
        features, _ = encoders.FactorizedEncoder(
            name="encoder",
            dtype=self.dtype,
            fprop_dtype=self.fprop_dtype,
            **self.encoder_params,
        )(inputs, train=train, return_intermediate=False)

        embeddings = vp_layers.AttenTokenPoolingLayer(
            name="atten_pooler",
            num_heads=self.encoder_params["num_heads"],
            hidden_dim=self.encoder_params["model_dim"],
            num_queries=1,
            dtype=self.dtype,
            fprop_dtype=self.fprop_dtype,
        )(features, paddings=None, train=train)
        embeddings = jnp.squeeze(embeddings, axis=-2)

        species_logits = vp_layers.FeedForward(
            name="species_head",
            output_dim=self.num_species,
            activation_fn=vp_layers.identity,
            dtype=self.dtype,
            fprop_dtype=self.fprop_dtype,
        )(embeddings)

        activity_logits = vp_layers.FeedForward(
            name="activity_head",
            output_dim=self.num_activities,
            activation_fn=vp_layers.identity,
            dtype=self.dtype,
            fprop_dtype=self.fprop_dtype,
        )(embeddings)

        action_logits = vp_layers.FeedForward(
            name="action_head",
            output_dim=self.num_actions,
            activation_fn=vp_layers.identity,
            dtype=self.dtype,
            fprop_dtype=self.fprop_dtype,
        )(embeddings)

        return species_logits, activity_logits, action_logits


# ===================================================================
# Data loading
# ===================================================================

def load_csv_samples(
    csv_path: str, video_dir: str
) -> list[tuple[str, int, int, np.ndarray]]:
    """Read a benchmark_1 metadata CSV and return per-clip label tuples.

    Returns a list of (video_path, species_idx, activity_idx, action_multi_hot).
    Rows with unrecognised species or activity labels are skipped with a warning.
    """
    samples = []
    skipped = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            species = row["species"].strip()
            activity = row["activity"].strip()
            if species not in SPECIES_TO_IDX or activity not in ACTIVITY_TO_IDX:
                skipped += 1
                continue

            video_path = os.path.join(video_dir, row["video_path"].strip())
            species_idx = SPECIES_TO_IDX[species]
            activity_idx = ACTIVITY_TO_IDX[activity]

            action_vec = np.zeros(NUM_ACTIONS, dtype=np.float32)
            for a in row["actions"].strip().split(";"):
                a = a.strip()
                if a in ACTION_TO_IDX:
                    action_vec[ACTION_TO_IDX[a]] = 1.0
            samples.append((video_path, species_idx, activity_idx, action_vec))

    if skipped:
        print(f"  Warning: skipped {skipped} rows with unrecognised labels")
    return samples


def read_and_preprocess_frames(
    source: str,
    target_num_frames: int = NUM_FRAMES,
    target_frame_size: tuple[int, int] = (FRAME_SIZE, FRAME_SIZE),
) -> np.ndarray:
    """Load an MP4 clip and return float32 [T, H, W, 3] in [0, 1]."""
    frames = mediapy.read_video(source)
    n = len(frames)
    if n == 0:
        raise ValueError(f"Empty video: {source}")
    indices = np.linspace(0, n, num=target_num_frames, endpoint=False, dtype=np.int32)
    frames = np.asarray([frames[i] for i in indices])

    h, w = frames.shape[1], frames.shape[2]
    target_h, target_w = target_frame_size
    scale = max(target_h / h, target_w / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    if (new_h, new_w) != (h, w):
        frames = mediapy.resize_video(frames, shape=(new_h, new_w))
    top = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    frames = frames[:, top : top + target_h, left : left + target_w]
    return mediapy.to_float01(frames)


def random_sample_and_preprocess(
    frames: np.ndarray,
    target_num_frames: int = NUM_FRAMES,
    target_frame_size: tuple[int, int] = (FRAME_SIZE, FRAME_SIZE),
) -> np.ndarray:
    """Randomly sample *target_num_frames* from raw video frames and preprocess.

    Returns float32 [T, H, W, 3] in [0, 1].  Sampling is with replacement
    when the video has fewer frames than requested.
    """
    n = len(frames)
    indices = sorted(np.random.choice(n, size=target_num_frames, replace=(n < target_num_frames)))
    sampled = np.asarray([frames[i] for i in indices])

    h, w = sampled.shape[1], sampled.shape[2]
    target_h, target_w = target_frame_size
    scale = max(target_h / h, target_w / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    if (new_h, new_w) != (h, w):
        sampled = mediapy.resize_video(sampled, shape=(new_h, new_w))
    top = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    sampled = sampled[:, top : top + target_h, left : left + target_w]
    return mediapy.to_float01(sampled)


def compute_sample_weights(
    samples: list[tuple[str, int, int, np.ndarray]],
) -> np.ndarray:
    """Compute per-sample sampling weights from inverse label frequencies.

    For each clip the weight is the sum of inverse frequencies for every class
    present: 1/freq(species) + 1/freq(activity) + sum(1/freq(action_j) for
    each active action_j).  The returned array is normalised to sum to 1.
    """
    n = len(samples)

    species_counts = np.zeros(NUM_SPECIES, dtype=np.float64)
    activity_counts = np.zeros(NUM_ACTIVITIES, dtype=np.float64)
    action_counts = np.zeros(NUM_ACTIONS, dtype=np.float64)

    for _, sp, act, action_vec in samples:
        species_counts[sp] += 1
        activity_counts[act] += 1
        action_counts += action_vec.astype(np.float64)

    species_counts = np.maximum(species_counts, 1.0)
    activity_counts = np.maximum(activity_counts, 1.0)
    action_counts = np.maximum(action_counts, 1.0)

    inv_species = 1.0 / species_counts
    inv_activity = 1.0 / activity_counts
    inv_action = 1.0 / action_counts

    weights = np.empty(n, dtype=np.float64)
    for i, (_, sp, act, action_vec) in enumerate(samples):
        w = inv_species[sp] + inv_activity[act] + np.dot(action_vec, inv_action)
        weights[i] = w

    weights /= weights.sum()
    return weights


def make_batches(
    samples: list[tuple[str, int, int, np.ndarray]],
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = False,
    drop_remainder: bool = False,
    sample_weights: np.ndarray | None = None,
):
    """Yield (videos, species, activity, actions) batches with parallel I/O.

    When *sample_weights* is provided, clips are drawn via weighted sampling
    with replacement (one full epoch = len(samples) draws).  Otherwise the
    dataset is iterated sequentially (with optional shuffle).

    videos:   float32 [B, T, H, W, 3]
    species:  int32   [B]
    activity: int32   [B]
    actions:  float32 [B, NUM_ACTIONS]
    """
    if sample_weights is not None:
        n_draws = len(samples)
        indices = np.random.choice(len(samples), size=n_draws, replace=True, p=sample_weights)
        items = [samples[i] for i in indices]
    else:
        items = list(samples)
        if shuffle:
            random.shuffle(items)

    def _load(item):
        path, sp, act, action_vec = item
        return read_and_preprocess_frames(path), sp, act, action_vec

    buf_videos, buf_species, buf_activity, buf_actions = [], [], [], []
    chunk_size = num_workers * 2
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for start in range(0, len(items), chunk_size):
            chunk = items[start : start + chunk_size]
            for frames, sp, act, action_vec in pool.map(_load, chunk):
                buf_videos.append(frames)
                buf_species.append(sp)
                buf_activity.append(act)
                buf_actions.append(action_vec)
                if len(buf_videos) == batch_size:
                    yield (
                        np.stack(buf_videos).astype(np.float32),
                        np.array(buf_species, dtype=np.int32),
                        np.array(buf_activity, dtype=np.int32),
                        np.stack(buf_actions).astype(np.float32),
                    )
                    buf_videos, buf_species, buf_activity, buf_actions = [], [], [], []
    if buf_videos and not drop_remainder:
        yield (
            np.stack(buf_videos).astype(np.float32),
            np.array(buf_species, dtype=np.int32),
            np.array(buf_activity, dtype=np.int32),
            np.stack(buf_actions).astype(np.float32),
        )


# ===================================================================
# Model / optimizer
# ===================================================================

def build_model_and_params(model_size: str = "base"):
    """Create MultiTaskClassifier and inject pretrained encoder weights."""
    encoder_config = {
        "base": vp.CONFIGS["videoprism_v1_base"],
        "large": vp.CONFIGS["videoprism_v1_large"],
    }[model_size]
    model_name = {
        "base": "videoprism_public_v1_base",
        "large": "videoprism_public_v1_large",
    }[model_size]

    classifier = MultiTaskClassifier(
        encoder_params=encoder_config,
        num_species=NUM_SPECIES,
        num_activities=NUM_ACTIVITIES,
        num_actions=NUM_ACTIONS,
    )
    key = jax.random.PRNGKey(0)
    dummy = jnp.zeros((1, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3))
    variables = classifier.init(key, dummy, train=False)
    pretrained = vp.load_pretrained_weights(model_name)
    params = dict(variables["params"])
    params["encoder"] = pretrained["params"]
    print("Parameter subtrees:", list(params.keys()))
    return classifier, params


def build_optimizer(params: dict, learning_rate: float = 1e-4):
    """Partitioned optimizer: encoder frozen, all heads use Adam."""

    def _tag(subtree, label):
        return jax.tree.map(lambda _: label, subtree)

    param_labels = {
        k: _tag(v, "frozen" if k == "encoder" else "trainable")
        for k, v in params.items()
    }
    tx = optax.multi_transform(
        {
            "trainable": optax.adam(learning_rate=learning_rate),
            "frozen": optax.set_to_zero(),
        },
        param_labels,
    )
    opt_state = tx.init(params)
    print(
        "Optimizer ready — encoder: frozen | atten_pooler + "
        "species/activity/action heads: trainable"
    )
    return tx, opt_state


# ===================================================================
# Train / eval steps
# ===================================================================

def make_train_step(classifier, tx):
    """JIT-compiled multi-task training step."""

    @jax.jit
    def train_step(
        params, opt_state, batch_videos, batch_species, batch_activity, batch_actions
    ):
        def loss_fn(p):
            sp_logits, act_logits, action_logits = classifier.apply(
                {"params": p}, batch_videos, train=True
            )
            sp_loss = optax.softmax_cross_entropy_with_integer_labels(
                sp_logits, batch_species
            ).mean()
            act_loss = optax.softmax_cross_entropy_with_integer_labels(
                act_logits, batch_activity
            ).mean()
            action_loss = optax.sigmoid_binary_cross_entropy(
                action_logits, batch_actions
            ).mean()
            total_loss = sp_loss + act_loss + action_loss
            return total_loss, (sp_logits, act_logits, action_logits, sp_loss, act_loss, action_loss)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        sp_logits, act_logits, action_logits, sp_loss, act_loss, action_loss = aux
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        sp_acc = jnp.mean(jnp.argmax(sp_logits, axis=-1) == batch_species)
        act_acc = jnp.mean(jnp.argmax(act_logits, axis=-1) == batch_activity)

        metrics = {
            "loss": loss,
            "sp_loss": sp_loss,
            "act_loss": act_loss,
            "action_loss": action_loss,
            "sp_acc": sp_acc,
            "act_acc": act_acc,
        }
        return new_params, new_opt_state, metrics

    return train_step


def make_eval_step(classifier):
    """JIT-compiled inference step returning all three logit sets."""

    @jax.jit
    def eval_step(params, batch_videos):
        sp_logits, act_logits, action_logits = classifier.apply(
            {"params": params}, batch_videos, train=False
        )
        return sp_logits, act_logits, action_logits

    return eval_step


# ===================================================================
# Metrics
# ===================================================================

def average_precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """Compute Average Precision truncated at rank K.

    Ranks all samples by y_scores (descending), considers only the top-k
    positions, and computes AP normalised by min(k, num_relevant).
    """
    n_relevant = int(y_true.sum())
    if n_relevant == 0:
        return 0.0
    sorted_indices = np.argsort(-y_scores)[:k]
    y_true_sorted = y_true[sorted_indices].astype(np.float64)
    cumsum = np.cumsum(y_true_sorted)
    precisions = cumsum / np.arange(1, len(y_true_sorted) + 1)
    return float(np.sum(precisions * y_true_sorted) / min(k, n_relevant))


def compute_map_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    multi_label: bool = False,
    from_probs: bool = False,
    pred_threshold: float | None = None,
) -> dict:
    """Compute mAP, mAP Rank-1, mAP Rank-5, Macro mAP, and Macro F1 per class and overall.

    For single-label tasks the labels (int indices) are expanded to one-vs-rest
    binary vectors and softmax probabilities are used as scores.
    For multi-label tasks the targets are already multi-hot and sigmoid
    probabilities are used.

    When *from_probs* is True the first argument is treated as probabilities
    (skip softmax/sigmoid).  When *pred_threshold* is set, hard predictions
    for single-label tasks use that threshold instead of argmax.
    """
    num_classes = len(class_names)

    if multi_label:
        if from_probs:
            probs = logits.astype(np.float64)
        else:
            probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        binary_labels = labels.astype(np.float64)
        threshold = pred_threshold if pred_threshold is not None else 0.5
        binary_preds = (probs >= threshold).astype(np.float64)
    else:
        if from_probs:
            probs = logits.astype(np.float64)
        else:
            logits_f = logits.astype(np.float64)
            exp_logits = np.exp(logits_f - logits_f.max(axis=-1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        binary_labels = np.zeros((len(labels), num_classes), dtype=np.float64)
        for i, lbl in enumerate(labels):
            binary_labels[i, lbl] = 1.0
        if pred_threshold is not None:
            binary_preds = (probs >= pred_threshold).astype(np.float64)
        else:
            hard_preds = np.argmax(probs, axis=-1)
            binary_preds = np.zeros_like(binary_labels)
            for i, pred in enumerate(hard_preds):
                binary_preds[i, pred] = 1.0

    per_class = {}
    ap_list, ap1_list, ap5_list, f1_list = [], [], [], []

    for c, name in enumerate(class_names):
        y_true = binary_labels[:, c]
        y_scores = probs[:, c]
        y_pred = binary_preds[:, c]
        n_positive = int(y_true.sum())

        if n_positive > 0:
            ap = float(average_precision_score(y_true, y_scores))
            ap1 = average_precision_at_k(y_true, y_scores, k=1)
            ap5 = average_precision_at_k(y_true, y_scores, k=5)
        else:
            ap, ap1, ap5 = 0.0, 0.0, 0.0

        tp = float(np.sum(y_true * y_pred))
        fp = float(np.sum((1 - y_true) * y_pred))
        fn = float(np.sum(y_true * (1 - y_pred)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        class_f1 = (2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0 else 0.0)

        ap_list.append(ap)
        ap1_list.append(ap1)
        ap5_list.append(ap5)
        f1_list.append(class_f1)

        per_class[name] = {
            "n_positive": n_positive,
            "ap": ap,
            "ap_rank1": ap1,
            "ap_rank5": ap5,
            "f1": class_f1,
        }

    present = [c for c in range(num_classes) if binary_labels[:, c].sum() > 0]
    mean_ap = float(np.mean([ap_list[c] for c in present])) if present else 0.0
    mean_ap_rank1 = float(np.mean([ap1_list[c] for c in present])) if present else 0.0
    mean_ap_rank5 = float(np.mean([ap5_list[c] for c in present])) if present else 0.0
    macro_f1 = float(np.mean([f1_list[c] for c in present])) if present else 0.0

    return {
        "mAP": mean_ap,
        "mAP_rank1": mean_ap_rank1,
        "mAP_rank5": mean_ap_rank5,
        "macro_mAP": mean_ap,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def evaluate(eval_step_fn, params, batches, n_batches: int) -> dict:
    """Full evaluation pass returning mAP, mAP Rank-1, mAP Rank-5, Macro mAP, and Macro F1 per task."""
    all_sp, all_act, all_action = [], [], []
    all_sp_labels, all_act_labels, all_action_labels = [], [], []

    pbar = tqdm(batches, total=n_batches, desc="Evaluating", unit="batch")
    for batch_videos, batch_species, batch_activity, batch_actions in pbar:
        sp_logits, act_logits, action_logits = eval_step_fn(
            params, jnp.asarray(batch_videos)
        )
        all_sp.append(np.asarray(sp_logits))
        all_act.append(np.asarray(act_logits))
        all_action.append(np.asarray(action_logits))
        all_sp_labels.append(batch_species)
        all_act_labels.append(batch_activity)
        all_action_labels.append(batch_actions)

    all_sp = np.concatenate(all_sp, axis=0)
    all_act = np.concatenate(all_act, axis=0)
    all_action = np.concatenate(all_action, axis=0)
    all_sp_labels = np.concatenate(all_sp_labels, axis=0)
    all_act_labels = np.concatenate(all_act_labels, axis=0)
    all_action_labels = np.concatenate(all_action_labels, axis=0)

    sp_metrics = compute_map_metrics(all_sp, all_sp_labels, SPECIES_CLASSES, multi_label=False)
    act_metrics = compute_map_metrics(all_act, all_act_labels, ACTIVITY_CLASSES, multi_label=False)
    action_metrics = compute_map_metrics(all_action, all_action_labels, ACTION_CLASSES, multi_label=True)

    print("  --- Species ---")
    print(f"    mAP        : {sp_metrics['mAP'] * 100:.2f}%")
    print(f"    mAP Rank-1 : {sp_metrics['mAP_rank1'] * 100:.2f}%")
    print(f"    mAP Rank-5 : {sp_metrics['mAP_rank5'] * 100:.2f}%")
    print(f"    Macro mAP  : {sp_metrics['macro_mAP'] * 100:.2f}%")
    print(f"    Macro F1   : {sp_metrics['macro_f1'] * 100:.2f}%")
    for name, info in sp_metrics["per_class"].items():
        print(
            f"      {name:15s}  n={info['n_positive']:5d}  "
            f"AP={info['ap'] * 100:.1f}%  "
            f"AP@1={info['ap_rank1'] * 100:.1f}%  "
            f"AP@5={info['ap_rank5'] * 100:.1f}%  "
            f"F1={info['f1'] * 100:.1f}%"
        )

    print("  --- Activity ---")
    print(f"    mAP        : {act_metrics['mAP'] * 100:.2f}%")
    print(f"    mAP Rank-1 : {act_metrics['mAP_rank1'] * 100:.2f}%")
    print(f"    mAP Rank-5 : {act_metrics['mAP_rank5'] * 100:.2f}%")
    print(f"    Macro mAP  : {act_metrics['macro_mAP'] * 100:.2f}%")
    print(f"    Macro F1   : {act_metrics['macro_f1'] * 100:.2f}%")
    for name, info in act_metrics["per_class"].items():
        print(
            f"      {name:20s}  n={info['n_positive']:5d}  "
            f"AP={info['ap'] * 100:.1f}%  "
            f"AP@1={info['ap_rank1'] * 100:.1f}%  "
            f"AP@5={info['ap_rank5'] * 100:.1f}%  "
            f"F1={info['f1'] * 100:.1f}%"
        )

    print("  --- Actions (multi-label) ---")
    print(f"    mAP        : {action_metrics['mAP'] * 100:.2f}%")
    print(f"    mAP Rank-1 : {action_metrics['mAP_rank1'] * 100:.2f}%")
    print(f"    mAP Rank-5 : {action_metrics['mAP_rank5'] * 100:.2f}%")
    print(f"    Macro mAP  : {action_metrics['macro_mAP'] * 100:.2f}%")
    print(f"    Macro F1   : {action_metrics['macro_f1'] * 100:.2f}%")
    for name, info in action_metrics["per_class"].items():
        print(
            f"      {name:20s}  n+={info['n_positive']:5d}  "
            f"AP={info['ap'] * 100:.1f}%  "
            f"AP@1={info['ap_rank1'] * 100:.1f}%  "
            f"AP@5={info['ap_rank5'] * 100:.1f}%  "
            f"F1={info['f1'] * 100:.1f}%"
        )

    return {
        "species": sp_metrics,
        "activity": act_metrics,
        "actions": action_metrics,
    }


def evaluate_multi_sample(
    eval_step_fn,
    params,
    samples: list[tuple[str, int, int, np.ndarray]],
    num_samples: int = 10,
    min_duration: float = 0.5,
) -> dict:
    """Multi-sample test evaluation with duration filtering.

    For each valid clip (>= *min_duration* seconds):
      1. Randomly sample 16 frames and run inference to get a prediction vector.
      2. Repeat *num_samples* times.
      3. Average the probability vectors across the samples.
      4. Threshold at 0.5 (softmax for species/activity, sigmoid for actions).
    """
    valid_samples: list[tuple[str, int, int, np.ndarray]] = []
    skipped = 0

    print(f"  Filtering clips shorter than {min_duration}s ...")
    for sample in samples:
        path = sample[0]
        try:
            meta = mediapy._get_video_metadata(path)
            dur = meta.num_images / meta.fps if meta.fps > 0 else 0.0
        except Exception:
            dur = 0.0
        if dur < min_duration:
            skipped += 1
            continue
        valid_samples.append(sample)

    print(f"  Skipped {skipped} clips < {min_duration}s, "
          f"{len(valid_samples)} valid clips remaining")

    all_sp_probs: list[np.ndarray] = []
    all_act_probs: list[np.ndarray] = []
    all_action_probs: list[np.ndarray] = []
    all_sp_labels: list[int] = []
    all_act_labels: list[int] = []
    all_action_labels: list[np.ndarray] = []

    pbar = tqdm(valid_samples, desc="Multi-sample eval", unit="clip")
    for path, sp, act, action_vec in pbar:
        raw_frames = mediapy.read_video(path)

        frame_sets = []
        for _ in range(num_samples):
            frame_sets.append(random_sample_and_preprocess(raw_frames))
        batch = np.stack(frame_sets).astype(np.float32)

        sp_logits, act_logits, action_logits = eval_step_fn(
            params, jnp.asarray(batch)
        )

        sp_probs = np.asarray(jax.nn.softmax(sp_logits, axis=-1))
        act_probs = np.asarray(jax.nn.softmax(act_logits, axis=-1))
        action_probs = np.asarray(jax.nn.sigmoid(action_logits))

        all_sp_probs.append(sp_probs.mean(axis=0))
        all_act_probs.append(act_probs.mean(axis=0))
        all_action_probs.append(action_probs.mean(axis=0))
        all_sp_labels.append(sp)
        all_act_labels.append(act)
        all_action_labels.append(action_vec)

    all_sp_probs_arr = np.stack(all_sp_probs)
    all_act_probs_arr = np.stack(all_act_probs)
    all_action_probs_arr = np.stack(all_action_probs)
    all_sp_labels_arr = np.array(all_sp_labels, dtype=np.int32)
    all_act_labels_arr = np.array(all_act_labels, dtype=np.int32)
    all_action_labels_arr = np.stack(all_action_labels)

    sp_metrics = compute_map_metrics(
        all_sp_probs_arr, all_sp_labels_arr, SPECIES_CLASSES,
        multi_label=False, from_probs=True, pred_threshold=0.5,
    )
    act_metrics = compute_map_metrics(
        all_act_probs_arr, all_act_labels_arr, ACTIVITY_CLASSES,
        multi_label=False, from_probs=True, pred_threshold=0.5,
    )
    action_metrics = compute_map_metrics(
        all_action_probs_arr, all_action_labels_arr, ACTION_CLASSES,
        multi_label=True, from_probs=True, pred_threshold=0.5,
    )

    print("  --- Species ---")
    print(f"    mAP        : {sp_metrics['mAP'] * 100:.2f}%")
    print(f"    mAP Rank-1 : {sp_metrics['mAP_rank1'] * 100:.2f}%")
    print(f"    mAP Rank-5 : {sp_metrics['mAP_rank5'] * 100:.2f}%")
    print(f"    Macro mAP  : {sp_metrics['macro_mAP'] * 100:.2f}%")
    print(f"    Macro F1   : {sp_metrics['macro_f1'] * 100:.2f}%")
    for name, info in sp_metrics["per_class"].items():
        print(
            f"      {name:15s}  n={info['n_positive']:5d}  "
            f"AP={info['ap'] * 100:.1f}%  "
            f"AP@1={info['ap_rank1'] * 100:.1f}%  "
            f"AP@5={info['ap_rank5'] * 100:.1f}%  "
            f"F1={info['f1'] * 100:.1f}%"
        )

    print("  --- Activity ---")
    print(f"    mAP        : {act_metrics['mAP'] * 100:.2f}%")
    print(f"    mAP Rank-1 : {act_metrics['mAP_rank1'] * 100:.2f}%")
    print(f"    mAP Rank-5 : {act_metrics['mAP_rank5'] * 100:.2f}%")
    print(f"    Macro mAP  : {act_metrics['macro_mAP'] * 100:.2f}%")
    print(f"    Macro F1   : {act_metrics['macro_f1'] * 100:.2f}%")
    for name, info in act_metrics["per_class"].items():
        print(
            f"      {name:20s}  n={info['n_positive']:5d}  "
            f"AP={info['ap'] * 100:.1f}%  "
            f"AP@1={info['ap_rank1'] * 100:.1f}%  "
            f"AP@5={info['ap_rank5'] * 100:.1f}%  "
            f"F1={info['f1'] * 100:.1f}%"
        )

    print("  --- Actions (multi-label) ---")
    print(f"    mAP        : {action_metrics['mAP'] * 100:.2f}%")
    print(f"    mAP Rank-1 : {action_metrics['mAP_rank1'] * 100:.2f}%")
    print(f"    mAP Rank-5 : {action_metrics['mAP_rank5'] * 100:.2f}%")
    print(f"    Macro mAP  : {action_metrics['macro_mAP'] * 100:.2f}%")
    print(f"    Macro F1   : {action_metrics['macro_f1'] * 100:.2f}%")
    for name, info in action_metrics["per_class"].items():
        print(
            f"      {name:20s}  n+={info['n_positive']:5d}  "
            f"AP={info['ap'] * 100:.1f}%  "
            f"AP@1={info['ap_rank1'] * 100:.1f}%  "
            f"AP@5={info['ap_rank5'] * 100:.1f}%  "
            f"F1={info['f1'] * 100:.1f}%"
        )

    return {
        "species": sp_metrics,
        "activity": act_metrics,
        "actions": action_metrics,
    }


# ===================================================================
# Checkpointing
# ===================================================================

def save_checkpoint(ckpt_dir: str, params, opt_state, epoch: int, global_step: int) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"checkpoint_step{global_step:07d}.pkl")
    state = {
        "params": jax.tree.map(np.array, params),
        "opt_state": jax.tree.map(np.array, opt_state),
        "epoch": epoch,
        "global_step": global_step,
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)
    return path


def load_checkpoint(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ===================================================================
# Training loop
# ===================================================================

def train(
    classifier,
    params,
    tx,
    opt_state,
    train_samples,
    val_samples,
    args,
):
    """Full training loop with per-epoch validation and checkpointing."""
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_ckpt_dir = os.path.join(args.ckpt_dir, run_id)
    start_epoch = 1
    global_step = 0
    recent_ckpts: list[str] = []

    print("Computing weighted batch sampling probabilities ...")
    train_weights = compute_sample_weights(train_samples)
    print(
        f"  weight range: [{train_weights.min():.6e}, {train_weights.max():.6e}]  "
        f"effective ratio: {train_weights.max() / train_weights.min():.1f}x"
    )

    if args.resume_ckpt_dir:
        ckpt_files = sorted(glob.glob(os.path.join(args.resume_ckpt_dir, "checkpoint_step*.pkl")))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints in {args.resume_ckpt_dir}")
        state = load_checkpoint(ckpt_files[-1])
        params = state["params"]
        opt_state = state["opt_state"]
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        run_ckpt_dir = args.resume_ckpt_dir
        recent_ckpts = list(ckpt_files[-args.keep_recent :])
        print(f"Resuming from epoch {state['epoch']} (step {global_step})")

    print(f"Run ID: {run_id}  |  Checkpoints → {run_ckpt_dir}")

    train_step_fn = make_train_step(classifier, tx)
    eval_step_fn = make_eval_step(classifier)
    history = {
        "loss": [],
        "sp_loss": [],
        "act_loss": [],
        "action_loss": [],
        "sp_acc": [],
        "act_acc": [],
        "val_sp_map": [],
        "val_act_map": [],
        "val_action_map": [],
    }

    best_val_metric = -1.0
    best_ckpt_path: str | None = None
    last_ckpt_path: str | None = None

    n_train_batches = math.ceil(len(train_samples) / args.batch_size)
    n_val_batches = math.ceil(len(val_samples) / args.batch_size) if val_samples else 0

    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'=' * 60}")

        train_batches = make_batches(
            train_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_weights=train_weights,
        )

        epoch_metrics = None
        batch_bar = tqdm(
            train_batches,
            total=n_train_batches,
            desc="  Training",
            unit="batch",
            leave=False,
        )
        for batch_videos, batch_species, batch_activity, batch_actions in batch_bar:
            params, opt_state, epoch_metrics = train_step_fn(
                params,
                opt_state,
                jnp.asarray(batch_videos),
                jnp.asarray(batch_species),
                jnp.asarray(batch_activity),
                jnp.asarray(batch_actions),
            )
            global_step += 1
            batch_bar.set_postfix(
                loss=f"{float(epoch_metrics['loss']):.4f}",
                sp=f"{float(epoch_metrics['sp_acc']):.3f}",
                act=f"{float(epoch_metrics['act_acc']):.3f}",
            )

            if global_step % args.ckpt_every == 0:
                last_ckpt_path = save_checkpoint(
                    run_ckpt_dir, params, opt_state, epoch, global_step
                )
                recent_ckpts.append(last_ckpt_path)
                if len(recent_ckpts) > args.keep_recent:
                    evicted = recent_ckpts.pop(0)
                    if evicted != best_ckpt_path:
                        os.remove(evicted)

        if epoch_metrics is not None:
            history["loss"].append(float(epoch_metrics["loss"]))
            history["sp_loss"].append(float(epoch_metrics["sp_loss"]))
            history["act_loss"].append(float(epoch_metrics["act_loss"]))
            history["action_loss"].append(float(epoch_metrics["action_loss"]))
            history["sp_acc"].append(float(epoch_metrics["sp_acc"]))
            history["act_acc"].append(float(epoch_metrics["act_acc"]))

        log = (
            f"  loss={float(epoch_metrics['loss']):.4f}  "
            f"sp_acc={float(epoch_metrics['sp_acc']):.4f}  "
            f"act_acc={float(epoch_metrics['act_acc']):.4f}"
        )

        if val_samples:
            val_batches = make_batches(
                val_samples,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
            )
            val_metrics = evaluate(eval_step_fn, params, val_batches, n_val_batches)
            val_sp_map = val_metrics["species"]["mAP"]
            val_act_map = val_metrics["activity"]["mAP"]
            val_action_map = val_metrics["actions"]["mAP"]
            history["val_sp_map"].append(val_sp_map)
            history["val_act_map"].append(val_act_map)
            history["val_action_map"].append(val_action_map)
            log += (
                f"  val_sp_map={val_sp_map:.4f}  "
                f"val_act_map={val_act_map:.4f}  "
                f"val_action_map={val_action_map:.4f}"
            )

            composite = (val_sp_map + val_act_map + val_action_map) / 3.0
            if composite > best_val_metric and last_ckpt_path is not None:
                best_val_metric = composite
                best_ckpt_path = last_ckpt_path
                print(f"  ★ New best checkpoint (composite={best_val_metric:.4f}): {best_ckpt_path}")

        print(log)

        history_path = os.path.join(args.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    save_checkpoint(run_ckpt_dir, params, opt_state, args.num_epochs, global_step)
    print(f"\nTraining complete. Best composite val metric={best_val_metric:.4f}")
    if best_ckpt_path:
        print(f"Best checkpoint: {best_ckpt_path}")

    return params, opt_state, history, run_ckpt_dir


# ===================================================================
# Plotting
# ===================================================================

def plot_training_curves(history: dict, output_path: str):
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("MammalPS Benchmark 1 — Multi-task Training Curves", fontsize=13, fontweight="bold")

    axes[0].plot(epochs, history["loss"], marker="o", linewidth=2, label="Total loss")
    axes[0].plot(epochs, history["sp_loss"], marker="s", linewidth=1.5, linestyle="--", label="Species loss")
    axes[0].plot(epochs, history["act_loss"], marker="^", linewidth=1.5, linestyle="--", label="Activity loss")
    axes[0].plot(epochs, history["action_loss"], marker="d", linewidth=1.5, linestyle="--", label="Action loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Losses")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8)

    axes[1].plot(epochs, [v * 100 for v in history["sp_acc"]], marker="o", linewidth=2, label="Species acc")
    axes[1].plot(epochs, [v * 100 for v in history["act_acc"]], marker="s", linewidth=2, label="Activity acc")
    if history.get("val_sp_map"):
        axes[1].plot(epochs, [v * 100 for v in history["val_sp_map"]], linestyle="--", marker="o", linewidth=2, label="Val species mAP")
    if history.get("val_act_map"):
        axes[1].plot(epochs, [v * 100 for v in history["val_act_map"]], linestyle="--", marker="s", linewidth=2, label="Val activity mAP")
    axes[1].set(xlabel="Epoch", ylabel="(%)", title="Species & Activity mAP")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=8)

    if history.get("val_action_map"):
        axes[2].plot(epochs, [v * 100 for v in history["val_action_map"]], marker="^", linewidth=2, color="tab:purple", label="Val action mAP")
    axes[2].set(xlabel="Epoch", ylabel="(%)", title="Action mAP")
    axes[2].set_ylim(0, 105)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Training curves saved to {output_path}")
    plt.close()


def plot_per_class_accuracy(metrics: dict, output_path: str):
    """Three-panel bar chart: per-species AP, per-activity AP, per-action AP."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle("MammalPS Benchmark 1 — Per-class AP Breakdown", fontsize=13, fontweight="bold")

    sp_names = list(metrics["species"]["per_class"].keys())
    sp_aps = [metrics["species"]["per_class"][n]["ap"] * 100 for n in sp_names]
    colors_sp = plt.cm.Set2(np.linspace(0, 1, len(sp_names)))
    bars = axes[0].bar(sp_names, sp_aps, color=colors_sp, edgecolor="k", linewidth=0.6)
    overall_sp = metrics["species"]["mAP"] * 100
    axes[0].axhline(overall_sp, color="black", linestyle="--", linewidth=1.5, label=f"mAP = {overall_sp:.1f}%")
    for bar, v in zip(bars, sp_aps):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    axes[0].set(xlabel="Species", ylabel="AP (%)", title="Per-species Average Precision", ylim=(0, 110))
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    plt.sca(axes[0])
    plt.xticks(rotation=25, ha="right")

    act_names = list(metrics["activity"]["per_class"].keys())
    act_aps = [metrics["activity"]["per_class"][n]["ap"] * 100 for n in act_names]
    colors_act = plt.cm.tab10(np.linspace(0, 1, len(act_names)))
    bars = axes[1].bar(act_names, act_aps, color=colors_act, edgecolor="k", linewidth=0.6)
    overall_act = metrics["activity"]["mAP"] * 100
    axes[1].axhline(overall_act, color="black", linestyle="--", linewidth=1.5, label=f"mAP = {overall_act:.1f}%")
    for bar, v in zip(bars, act_aps):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    axes[1].set(xlabel="Activity", ylabel="AP (%)", title="Per-activity Average Precision", ylim=(0, 110))
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    plt.sca(axes[1])
    plt.xticks(rotation=35, ha="right")

    action_names = list(metrics["actions"]["per_class"].keys())
    action_aps = [metrics["actions"]["per_class"][n]["ap"] * 100 for n in action_names]
    colors_action = plt.cm.tab20(np.linspace(0, 1, len(action_names)))
    bars = axes[2].bar(action_names, action_aps, color=colors_action, edgecolor="k", linewidth=0.6)
    overall_map = metrics["actions"]["mAP"] * 100
    axes[2].axhline(overall_map, color="black", linestyle="--", linewidth=1.5, label=f"mAP = {overall_map:.1f}%")
    for bar, v in zip(bars, action_aps):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=6)
    axes[2].set(xlabel="Action", ylabel="AP (%)", title="Per-action Average Precision", ylim=(0, 110))
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    plt.sca(axes[2])
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Per-class breakdown plot saved to {output_path}")
    plt.close()


# ===================================================================
# Common setup
# ===================================================================

def print_env_info():
    print(f"JAX version : {jax.__version__}")
    print(f"JAX platform: {jax.extend.backend.get_backend().platform}")
    print(f"JAX devices : {jax.device_count()}")
    print(f"Species classes   ({NUM_SPECIES}): {SPECIES_CLASSES}")
    print(f"Activity classes  ({NUM_ACTIVITIES}): {ACTIVITY_CLASSES}")
    print(f"Action classes    ({NUM_ACTIONS}): {ACTION_CLASSES}")
    print()


def resolve_checkpoint(args) -> str:
    """Return the path to a single checkpoint .pkl file from the CLI args."""
    if args.eval_ckpt:
        if not os.path.isfile(args.eval_ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {args.eval_ckpt}")
        return args.eval_ckpt

    if args.ckpt_dir:
        ckpt_files = sorted(glob.glob(os.path.join(args.ckpt_dir, "**", "checkpoint_step*.pkl"), recursive=True))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints found under {args.ckpt_dir}")
        return ckpt_files[-1]

    raise ValueError("Provide --eval_ckpt or --ckpt_dir to locate a checkpoint")


# ===================================================================
# CLI: train subcommand
# ===================================================================

def run_train(args):
    """Training + validation phase."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    video_dir = os.path.join(args.data_dir, "clips")
    train_csv = args.train_csv or os.path.join(args.data_dir, "metadata", "train.csv")
    val_csv = args.val_csv or os.path.join(args.data_dir, "metadata", "val.csv")

    print_env_info()

    print("Loading train samples ...")
    train_samples = load_csv_samples(train_csv, video_dir)
    print(f"  {len(train_samples)} train clips")

    print("Loading val samples ...")
    val_samples = load_csv_samples(val_csv, video_dir)
    print(f"  {len(val_samples)} val clips")
    print()

    print(
        f"Building VideoPrism {args.model_size} multi-task classifier "
        f"({NUM_SPECIES} species, {NUM_ACTIVITIES} activities, {NUM_ACTIONS} actions) ..."
    )
    classifier, params = build_model_and_params(model_size=args.model_size)
    tx, opt_state = build_optimizer(params, learning_rate=args.learning_rate)
    print()

    print("Starting training ...")
    params, opt_state, history, run_ckpt_dir = train(
        classifier, params, tx, opt_state,
        train_samples, val_samples, args,
    )
    plot_training_curves(history, os.path.join(args.output_dir, "training_curves.png"))

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nCheckpoints saved in {run_ckpt_dir}")
    print(f"History & plots  in {args.output_dir}")


# ===================================================================
# CLI: test subcommand
# ===================================================================

def run_test(args):
    """Test-set evaluation phase."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    video_dir = os.path.join(args.data_dir, "clips")
    test_csv = args.test_csv or os.path.join(args.data_dir, "metadata", "test.csv")

    print_env_info()

    print("Loading test samples ...")
    test_samples = load_csv_samples(test_csv, video_dir)
    print(f"  {len(test_samples)} test clips")
    print()

    print(
        f"Building VideoPrism {args.model_size} multi-task classifier "
        f"({NUM_SPECIES} species, {NUM_ACTIVITIES} activities, {NUM_ACTIONS} actions) ..."
    )
    classifier, params = build_model_and_params(model_size=args.model_size)

    ckpt_path = resolve_checkpoint(args)
    print(f"Loading checkpoint: {ckpt_path}")
    state = load_checkpoint(ckpt_path)
    params = state["params"]
    print()

    print("=" * 60)
    print(f"Test set evaluation  (multi-sample={args.num_test_samples}, "
          f"min_duration={args.min_clip_duration}s)")
    print("=" * 60)

    eval_step_fn = make_eval_step(classifier)
    test_metrics = evaluate_multi_sample(
        eval_step_fn,
        params,
        test_samples,
        num_samples=args.num_test_samples,
        min_duration=args.min_clip_duration,
    )

    plot_per_class_accuracy(
        test_metrics, os.path.join(args.output_dir, "per_class_breakdown.png")
    )

    results_path = os.path.join(args.output_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


# ===================================================================
# Argument parsing
# ===================================================================

def _add_common_args(p: argparse.ArgumentParser):
    """Arguments shared by both train and test subcommands."""
    p.add_argument("--data_dir", type=str, default="../mammalps-dataset/benchmark_1",
                    help="Root of benchmark_1 dataset")
    p.add_argument("--model_size", type=str, default="base", choices=["base", "large"],
                    help="VideoPrism backbone size")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4,
                    help="Parallel video loading threads")
    p.add_argument("--output_dir", type=str, default="results/benchmark1",
                    help="Directory for plots and result JSON")
    p.add_argument("--seed", type=int, default=42)


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune VideoPrism on MammalPS benchmark_1 (multi-task)"
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ---- train ----
    p_train = sub.add_parser("train", help="Run training + validation")
    _add_common_args(p_train)
    p_train.add_argument("--train_csv", type=str, default=None,
                         help="Path to train CSV (default: <data_dir>/metadata/train.csv)")
    p_train.add_argument("--val_csv", type=str, default=None,
                         help="Path to val CSV (default: <data_dir>/metadata/val.csv)")
    p_train.add_argument("--num_epochs", type=int, default=30)
    p_train.add_argument("--learning_rate", type=float, default=1e-4)
    p_train.add_argument("--ckpt_dir", type=str, default="checkpoints/benchmark1_finetune")
    p_train.add_argument("--ckpt_every", type=int, default=50,
                         help="Save checkpoint every N steps")
    p_train.add_argument("--keep_recent", type=int, default=5,
                         help="Number of recent checkpoints to keep on disk")
    p_train.add_argument("--resume_ckpt_dir", type=str, default=None,
                         help="Resume training from this checkpoint directory")

    # ---- test ----
    p_test = sub.add_parser("test", help="Evaluate on the test set")
    _add_common_args(p_test)
    p_test.add_argument("--test_csv", type=str, default=None,
                        help="Path to test CSV (default: <data_dir>/metadata/test.csv)")
    p_test.add_argument("--eval_ckpt", type=str, default=None,
                        help="Path to a specific checkpoint .pkl file")
    p_test.add_argument("--ckpt_dir", type=str, default=None,
                        help="Checkpoint directory (uses latest checkpoint found)")
    p_test.add_argument("--num_test_samples", type=int, default=10,
                        help="Number of random frame samples per clip for multi-sample eval")
    p_test.add_argument("--min_clip_duration", type=float, default=0.5,
                        help="Skip clips shorter than this duration (seconds)")

    return p.parse_args()


def main():
    args = parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "test":
        run_test(args)


if __name__ == "__main__":
    main()
