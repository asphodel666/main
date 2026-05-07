#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import functional as TVF


DEFAULT_TOTAL_BUDGET_SEC = 6 * 60 * 60
DEFAULT_NUMRUNS = 3
DEFAULT_NSEEDS = 100
DEFAULT_K_LIST = (50, 100, 1000, 10000)

DEFAULT_NOISE_SIGMA = 0.05
DEFAULT_EPS_SWITCH = 0.10
DEFAULT_MAX_POOL = 5000

DEFAULT_DIVERSITY_THRESHOLD_BASELINE = 6.0
DEFAULT_DIVERSITY_THRESHOLD_ADAPTIVE = 0.0  # OFF by default for AdaptiveLite.

DEFAULT_PRIORITIZE_SUBSAMPLE = 64
DEFAULT_EMB_SUBSAMPLE = 32
DEFAULT_USE_DIVERSITY = 0
DEFAULT_RANDOM_SEED_PROB = 0.5


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    display_name: str
    dataset_cls: type
    mean: float
    std: float
    model_path: str
    raw_csv_name: str
    summary_csv_name: str


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "mnist": DatasetConfig(
        name="mnist",
        display_name="MNIST",
        dataset_cls=datasets.MNIST,
        mean=0.1307,
        std=0.3081,
        model_path="lenet5_mnist.pth",
        raw_csv_name="mnist_results_diploma.csv",
        summary_csv_name="mnist_summary_diploma.csv",
    ),
    "fashionmnist": DatasetConfig(
        name="fashionmnist",
        display_name="Fashion-MNIST",
        dataset_cls=datasets.FashionMNIST,
        mean=0.2860,
        std=0.3530,
        model_path="lenet5_fashionmnist.pth",
        raw_csv_name="fashionmnist_results_diploma.csv",
        summary_csv_name="fashionmnist_summary_diploma.csv",
    ),
}


RAW_FIELDNAMES = [
    "dataset",
    "method",
    "KSWITCH",
    "run",
    "seed",
    "total_budget_sec",
    "cell_budget_sec",
    "nseeds",
    "only_correct_seeds",
    "dataset_mean",
    "dataset_std",
    "train_epochs",
    "model_path",
    "noise_sigma",
    "eps_switch",
    "max_pool",
    "diversity_threshold_baseline",
    "diversity_threshold_adaptive",
    "use_diversity",
    "emb_subsample",
    "prioritize_subsample",
    "random_seed_prob",
    "seconds",
    "iterations",
    "iters_per_sec",
    "failures",
    "fail_misclassification",
    "fail_nanorinf",
    "failures_per_sec",
    "final_cov_nc",
    "final_cov_nbc",
    "final_cov_snac",
    "final_cov_mean",
    "pool_size",
]


SUMMARY_FIELDNAMES = [
    "dataset",
    "method",
    "KSWITCH",
    "runs",
    "iters_per_sec_mean",
    "iters_per_sec_std",
    "failures_mean",
    "failures_std",
    "failures_per_sec_mean",
    "failures_per_sec_std",
    "final_cov_nc_mean",
    "final_cov_nbc_mean",
    "final_cov_snac_mean",
    "final_cov_mean_mean",
    "pool_size_mean",
]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_min(mean: float, std: float) -> float:
    return (0.0 - float(mean)) / float(std)


def clamp_normalized(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    mn = (0.0 - float(mean)) / float(std)
    mx = (1.0 - float(mean)) / float(std)
    return torch.clamp(x, mn, mx)


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


class LeNet5(nn.Module):
    """Small LeNet-like CNN for 1x28x28 images and 10 classes."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def make_transform(cfg: DatasetConfig) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((cfg.mean,), (cfg.std,)),
    ])


def train_lenet_if_missing(
    device: torch.device,
    path: str,
    cfg: DatasetConfig,
    data_root: str,
    epochs: int,
    batch_size: int,
    download: bool,
) -> None:
    print(f"[WARN] Model weights not found: {path}")
    print(f"[WARN] Training LeNet5 on {cfg.display_name} for {epochs} epoch(s).")

    transform = make_transform(cfg)
    try:
        train_ds = cfg.dataset_cls(root=data_root, train=True, download=download, transform=transform)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load/download {cfg.display_name}. "
            f"Check internet connection, or put dataset files into '{data_root}', "
            f"or run with --download 0 if data already exists. Original error: {exc}"
        ) from exc

    use_cuda = device.type == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=bool(use_cuda),
    )

    model = LeNet5().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.train()
    for ep in range(int(epochs)):
        losses: list[float] = []
        correct = 0
        total = 0
        t0 = time.perf_counter()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu()))
            pred = torch.argmax(logits.detach(), dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
        dt = time.perf_counter() - t0
        acc = correct / max(total, 1)
        print(f"[TRAIN] {cfg.name} epoch={ep + 1}/{epochs} loss={np.mean(losses):.4f} train_acc={acc:.4f} time={dt:.1f}s")

    Path(path).parent.mkdir(parents=True, exist_ok=True) if str(Path(path).parent) not in (".", "") else None
    model.eval()
    torch.save(model.state_dict(), path)
    print(f"[INFO] Saved trained weights: {path}")


def load_model(
    device: torch.device,
    path: str,
    cfg: DatasetConfig,
    data_root: str,
    train_epochs: int,
    train_batch_size: int,
    download: bool,
) -> nn.Module:
    if not os.path.exists(path):
        train_lenet_if_missing(
            device=device,
            path=path,
            cfg=cfg,
            data_root=data_root,
            epochs=train_epochs,
            batch_size=train_batch_size,
            download=download,
        )
    model = LeNet5().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, test_ds, max_items: int = 2000) -> float:
    dev = model_device(model)
    model.eval()
    total = 0
    correct = 0
    limit = min(int(max_items), len(test_ds)) if max_items > 0 else len(test_ds)
    for i in range(limit):
        img, lbl = test_ds[i]
        logits = model(img.unsqueeze(0).to(dev))
        pred = int(torch.argmax(logits, dim=1).item())
        correct += int(pred == int(lbl))
        total += 1
    return float(correct / max(total, 1))


class ReLUActivationCollector:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.handles = []
        self.last_acts: list[torch.Tensor] = []

        def hook_fn(_module, _inp, out):
            if isinstance(out, torch.Tensor):
                self.last_acts.append(out)

        for m in self.model.modules():
            if isinstance(m, nn.ReLU):
                self.handles.append(m.register_forward_hook(hook_fn))

    def clear(self) -> None:
        self.last_acts = []

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


class Coverage3:
    @staticmethod
    def nc(acts: list[torch.Tensor], threshold: float = 0.0) -> float:
        if not acts:
            return 0.0
        total = 0
        active = 0
        for a in acts:
            flat = a.flatten()
            total += flat.numel()
            active += int((flat > threshold).sum().item())
        return active / total if total else 0.0

    @staticmethod
    def nbc(acts: list[torch.Tensor], threshold: float = 0.1) -> float:
        if not acts:
            return 0.0
        total = 0
        boundary = 0
        for a in acts:
            flat = a.flatten()
            total += flat.numel()
            boundary += int(((flat < -threshold) | (flat > threshold)).sum().item())
        return boundary / total if total else 0.0

    @staticmethod
    def snac(acts: list[torch.Tensor], strong_threshold: float = 0.75) -> float:
        if not acts:
            return 0.0
        total = 0
        strong = 0
        for a in acts:
            flat = a.flatten()
            mx = float(flat.max().item())
            if mx <= 0:
                continue
            total += flat.numel()
            strong += int((flat > strong_threshold * mx).sum().item())
        return strong / total if total else 0.0

    @staticmethod
    def vector(acts: list[torch.Tensor]) -> list[float]:
        v = [Coverage3.nc(acts), Coverage3.nbc(acts), Coverage3.snac(acts)]
        return [float(np.clip(x, 0.0, 1.0)) for x in v]


@torch.no_grad()
def failure_oracle(model: nn.Module, x: torch.Tensor, seed_label: int) -> tuple[bool, str | None]:
    dev = model_device(model)
    xb = x.unsqueeze(0).to(dev) if x.dim() == 3 else x.to(dev)
    logits = model(xb)
    if not torch.isfinite(logits).all().item():
        return True, "nanorinf"
    pred = int(torch.argmax(logits, dim=1).item())
    if pred != int(seed_label):
        return True, "misclassification"
    return False, None


def mutate_input(seed_img: torch.Tensor, noise_sigma: float, mean: float, std: float) -> torch.Tensor:
    mutant = seed_img.clone()
    mutant = mutant + torch.randn_like(mutant) * float(noise_sigma)
    if mutant.dim() == 3 and mutant.shape[-2:] == (28, 28):
        angle = float(np.random.uniform(-15, 15))
        mutant = TVF.rotate(mutant, angle=angle, fill=float(norm_min(mean, std)))
    mutant = mutant * float(np.random.uniform(0.8, 1.2))
    return clamp_normalized(mutant, mean, std)


def activation_embedding(acts: list[torch.Tensor]) -> torch.Tensor:
    if not acts:
        return torch.zeros(2, dtype=torch.float32)
    feats = []
    for a in acts:
        flat = a.flatten()
        feats.append(flat.mean())
        feats.append(flat.std(unbiased=False))
    return torch.stack(feats).detach().to("cpu", dtype=torch.float32)


def approx_diverse_enough(emb: torch.Tensor, pool: deque[torch.Tensor], threshold: float, subsample: int) -> bool:
    if threshold <= 0:
        return True
    n = len(pool)
    if n == 0:
        return True
    m = min(int(subsample), n)
    idxs = np.random.choice(n, size=m, replace=False)
    for i in idxs:
        if float(torch.dist(emb, pool[i]).item()) <= threshold:
            return False
    return True


class TensorFuzzStyleFuzzer:
    def __init__(self, model: nn.Module, diversity_threshold: float, max_pool: int, mean: float, std: float) -> None:
        self.model = model
        self.max_pool = int(max_pool)
        self.mean = float(mean)
        self.std = float(std)
        self.seedpool = deque(maxlen=self.max_pool)
        self.embed_pool = deque(maxlen=self.max_pool)
        self.diversity_threshold = float(diversity_threshold)

        self.fail_counts = defaultdict(int)
        self.total_failures = 0
        self.total_iterations = 0
        self.final_cov = [0.0, 0.0, 0.0]

    def _push_seed(self, img: torch.Tensor, lbl: int) -> None:
        self.seedpool.append((img.detach().to("cpu"), int(lbl)))

    @torch.no_grad()
    def run_time_budget(self, initial_seeds, time_budget_sec, collector, noise_sigma):
        self.seedpool.clear()
        self.embed_pool.clear()
        self.fail_counts.clear()
        self.total_failures = 0
        self.total_iterations = 0
        self.final_cov = [0.0, 0.0, 0.0]

        for img, lbl in initial_seeds:
            self._push_seed(img, lbl)

        dev = model_device(self.model)
        for img, _lbl in list(self.seedpool)[: min(len(self.seedpool), 64)]:
            collector.clear()
            _ = self.model(img.unsqueeze(0).to(dev))
            self.embed_pool.append(activation_embedding(collector.last_acts))

        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < time_budget_sec:
            seed_img, seed_lbl = self.seedpool[np.random.randint(0, len(self.seedpool))]
            mutant = mutate_input(seed_img, noise_sigma=noise_sigma, mean=self.mean, std=self.std)

            is_fail, failtype = failure_oracle(self.model, mutant, seed_lbl)
            if is_fail:
                self.total_failures += 1
                self.fail_counts[str(failtype)] += 1

            collector.clear()
            _ = self.model(mutant.unsqueeze(0).to(dev))
            acts = collector.last_acts

            emb = activation_embedding(acts)
            if approx_diverse_enough(emb, self.embed_pool, self.diversity_threshold, subsample=32):
                self._push_seed(mutant, seed_lbl)
                self.embed_pool.append(emb)

            cov = Coverage3.vector(acts)
            self.final_cov = [max(a, b) for a, b in zip(self.final_cov, cov)]
            self.total_iterations += 1

        seconds = time.perf_counter() - t0
        return {
            "seconds": float(seconds),
            "iterations": int(self.total_iterations),
            "iters_per_sec": float(self.total_iterations / max(seconds, 1e-9)),
            "failures": int(self.total_failures),
            "fail_misclassification": int(self.fail_counts.get("misclassification", 0)),
            "fail_nanorinf": int(self.fail_counts.get("nanorinf", 0)),
            "final_cov": self.final_cov,
        }


class DeepHunterStyleFuzzer:
    def __init__(self, model: nn.Module, max_pool: int, mean: float, std: float) -> None:
        self.model = model
        self.max_pool = int(max_pool)
        self.mean = float(mean)
        self.std = float(std)
        self.seedpool = deque(maxlen=self.max_pool)

        self.fail_counts = defaultdict(int)
        self.total_failures = 0
        self.total_iterations = 0
        self.final_cov = [0.0, 0.0, 0.0]

    def _push_seed(self, img: torch.Tensor, lbl: int) -> None:
        self.seedpool.append((img.detach().to("cpu"), int(lbl)))

    @staticmethod
    def weighted_score(cov: list[float], w: np.ndarray | None = None) -> float:
        c = np.array(cov, dtype=float)
        if w is None:
            return float(np.dot(c, np.ones_like(c)))
        return float(np.dot(c, w))

    @torch.no_grad()
    def run_time_budget(self, initial_seeds, time_budget_sec, collector, noise_sigma, weights=None):
        self.seedpool.clear()
        self.fail_counts.clear()
        self.total_failures = 0
        self.total_iterations = 0
        self.final_cov = [0.0, 0.0, 0.0]

        for img, lbl in initial_seeds:
            self._push_seed(img, lbl)

        w = None if weights is None else np.array(weights, dtype=float)
        current_cov = [0.0, 0.0, 0.0]
        current_score = self.weighted_score(current_cov, w=w)

        dev = model_device(self.model)
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < time_budget_sec:
            seed_img, seed_lbl = self.seedpool[np.random.randint(0, len(self.seedpool))]
            mutant = mutate_input(seed_img, noise_sigma=noise_sigma, mean=self.mean, std=self.std)

            is_fail, failtype = failure_oracle(self.model, mutant, seed_lbl)
            if is_fail:
                self.total_failures += 1
                self.fail_counts[str(failtype)] += 1

            collector.clear()
            _ = self.model(mutant.unsqueeze(0).to(dev))
            acts = collector.last_acts

            cand_cov = [max(a, b) for a, b in zip(current_cov, Coverage3.vector(acts))]
            cand_score = self.weighted_score(cand_cov, w=w)

            if cand_score > current_score + 1e-12:
                self._push_seed(mutant, seed_lbl)
                current_cov = cand_cov
                current_score = cand_score

            self.final_cov = [max(a, b) for a, b in zip(self.final_cov, current_cov)]
            self.total_iterations += 1

        seconds = time.perf_counter() - t0
        return {
            "seconds": float(seconds),
            "iterations": int(self.total_iterations),
            "iters_per_sec": float(self.total_iterations / max(seconds, 1e-9)),
            "failures": int(self.total_failures),
            "fail_misclassification": int(self.fail_counts.get("misclassification", 0)),
            "fail_nanorinf": int(self.fail_counts.get("nanorinf", 0)),
            "final_cov": self.final_cov,
        }


class AdaptivePool:
    def __init__(self, maxlen: int) -> None:
        self.maxlen = int(maxlen)
        self.imgs = deque(maxlen=self.maxlen)
        self.lbls = deque(maxlen=self.maxlen)
        self.covs = deque(maxlen=self.maxlen)

    def __len__(self) -> int:
        return len(self.imgs)

    def add(self, img: torch.Tensor, lbl: int, cov: list[float]) -> None:
        self.imgs.append(img.detach().to("cpu"))
        self.lbls.append(int(lbl))
        self.covs.append([float(x) for x in cov])

    def sample_random(self) -> tuple[torch.Tensor, int, list[float]]:
        n = len(self)
        i = int(np.random.randint(0, n))
        return self.imgs[i], int(self.lbls[i]), list(self.covs[i])

    def sample_worst_from_subset(self, active_metric: int, subsample: int) -> tuple[torch.Tensor, int, list[float]]:
        n = len(self)
        m = min(int(subsample), n)
        idxs = np.random.randint(0, n, size=m)
        worst_i = int(idxs[0])
        worst_v = self.covs[worst_i][active_metric]
        for i in idxs[1:]:
            v = self.covs[int(i)][active_metric]
            if v < worst_v:
                worst_v = v
                worst_i = int(i)
        return self.imgs[worst_i], int(self.lbls[worst_i]), list(self.covs[worst_i])


class AdaptiveLiteFastBalanced:
    def __init__(
        self,
        model: nn.Module,
        K: int,
        eps_switch: float,
        max_pool: int,
        use_diversity: bool,
        diversity_threshold: float,
        emb_subsample: int,
        prioritize_subsample: int,
        random_seed_prob: float,
        mean: float,
        std: float,
    ) -> None:
        self.model = model
        self.K = int(K)
        self.eps_switch = float(eps_switch)
        self.mean = float(mean)
        self.std = float(std)

        self.pool = AdaptivePool(maxlen=max_pool)
        self.embed_pool = deque(maxlen=int(max_pool))

        self.use_diversity = bool(use_diversity)
        self.diversity_threshold = float(diversity_threshold)
        self.emb_subsample = int(emb_subsample)

        self.prioritize_subsample = int(prioritize_subsample)
        self.random_seed_prob = float(random_seed_prob)

        self.active_metric = 0
        self.accepted_since_switch = 0
        self.window_start_cov = [0.0, 0.0, 0.0]
        self.current_best_cov = [0.0, 0.0, 0.0]

        self.fail_counts = defaultdict(int)
        self.total_failures = 0
        self.total_iterations = 0

    @staticmethod
    def choose_metric(start_cov: list[float], end_cov: list[float], eps_switch: float) -> int:
        if np.random.rand() < eps_switch:
            return int(np.argmin(end_cov))
        velocities = [end_cov[i] - start_cov[i] for i in range(len(end_cov))]
        return int(np.argmax(velocities))

    def _accept_seed(self, img: torch.Tensor, lbl: int, cov: list[float], emb: torch.Tensor) -> bool:
        if self.use_diversity:
            if not approx_diverse_enough(emb, self.embed_pool, self.diversity_threshold, subsample=self.emb_subsample):
                return False
        self.pool.add(img, lbl, cov=cov)
        self.embed_pool.append(emb)
        self.accepted_since_switch += 1
        self.current_best_cov = [max(a, b) for a, b in zip(self.current_best_cov, cov)]
        return True

    @torch.no_grad()
    def run_time_budget(self, initial_seeds, time_budget_sec, collector, noise_sigma):
        self.pool = AdaptivePool(maxlen=self.pool.maxlen)
        self.embed_pool.clear()

        self.fail_counts.clear()
        self.total_failures = 0
        self.total_iterations = 0

        self.active_metric = 0
        self.accepted_since_switch = 0
        self.window_start_cov = [0.0, 0.0, 0.0]
        self.current_best_cov = [0.0, 0.0, 0.0]

        dev = model_device(self.model)

        for img, lbl in initial_seeds:
            collector.clear()
            _ = self.model(img.unsqueeze(0).to(dev))
            acts = collector.last_acts
            cov = Coverage3.vector(acts)
            emb = activation_embedding(acts)
            self.pool.add(img, int(lbl), cov=cov)
            self.embed_pool.append(emb)
            self.current_best_cov = [max(a, b) for a, b in zip(self.current_best_cov, cov)]

        self.window_start_cov = self.current_best_cov.copy()

        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < time_budget_sec:
            if np.random.rand() < self.random_seed_prob:
                seed_img, seed_lbl, _seed_cov = self.pool.sample_random()
            else:
                seed_img, seed_lbl, _seed_cov = self.pool.sample_worst_from_subset(self.active_metric, self.prioritize_subsample)

            mutant = mutate_input(seed_img, noise_sigma=noise_sigma, mean=self.mean, std=self.std)
            is_fail, failtype = failure_oracle(self.model, mutant, seed_lbl)
            if is_fail:
                self.total_failures += 1
                self.fail_counts[str(failtype)] += 1

            collector.clear()
            _ = self.model(mutant.unsqueeze(0).to(dev))
            acts = collector.last_acts
            cov = Coverage3.vector(acts)

            improved_any = any(cov[i] > self.current_best_cov[i] + 1e-12 for i in range(3))
            if improved_any:
                emb = activation_embedding(acts)
                self._accept_seed(mutant, seed_lbl, cov=cov, emb=emb)

            if self.K > 0 and self.accepted_since_switch >= self.K:
                self.active_metric = self.choose_metric(self.window_start_cov, self.current_best_cov, self.eps_switch)
                self.window_start_cov = self.current_best_cov.copy()
                self.accepted_since_switch = 0

            self.total_iterations += 1

        seconds = time.perf_counter() - t0
        return {
            "seconds": float(seconds),
            "iterations": int(self.total_iterations),
            "iters_per_sec": float(self.total_iterations / max(seconds, 1e-9)),
            "failures": int(self.total_failures),
            "fail_misclassification": int(self.fail_counts.get("misclassification", 0)),
            "fail_nanorinf": int(self.fail_counts.get("nanorinf", 0)),
            "final_cov": self.current_best_cov,
            "pool_size": int(len(self.pool)),
        }


def fmt_cov3(cov: list[float]) -> str:
    return "[" + ", ".join(f"{c:.3f}" for c in cov) + "]"


def parse_k_list(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


@torch.no_grad()
def build_initial_seeds(test_ds, model: nn.Module, nseeds: int, only_correct: bool) -> list[tuple[torch.Tensor, int]]:
    dev = model_device(model)
    seeds: list[tuple[torch.Tensor, int]] = []
    scanned = 0
    skipped_wrong = 0
    for i in range(len(test_ds)):
        img, lbl = test_ds[i]
        scanned += 1
        if only_correct:
            logits = model(img.unsqueeze(0).to(dev))
            pred = int(torch.argmax(logits, dim=1).item())
            if pred != int(lbl):
                skipped_wrong += 1
                continue
        seeds.append((img.detach().to("cpu"), int(lbl)))
        if len(seeds) >= int(nseeds):
            break
    if len(seeds) < int(nseeds):
        raise RuntimeError(
            f"Could not collect {nseeds} initial seeds. Collected {len(seeds)}. "
            f"Try --only_correct_seeds 0 or train more epochs."
        )
    print(
        f"[INFO] Initial seeds: {len(seeds)} "
        f"(scanned={scanned}, skipped_wrong={skipped_wrong}, only_correct={int(only_correct)})"
    )
    return seeds


def make_common_row(args, cfg: DatasetConfig, method: str, k: int, run_idx: int, run_seed: int, cell_budget: float) -> dict:
    return {
        "dataset": cfg.name,
        "method": method,
        "KSWITCH": int(k),
        "run": int(run_idx + 1),
        "seed": int(run_seed),
        "total_budget_sec": float(args.total_budget_sec),
        "cell_budget_sec": float(cell_budget),
        "nseeds": int(args.nseeds),
        "only_correct_seeds": int(args.only_correct_seeds),
        "dataset_mean": float(cfg.mean),
        "dataset_std": float(cfg.std),
        "train_epochs": int(args.train_epochs),
        "model_path": str(args.model_path or cfg.model_path),
        "noise_sigma": float(args.noise_sigma),
        "eps_switch": float(args.eps_switch),
        "max_pool": int(args.max_pool),
        "diversity_threshold_baseline": float(args.diversity_threshold_baseline),
        "diversity_threshold_adaptive": float(args.diversity_threshold_adaptive),
        "use_diversity": int(args.use_diversity),
        "emb_subsample": int(args.emb_subsample),
        "prioritize_subsample": int(args.prioritize_subsample),
        "random_seed_prob": float(args.random_seed_prob),
    }


def append_result_metrics(row: dict, result: dict, cov: list[float], pool_size: int) -> dict:
    cov_mean = float(np.mean(cov))
    failps = float(result["failures"] / max(result["seconds"], 1e-9))
    row.update({
        "seconds": float(result["seconds"]),
        "iterations": int(result["iterations"]),
        "iters_per_sec": float(result["iters_per_sec"]),
        "failures": int(result["failures"]),
        "fail_misclassification": int(result["fail_misclassification"]),
        "fail_nanorinf": int(result["fail_nanorinf"]),
        "failures_per_sec": float(failps),
        "final_cov_nc": float(cov[0]),
        "final_cov_nbc": float(cov[1]),
        "final_cov_snac": float(cov[2]),
        "final_cov_mean": float(cov_mean),
        "pool_size": int(pool_size),
    })
    return row


def write_csv(path: str | Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0


def build_summary_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(str(r["dataset"]), str(r["method"]), int(r["KSWITCH"]))].append(r)

    out: list[dict] = []
    for (dataset_name, method, k), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        ips = [float(r["iters_per_sec"]) for r in items]
        fails = [float(r["failures"]) for r in items]
        failps = [float(r["failures_per_sec"]) for r in items]
        cov_nc = [float(r["final_cov_nc"]) for r in items]
        cov_nbc = [float(r["final_cov_nbc"]) for r in items]
        cov_snac = [float(r["final_cov_snac"]) for r in items]
        cov_mean = [float(r["final_cov_mean"]) for r in items]
        pool_size = [float(r["pool_size"]) for r in items]
        out.append({
            "dataset": dataset_name,
            "method": method,
            "KSWITCH": int(k),
            "runs": int(len(items)),
            "iters_per_sec_mean": mean(ips),
            "iters_per_sec_std": stdev(ips),
            "failures_mean": mean(fails),
            "failures_std": stdev(fails),
            "failures_per_sec_mean": mean(failps),
            "failures_per_sec_std": stdev(failps),
            "final_cov_nc_mean": mean(cov_nc),
            "final_cov_nbc_mean": mean(cov_nbc),
            "final_cov_snac_mean": mean(cov_snac),
            "final_cov_mean_mean": mean(cov_mean),
            "pool_size_mean": mean(pool_size),
        })
    return out


def print_summary(summary_rows: list[dict], dataset_name: str) -> None:
    print("=" * 88)
    print(f"[SUMMARY] {dataset_name}")
    print("method                      K      speed_mean     fail/s_mean    cov_mean")
    print("-" * 88)
    for r in summary_rows:
        if r["dataset"] != dataset_name:
            continue
        print(
            f"{str(r['method'])[:26]:26s} "
            f"{int(r['KSWITCH']):6d} "
            f"{float(r['iters_per_sec_mean']):13.2f} "
            f"{float(r['failures_per_sec_mean']):14.3f} "
            f"{float(r['final_cov_mean_mean']):10.3f}"
        )


def dataset_sequence(dataset_arg: str) -> list[DatasetConfig]:
    if dataset_arg == "both":
        return [DATASET_CONFIGS["mnist"], DATASET_CONFIGS["fashionmnist"]]
    return [DATASET_CONFIGS[dataset_arg]]


def run_one_dataset(args, cfg: DatasetConfig, device: torch.device) -> tuple[list[dict], list[dict]]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = str(args.model_path) if args.model_path else cfg.model_path
    if args.dataset == "both" and args.model_path:
        raise ValueError("Do not use --model_path with --dataset both. Use default separate model paths.")

    raw_csv = Path(args.out_csv) if args.out_csv and args.dataset != "both" else out_dir / cfg.raw_csv_name
    summary_csv = Path(args.summary_csv) if args.summary_csv and args.dataset != "both" else out_dir / cfg.summary_csv_name

    k_list = parse_k_list(args.k_values)
    baselines = ["tensorfuzz", "deephunter"]
    adaptive_methods = ["adaptivelite_fastbalanced"]
    grid_cells = len(baselines) * int(args.num_runs) + len(adaptive_methods) * int(args.num_runs) * len(k_list)
    cell_budget = float(args.total_budget_sec) / max(1, grid_cells)

    print("=" * 88)
    print(f"[DATASET] {cfg.display_name} ({cfg.name})")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] total_budget_sec={args.total_budget_sec:.1f} for this dataset")
    print(f"[INFO] runs={args.num_runs} nseeds={args.nseeds} only_correct_seeds={args.only_correct_seeds}")
    print(f"[INFO] K values (Adaptive only): {k_list}")
    print(f"[INFO] grid_cells={grid_cells} => cell_budget_sec={cell_budget:.1f} (~{cell_budget / 60:.1f} min)")
    print(f"[INFO] data_root={args.data_root} download={args.download}")
    print(f"[INFO] model_path={model_path}")

    model = load_model(
        device=device,
        path=model_path,
        cfg=cfg,
        data_root=args.data_root,
        train_epochs=args.train_epochs,
        train_batch_size=args.train_batch_size,
        download=bool(args.download),
    )

    transform = make_transform(cfg)
    try:
        test_ds = cfg.dataset_cls(root=args.data_root, train=False, download=bool(args.download), transform=transform)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load/download test split for {cfg.display_name}. "
            f"Check internet connection or data_root='{args.data_root}'. Original error: {exc}"
        ) from exc

    if int(args.eval_items) > 0:
        acc = evaluate_accuracy(model, test_ds, max_items=int(args.eval_items))
        print(f"[INFO] Clean accuracy on first {min(int(args.eval_items), len(test_ds))} test items: {acc:.4f}")

    initial_seeds = build_initial_seeds(
        test_ds=test_ds,
        model=model,
        nseeds=int(args.nseeds),
        only_correct=bool(args.only_correct_seeds),
    )

    collector = ReLUActivationCollector(model)
    rows: list[dict] = []

    try:
        print("=" * 88)
        print("[BASELINES] TensorFuzz + DeepHunter (no K loop)")
        for run_idx in range(int(args.num_runs)):
            run_seed = 1000 + run_idx
            set_all_seeds(run_seed)

            f1 = TensorFuzzStyleFuzzer(
                model=model,
                diversity_threshold=args.diversity_threshold_baseline,
                max_pool=args.max_pool,
                mean=cfg.mean,
                std=cfg.std,
            )
            r1 = f1.run_time_budget(initial_seeds, cell_budget, collector, args.noise_sigma)
            cov1 = r1["final_cov"]
            row1 = make_common_row(args, cfg, "tensorfuzz", 0, run_idx, run_seed, cell_budget)
            rows.append(append_result_metrics(row1, r1, cov1, pool_size=len(f1.seedpool)))
            print(
                f"tensorfuzz   dataset={cfg.name} run={run_idx + 1}/{args.num_runs} "
                f"cell={cell_budget / 60:.1f}min time={r1['seconds']:.1f}s "
                f"it={r1['iterations']} speed={r1['iters_per_sec']:.1f} it/s "
                f"fail={r1['failures']} (mis={r1['fail_misclassification']}, nan={r1['fail_nanorinf']}) "
                f"fail/s={rows[-1]['failures_per_sec']:.3f} cov={fmt_cov3(cov1)} mean={rows[-1]['final_cov_mean']:.3f}"
            )

            f2 = DeepHunterStyleFuzzer(model=model, max_pool=args.max_pool, mean=cfg.mean, std=cfg.std)
            r2 = f2.run_time_budget(initial_seeds, cell_budget, collector, args.noise_sigma)
            cov2 = r2["final_cov"]
            row2 = make_common_row(args, cfg, "deephunter", 0, run_idx, run_seed, cell_budget)
            rows.append(append_result_metrics(row2, r2, cov2, pool_size=len(f2.seedpool)))
            print(
                f"deephunter   dataset={cfg.name} run={run_idx + 1}/{args.num_runs} "
                f"cell={cell_budget / 60:.1f}min time={r2['seconds']:.1f}s "
                f"it={r2['iterations']} speed={r2['iters_per_sec']:.1f} it/s "
                f"fail={r2['failures']} (mis={r2['fail_misclassification']}, nan={r2['fail_nanorinf']}) "
                f"fail/s={rows[-1]['failures_per_sec']:.3f} cov={fmt_cov3(cov2)} mean={rows[-1]['final_cov_mean']:.3f}"
            )

        for k in k_list:
            print("=" * 88)
            print(f"[ADAPTIVE-FAST-BAL] dataset={cfg.name} KSWITCH={k}")
            for run_idx in range(int(args.num_runs)):
                run_seed = 2000 + run_idx
                set_all_seeds(run_seed)

                fl = AdaptiveLiteFastBalanced(
                    model=model,
                    K=k,
                    eps_switch=args.eps_switch,
                    max_pool=args.max_pool,
                    use_diversity=bool(args.use_diversity),
                    diversity_threshold=args.diversity_threshold_adaptive,
                    emb_subsample=args.emb_subsample,
                    prioritize_subsample=args.prioritize_subsample,
                    random_seed_prob=args.random_seed_prob,
                    mean=cfg.mean,
                    std=cfg.std,
                )

                rl = fl.run_time_budget(initial_seeds, cell_budget, collector, args.noise_sigma)
                covl = rl["final_cov"]
                rowl = make_common_row(args, cfg, "adaptivelite_fastbalanced", int(k), run_idx, run_seed, cell_budget)
                rows.append(append_result_metrics(rowl, rl, covl, pool_size=int(rl["pool_size"])))
                print(
                    f"adaptivelite dataset={cfg.name} run={run_idx + 1}/{args.num_runs} "
                    f"cell={cell_budget / 60:.1f}min time={rl['seconds']:.1f}s "
                    f"it={rl['iterations']} speed={rl['iters_per_sec']:.1f} it/s "
                    f"fail={rl['failures']} (mis={rl['fail_misclassification']}, nan={rl['fail_nanorinf']}) "
                    f"fail/s={rows[-1]['failures_per_sec']:.3f} pool={rl['pool_size']} "
                    f"cov={fmt_cov3(covl)} mean={rows[-1]['final_cov_mean']:.3f}"
                )

    finally:
        collector.remove()

    summary_rows = build_summary_rows(rows)
    write_csv(raw_csv, rows, RAW_FIELDNAMES)
    write_csv(summary_csv, summary_rows, SUMMARY_FIELDNAMES)
    print_summary(summary_rows, cfg.name)
    print("=" * 88)
    print(f"[DONE] Raw results saved to: {raw_csv}")
    print(f"[DONE] Summary saved to: {summary_csv}")
    return rows, summary_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashionmnist", "both"])
    ap.add_argument("--data_root", type=str, default=".data")
    ap.add_argument("--download", type=int, default=1)

    ap.add_argument("--total_budget_sec", type=float, default=DEFAULT_TOTAL_BUDGET_SEC)
    ap.add_argument("--num_runs", type=int, default=DEFAULT_NUMRUNS)
    ap.add_argument("--nseeds", type=int, default=DEFAULT_NSEEDS)
    ap.add_argument("--only_correct_seeds", type=int, default=1)
    ap.add_argument("--k_values", type=str, default=",".join(map(str, DEFAULT_K_LIST)))

    ap.add_argument("--noise_sigma", type=float, default=DEFAULT_NOISE_SIGMA)
    ap.add_argument("--eps_switch", type=float, default=DEFAULT_EPS_SWITCH)
    ap.add_argument("--max_pool", type=int, default=DEFAULT_MAX_POOL)

    ap.add_argument("--diversity_threshold_baseline", type=float, default=DEFAULT_DIVERSITY_THRESHOLD_BASELINE)
    ap.add_argument("--diversity_threshold_adaptive", type=float, default=DEFAULT_DIVERSITY_THRESHOLD_ADAPTIVE)
    ap.add_argument("--use_diversity", type=int, default=DEFAULT_USE_DIVERSITY)
    ap.add_argument("--emb_subsample", type=int, default=DEFAULT_EMB_SUBSAMPLE)

    ap.add_argument("--prioritize_subsample", type=int, default=DEFAULT_PRIORITIZE_SUBSAMPLE)
    ap.add_argument("--random_seed_prob", type=float, default=DEFAULT_RANDOM_SEED_PROB)

    ap.add_argument("--train_epochs", type=int, default=3)
    ap.add_argument("--train_batch_size", type=int, default=128)
    ap.add_argument("--eval_items", type=int, default=2000)

    ap.add_argument("--model_path", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--summary_csv", type=str, default=None)
    ap.add_argument("--combined_csv", type=str, default=None)
    ap.add_argument("--combined_summary_csv", type=str, default=None)

    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    all_rows: list[dict] = []
    all_summary_rows: list[dict] = []

    for cfg in dataset_sequence(args.dataset):
        rows, summary_rows = run_one_dataset(args, cfg, device)
        all_rows.extend(rows)
        all_summary_rows.extend(summary_rows)

    if len(dataset_sequence(args.dataset)) > 1:
        out_dir = Path(args.out_dir)
        combined_csv = Path(args.combined_csv) if args.combined_csv else out_dir / "all_datasets_results_diploma.csv"
        combined_summary_csv = Path(args.combined_summary_csv) if args.combined_summary_csv else out_dir / "all_datasets_summary_diploma.csv"
        write_csv(combined_csv, all_rows, RAW_FIELDNAMES)
        write_csv(combined_summary_csv, all_summary_rows, SUMMARY_FIELDNAMES)
        print("=" * 88)
        print(f"[DONE] Combined raw results saved to: {combined_csv}")
        print(f"[DONE] Combined summary saved to: {combined_summary_csv}")


if __name__ == "__main__":
    main()
