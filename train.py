"""Praktikum CNN Cat/Dog: audit, split, training, evaluasi, dan laporan.

Jalankan: python train.py --data-dir data/PetImages --output-dir outputs/baseline
Seluruh shape tensor menggunakan urutan PyTorch N,C,H,W.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
import warnings
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "dl006-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

DATASET_URL = "https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset"
CLASSES = ("Cat", "Dog")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def set_seed(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def audit_dataset(root: Path) -> tuple[list[dict], dict]:
    """Verifikasi struktur, decode penuh, ukuran, dan duplikat piksel persis."""
    valid = []
    rejected = []
    ignored = []
    image_warnings = []
    discovered = Counter()
    original_modes = Counter()
    sizes = []
    for label, class_name in enumerate(CLASSES):
        folder = root / class_name
        if not folder.is_dir():
            raise ValueError(f"Folder wajib tidak ada: {folder}. Gunakan --data-dir .../PetImages")
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                ignored.append(relative)
                continue
            discovered[class_name] += 1
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with Image.open(path) as image:
                        image.verify()
                    # verify() saja tidak cukup untuk mendeteksi semua JPEG terpotong.
                    with Image.open(path) as image:
                        mode = image.mode
                        rgb = ImageOps.exif_transpose(image).convert("RGB")
                        rgb.load()
                        width, height = rgb.size
                        digest = hashlib.sha256()
                        digest.update(f"{width}x{height}:RGB:".encode())
                        digest.update(rgb.tobytes())
                    for warning in caught:
                        image_warnings.append({"path": relative, "warning": str(warning.message)})
                original_modes[mode] += 1
                sizes.append((width, height))
                valid.append({"path": relative, "label": label, "width": width,
                              "height": height, "pixel_sha256": digest.hexdigest()})
            except (OSError, ValueError, SyntaxError, UnidentifiedImageError,
                    Image.DecompressionBombError) as error:
                rejected.append({"path": relative, "label": label,
                                 "reason": f"{type(error).__name__}: {error}"})
            if sum(discovered.values()) % 2500 == 0:
                print(f"Audit: {sum(discovered.values())} gambar diperiksa", flush=True)
    # Hilangkan duplikat persis sebelum split agar tidak masuk ke split berbeda.
    groups = defaultdict(list)
    for row in valid:
        groups[row["pixel_sha256"]].append(row)
    unique = []
    duplicates = []
    conflicts = []
    for group in groups.values():
        if len({row["label"] for row in group}) > 1:
            conflicts.extend(group)  # Label berbeda pada piksel sama: keluarkan seluruh grup.
        else:
            unique.append(group[0])
            duplicates.extend({"path": row["path"], "kept": group[0]["path"]}
                              for row in group[1:])
    counts = {name: sum(row["label"] == label for row in unique)
              for label, name in enumerate(CLASSES)}
    if any(count < 10 for count in counts.values()):
        raise ValueError(f"Minimal 10 gambar unik valid per kelas diperlukan; ditemukan {counts}")
    width_values = [size[0] for size in sizes]
    height_values = [size[1] for size in sizes]
    summary = {
        "root": str(root.resolve()), "discovered_per_class": dict(discovered),
        "decoded_count": len(valid), "usable_per_class": counts,
        "usable_count": len(unique), "corrupt_count": len(rejected),
        "duplicate_count": len(duplicates), "conflicting_label_count": len(conflicts),
        "ignored_count": len(ignored), "original_modes": dict(original_modes),
        "original_width": {"min": min(width_values), "max": max(width_values),
                           "median": float(np.median(width_values))},
        "original_height": {"min": min(height_values), "max": max(height_values),
                            "median": float(np.median(height_values))},
        "unique_sizes": len(set(sizes)), "warning_count": len(image_warnings),
        "duplicate_policy": "deduplikasi piksel RGB setelah koreksi EXIF; near-duplicate belum terdeteksi",
    }
    return unique, {"summary": summary, "rejected": rejected, "duplicates": duplicates,
                    "conflicting_labels": conflicts, "ignored": ignored, "warnings": image_warnings}


def split_dataset(rows: list[dict], seed: int, val_ratio: float,
                  test_ratio: float) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for label in range(len(CLASSES)):
        group = sorted((row for row in rows if row["label"] == label), key=lambda row: row["path"])
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_ratio))
        n_test = max(1, int(len(group) * test_ratio))
        if n_val + n_test >= len(group):
            raise ValueError("Rasio split tidak menyisakan gambar training.")
        splits["validation"].extend(group[:n_val])
        splits["test"].extend(group[n_val:n_val + n_test])
        splits["train"].extend(group[n_val + n_test:])
    hashes = [{row["pixel_sha256"] for row in split} for split in splits.values()]
    paths = [{row["path"] for row in split} for split in splits.values()]
    for left in range(3):
        for right in range(left + 1, 3):
            if hashes[left] & hashes[right] or paths[left] & paths[right]:
                raise RuntimeError("Kebocoran data: path/piksel identik ditemukan di beberapa split.")
    if sum(map(len, splits.values())) != len(rows):
        raise RuntimeError("Split tidak mencakup seluruh gambar unik valid.")
    return splits


def preprocess(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return np.array(rgb.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


class CatsDogsDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict], size: int, cache: bool):
        self.root, self.rows, self.size = root, rows, size
        self.cache = None
        if cache:
            self.cache = np.stack([preprocess(root / row["path"], size) for row in rows])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        pixels = self.cache[index] if self.cache is not None else preprocess(self.root / row["path"], self.size)
        # H,W,C uint8 -> C,H,W float32; label float32 untuk BCE satu logit.
        image = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float().div_(255.0)
        return image, torch.tensor(float(row["label"]), dtype=torch.float32)


class SmallCNN(nn.Module):
    def __init__(self, image_size: int):
        super().__init__()
        spatial = image_size // 2 // 2
        self.layers = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)),
            ("relu1", nn.ReLU()),
            ("pool1", nn.MaxPool2d(2, stride=2)),
            ("conv2", nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
            ("relu2", nn.ReLU()),
            ("pool2", nn.MaxPool2d(2, stride=2)),
            ("flatten", nn.Flatten()),
            ("dense", nn.Linear(32 * spatial * spatial, 64)),
            ("relu3", nn.ReLU()),
            ("dropout", nn.Dropout(0.25)),
            ("output", nn.Linear(64, 1)),
        ]))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images).squeeze(1)


def manual_architecture(size: int) -> list[dict]:
    """Perhitungan independen sebelum dummy forward, bukan membaca shape hasil model."""
    rows = []
    shape = (3, size, size)
    specs = [("conv1", "conv", 16), ("relu1", "relu", None), ("pool1", "pool", None),
             ("conv2", "conv", 32), ("relu2", "relu", None), ("pool2", "pool", None),
             ("flatten", "flatten", None), ("dense", "linear", 64),
             ("relu3", "relu", None), ("dropout", "dropout", None), ("output", "linear", 1)]
    for name, kind, out_features in specs:
        incoming = shape
        parameters = 0
        if kind == "conv":
            channels, height, width = shape
            kernel, padding, stride = 3, 1, 1
            height = math.floor((height + 2 * padding - kernel) / stride) + 1
            width = math.floor((width + 2 * padding - kernel) / stride) + 1
            parameters = (channels * kernel * kernel + 1) * out_features
            shape = (out_features, height, width)
        elif kind == "pool":
            channels, height, width = shape
            shape = (channels, math.floor((height - 2) / 2) + 1,
                     math.floor((width - 2) / 2) + 1)
        elif kind == "flatten":
            shape = (math.prod(shape),)
        elif kind == "linear":
            parameters = (shape[0] + 1) * out_features
            shape = (out_features,)
        rows.append({"layer": name, "input_shape": list(incoming),
                     "manual_output_shape": list(shape), "manual_parameters": parameters})
    return rows


def verify_architecture(model: SmallCNN, size: int) -> list[dict]:
    rows = manual_architecture(size)
    model.eval()
    tensor = torch.zeros(1, 3, size, size)
    with torch.inference_mode():
        for row, (name, layer) in zip(rows, model.layers.named_children()):
            tensor = layer(tensor)
            shape = list(tensor.shape[1:])
            params = sum(parameter.numel() for parameter in layer.parameters())
            if row["layer"] != name or row["manual_output_shape"] != shape or row["manual_parameters"] != params:
                raise RuntimeError(f"Perhitungan manual berbeda dari PyTorch pada {name}")
            row.update({"framework_output_shape": shape, "framework_parameters": params, "verified": True})
    total = sum(row["manual_parameters"] for row in rows)
    if total != sum(parameter.numel() for parameter in model.parameters()):
        raise RuntimeError("Total parameter manual berbeda dari model.")
    return rows


def architecture_table(rows: list[dict]) -> str:
    text = "| Layer | Input C,H,W / fitur | Output manual | Output PyTorch | Parameter manual / PyTorch |\n"
    text += "|---|---|---|---|---:|\n"
    for row in rows:
        text += (f"| {row['layer']} | {row['input_shape']} | {row['manual_output_shape']} | "
                 f"{row['framework_output_shape']} | {row['manual_parameters']:,} / {row['framework_parameters']:,} |\n")
    return text


def check_batch(images: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor, size: int) -> None:
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, size, size):
        raise ValueError(f"Shape input harus (N,3,{size},{size}), ditemukan {tuple(images.shape)}")
    if logits.shape != labels.shape or labels.ndim != 1:
        raise ValueError(f"Shape logit {tuple(logits.shape)} dan label {tuple(labels.shape)} harus sama: (N,)")
    if labels.dtype != torch.float32 or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Label BCE harus float32 dengan nilai Cat=0 atau Dog=1.")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Pixel input harus finite dan berada di [0,1].")
    if not torch.isfinite(logits).all():
        raise ValueError("Logit mengandung NaN/Inf; periksa learning rate dan pipeline.")


def metrics_from_predictions(labels: list[int], predictions: list[int], loss: float) -> dict:
    matrix = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(labels, predictions):
        matrix[actual, predicted] += 1
    per_class = {}
    for index, name in enumerate(CLASSES):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum()) - tp
        fn = int(matrix[index, :].sum()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[name] = {"precision": precision, "recall": recall,
                           "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                           "support": int(matrix[index].sum())}
    return {"loss": loss, "accuracy": float(np.trace(matrix) / matrix.sum()),
            "balanced_accuracy": sum(value["recall"] for value in per_class.values()) / 2,
            "macro_f1": sum(value["f1"] for value in per_class.values()) / 2,
            "per_class": per_class, "confusion_matrix": matrix.tolist(), "samples": len(labels)}


def run_epoch(model: SmallCNN, loader: DataLoader, loss_fn: nn.Module,
              device: torch.device, size: int, optimizer=None, collect=False) -> tuple[dict, list[float]]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    labels_all, predictions_all, probabilities = [], [], []
    with torch.set_grad_enabled(training):
        for batch_index, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if batch_index == 0:
                check_batch(images, labels, logits, size)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Loss NaN/Inf; training dihentikan.")
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.item()) * labels.numel()
            probs = torch.sigmoid(logits.detach())
            labels_all.extend(labels.detach().cpu().to(torch.int64).tolist())
            predictions_all.extend((probs >= 0.5).cpu().to(torch.int64).tolist())
            if collect:
                probabilities.extend(probs.cpu().tolist())
    return metrics_from_predictions(labels_all, predictions_all, loss_sum / len(labels_all)), probabilities


def plot_history(history: list[dict], destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row["epoch"] for row in history]
    for axis, metric in zip(axes, ("loss", "accuracy")):
        axis.plot(epochs, [row[f"train_{metric}"] for row in history], label="Train", marker="o")
        axis.plot(epochs, [row[f"val_{metric}"] for row in history], label="Validation", marker="o")
        axis.set(xlabel="Epoch", ylabel=metric.title(), title=f"Train / Validation {metric}")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def plot_confusion(metrics: dict, destination: Path) -> None:
    matrix = np.array(metrics["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(5, 4))
    heatmap = axis.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(value), ha="center", va="center",
                  color="white" if value > matrix.max() / 2 else "black")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASSES, yticklabels=CLASSES,
             xlabel="Prediksi", ylabel="Label sebenarnya", title="Confusion matrix — test")
    fig.colorbar(heatmap, ax=axis)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def export_predictions(rows: list[dict], probabilities: list[float], root: Path,
                       output: Path, interpretations: dict | None = None) -> list[dict]:
    predictions = [{"path": row["path"], "actual": CLASSES[row["label"]],
                    "predicted": CLASSES[int(probability >= 0.5)],
                    "probability_dog": probability}
                   for row, probability in zip(rows, probabilities)]
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    errors = [row for row in predictions if row["actual"] != row["predicted"]]
    # Tiga contoh deterministik, tanpa memilih berdasarkan confidence.
    examples = errors[:3]
    for example in examples:
        if interpretations and example["path"] in interpretations:
            example["interpretation"] = interpretations[example["path"]]
    if examples:
        (output / "misclassified").mkdir()
        fig, axes = plt.subplots(1, len(examples), figsize=(5 * len(examples), 5), squeeze=False)
        for index, (axis, row) in enumerate(zip(axes[0], examples), start=1):
            with Image.open(root / row["path"]) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                # Preview terbatas, bukan unggahan ulang dataset lengkap.
                rgb.thumbnail((640, 640))
                rgb.save(output / "misclassified" / f"example_{index}.jpg")
                axis.imshow(rgb)
            axis.set_title(f"{row['path']}\nAktual: {row['actual']} | Prediksi: {row['predicted']}\nP(Dog)={row['probability_dog']:.3f}")
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(output / "misclassified_examples.png", dpi=140)
        plt.close(fig)
    save_json(output / "misclassified_examples.json", {"total_errors": len(errors), "examples": examples})
    return examples


def write_report(output: Path, config: dict, audit: dict, rows: list[dict],
                 splits: dict, history: list[dict], validation: dict,
                 test: dict, examples: list[dict], best_epoch: int, training_seconds: float) -> None:
    summary = audit["summary"]
    provenance = config["dataset_provenance"]
    download_date = provenance.get("downloaded_at_utc") or "Belum tercatat; isi tanggal unduh asli pada download_metadata.json."
    text = "# Laporan Praktik Deep Learning Sesi 6\n\n"
    text += f"Dataset: [Microsoft Cats vs Dogs di Kaggle]({DATASET_URL}).\n\n"
    text += f"Tanggal unduh UTC: **{download_date}**. Sumber provenance: `run_config.json`.\n\n"
    text += "## Pemeriksaan dataset dan preprocessing\n\n"
    text += (f"Ditemukan {summary['discovered_per_class']}; {summary['corrupt_count']} gambar gagal verifikasi/decode, "
             f"{summary['duplicate_count']} duplikat piksel dihapus, dan {summary['conflicting_label_count']} file dengan label konflik dikeluarkan. "
             f"Tersisa {summary['usable_count']} gambar unik valid, dengan jumlah per kelas {summary['usable_per_class']}. "
             f"File noncitra yang diabaikan: {summary['ignored_count']}; peringatan decoder: {summary['warning_count']}.\n\n")
    text += (f"Lebar asli min/median/max: {summary['original_width']}; tinggi: {summary['original_height']}. "
             f"Jumlah kombinasi ukuran asli: {summary['unique_sizes']}; mode asli: {summary['original_modes']}. "
             "Rincian file rusak dan alasan tersedia secara lokal di `dataset_audit.json`.\n\n")
    text += (f"Gambar dikoreksi orientasi EXIF, dikonversi RGB, lalu di-resize bilinear menjadi {config['image_size']}×{config['image_size']}. "
             "Resize langsung dapat mengubah aspect ratio. Pixel uint8 diubah ke float32 dan dibagi 255 sehingga berada di [0,1]. "
             "Tidak digunakan augmentasi pada baseline ini. Tensor input berbentuk N,C,H,W; Cat=0 dan Dog=1.\n\n")
    text += "## Pembagian data\n\n| Split | Cat | Dog | Total |\n|---|---:|---:|---:|\n"
    for split_name, split_rows in splits.items():
        counts = [sum(row["label"] == label for row in split_rows) for label in range(2)]
        text += f"| {split_name} | {counts[0]} | {counts[1]} | {len(split_rows)} |\n"
    text += (f"\nSplit stratifikasi per kelas dengan seed {config['seed']}; target train/validation/test "
             f"{config['train_ratio']:.0%}/{config['val_ratio']:.0%}/{config['test_ratio']:.0%}. "
             "Pembulatan rasio dilakukan ke bawah untuk validation/test; sisanya masuk train. "
             "Manifest lengkap disimpan di `splits.json` dan hash manifest di `run_config.json`. "
             "Pemeriksaan disjoint path dan hash piksel dilakukan otomatis. Deduplikasi ini belum mendeteksi near-duplicate. "
             "Pemilihan checkpoint hanya berdasarkan validation loss. Threshold 0,5 ditetapkan sebelum training; "
             "test baru dievaluasi sekali setelah checkpoint dipilih.\n\n")
    text += "## Shape dan parameter sebelum training\n\n"
    text += ("Untuk konvolusi dilation=1 dan groups=1: `Hout=floor((Hin+2P−K)/S)+1`, begitu juga W; "
             "`parameter=(K×K×Cin+1)×Cout`, termasuk bias. Kedua conv menggunakan K=3, P=1, S=1. "
             "Pooling menggunakan K=2, S=2, P=0 dan tidak memiliki parameter terlatih. "
             "Linear memiliki `(fitur_input+1)×fitur_output` parameter. Batch N tidak dihitung sebagai fitur atau parameter.\n\n")
    text += architecture_table(rows)
    text += f"\nTotal parameter: **{sum(row['manual_parameters'] for row in rows):,}**. Semua perhitungan diverifikasi melalui dummy forward PyTorch sebelum training.\n\n"
    text += "## Konfigurasi dan training\n\n"
    text += (f"Environment: Python {config['python_version']}, PyTorch {config['torch_version']}, "
             f"NumPy {config['numpy_version']}, Pillow {config['pillow_version']}, Matplotlib {config['matplotlib_version']}. "
             f"Device: {config['device']} ({config['device_name']}); thread CPU: {config['threads']}; "
             f"batch size: {config['batch_size']}; DataLoader workers: {config['workers']}. "
             f"Optimizer Adam, learning rate {config['learning_rate']}; loss BCEWithLogitsLoss; "
             "output satu logit per gambar (sigmoid hanya saat menghitung probabilitas). "
             "Dropout 0,25 di classifier, tanpa pretrained weights/transfer learning.\n\n")
    text += (f"Epoch maksimum: {config['epochs']}; epoch aktual: {len(history)}; patience: {config['patience']}; "
             f"epoch terpilih: {best_epoch}; waktu training termasuk validation setiap epoch dan penyimpanan checkpoint: "
             f"{training_seconds:.2f} detik. Audit, cache preprocessing, dan evaluasi akhir tidak termasuk waktu training ini. "
             "Seed Python/NumPy/PyTorch dan worker diatur; deterministic algorithms aktif. "
             "Kesamaan bit demi bit antar-versi framework/perangkat tidak dijamin.\n\n")
    text += "| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy | Detik |\n|---:|---:|---:|---:|---:|---:|\n"
    for row in history:
        text += (f"| {row['epoch']} | {row['train_loss']:.4f} | {row['train_accuracy']:.4f} | "
                 f"{row['val_loss']:.4f} | {row['val_accuracy']:.4f} | {row['seconds']:.2f} |\n")
    text += "\nTrain metrics pada tabel dihitung selama update bobot dengan dropout aktif; validation dihitung dalam mode eval.\n\n![Training curve](training_curves.png)\n\n"
    text += "## Evaluasi checkpoint terpilih\n\n| Split | Loss | Accuracy | Balanced accuracy | Macro F1 |\n|---|---:|---:|---:|---:|\n"
    for name, metrics in (("Validation", validation), ("Test", test)):
        text += f"| {name} | {metrics['loss']:.4f} | {metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | {metrics['macro_f1']:.4f} |\n"
    text += "\n| Split | Kelas | Precision | Recall | F1 | Support |\n|---|---|---:|---:|---:|---:|\n"
    for name, metrics in (("Validation", validation), ("Test", test)):
        for class_name, values in metrics["per_class"].items():
            text += f"| {name} | {class_name} | {values['precision']:.4f} | {values['recall']:.4f} | {values['f1']:.4f} | {values['support']} |\n"
    text += "\n![Confusion matrix test](confusion_matrix.png)\n\nBaris = aktual; kolom = prediksi; urutan Cat, Dog.\n\n"
    text += "## Contoh salah klasifikasi\n\n"
    if examples:
        text += "![Contoh kesalahan test](misclassified_examples.png)\n\n"
        for index, example in enumerate(examples, start=1):
            text += (f"- Contoh {index}: `{example['path']}`; aktual {example['actual']}, prediksi {example['predicted']}, "
                     f"P(Dog)={example['probability_dog']:.4f}. Interpretasi visual: "
                     f"{example.get('interpretation', '**perlu diisi setelah inspeksi preview gambar**')}.\n")
    if len(examples) < 3:
        text += f"\nHanya {len(examples)} kesalahan ditemukan. Jangan membuat contoh fiktif untuk memenuhi tiga contoh.\n"
    text += "\nInterpretasi visual berasal dari inspeksi gambar; kaitan dengan keputusan model merupakan hipotesis kecuali dibuktikan melalui analisis lebih lanjut.\n"
    text += "\n## Verifikasi kesalahan pipeline\n\n"
    text += ("Risiko shape mismatch: Conv2d memerlukan N,C,H,W, sedangkan array dari Pillow berbentuk H,W,C. "
             "Dataset menjalankan permute(2,0,1); batch pertama setiap epoch memeriksa bentuk N,3,H,W, rentang pixel, dan label. "
             "BCEWithLogitsLoss memerlukan bentuk logit dan target yang sama: keduanya N, bertipe float32 untuk label. "
             "Model hanya memakai squeeze(1) sehingga batch terakhir berukuran satu tetap berbentuk N. "
             "Label selain 0/1 dan logit NaN/Inf memicu error. File rusak diverifikasi dengan verify() lalu decode/load penuh sebelum split; "
             "file gagal dicatat dan dikeluarkan, tanpa mengaktifkan LOAD_TRUNCATED_IMAGES atau mengubah file asli.\n\n")
    text += "## Jawaban pertanyaan analisis\n\nJawaban hanya nomor **1 dan 2** tersedia di [ANALISIS.md](../../ANALISIS.md). Nomor 3–6 disediakan untuk dikerjakan anggota kelompok lain.\n"
    (output / "laporan.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/PetImages"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interpretations-file", type=Path, help="JSON path gambar -> interpretasi visual, hanya dipakai sesudah evaluasi")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--shape-only", action="store_true")
    args = parser.parse_args()
    if min(args.image_size, args.batch_size, args.epochs, args.patience, args.threads) < 1 or args.image_size < 4:
        parser.error("image-size >=4; batch-size, epochs, patience, threads harus positif.")
    if args.workers < 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("workers >=0 dan learning-rate finite >0 diperlukan.")
    if not (0 < args.val_ratio < 1 and 0 < args.test_ratio < 1 and args.val_ratio + args.test_ratio < 1):
        parser.error("Rasio validation/test harus positif dan jumlahnya <1.")
    if not 0 <= args.seed < 2**32:
        parser.error("Seed harus berada di [0, 2**32).")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed, args.threads)
    model = SmallCNN(args.image_size)
    architecture = verify_architecture(model, args.image_size)
    print(architecture_table(architecture), flush=True)
    if args.shape_only:
        return
    root, output = args.data_dir.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output sudah berisi file: {output}. Pilih direktori baru agar hasil sebelumnya tidak tertimpa.")
    output.mkdir(parents=True, exist_ok=True)
    rows, audit = audit_dataset(root)
    splits = split_dataset(rows, args.seed, args.val_ratio, args.test_ratio)
    save_json(output / "dataset_audit.json", audit)
    save_json(output / "audit_summary.json", audit["summary"])
    save_json(output / "splits.json", splits)
    save_json(output / "architecture.json", architecture)
    print(json.dumps(audit["summary"], indent=2), flush=True)
    if args.audit_only:
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA tidak tersedia; gunakan --device cpu atau auto.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    metadata_path = root.parent / "download_metadata.json"
    provenance = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {"dataset_url": DATASET_URL}
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"dataset_url": DATASET_URL, "dataset_provenance": provenance,
                   "run_started_at_utc": datetime.now(timezone.utc).isoformat(),
                   "train_ratio": 1 - args.val_ratio - args.test_ratio,
                   "labels": {name: index for index, name in enumerate(CLASSES)},
                   "threshold": 0.5, "optimizer": "Adam", "loss": "BCEWithLogitsLoss",
                   "input_shape": [None, 3, args.image_size, args.image_size],
                   "parameters": sum(parameter.numel() for parameter in model.parameters()),
                   "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or platform.machine(),
                   "python_version": platform.python_version(), "torch_version": str(torch.__version__),
                   "numpy_version": np.__version__, "pillow_version": Image.__version__,
                   "matplotlib_version": matplotlib.__version__, "platform": platform.platform(),
                   "split_manifest_sha256": hashlib.sha256((output / "splits.json").read_bytes()).hexdigest()})
    save_json(output / "run_config.json", config)
    print("Menyiapkan train/validation loader; test loader dibuat setelah model dipilih.", flush=True)
    generator = torch.Generator().manual_seed(args.seed)
    def make_loader(name: str, shuffle: bool = False) -> DataLoader:
        dataset = CatsDogsDataset(root, splits[name], args.image_size, args.cache_images)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.workers, generator=generator if shuffle else torch.Generator().manual_seed(args.seed + 1),
                          worker_init_fn=seed_worker, pin_memory=device.type == "cuda")
    train_loader = make_loader("train", shuffle=True)
    val_loader = make_loader("validation")
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_loss, best_epoch, wait_count = float("inf"), 0, 0
    history = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics, _ = run_epoch(model, train_loader, loss_fn, device, args.image_size, optimizer)
        val_metrics, _ = run_epoch(model, val_loader, loss_fn, device, args.image_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        row = {"epoch": epoch, "train_loss": train_metrics["loss"],
               "train_accuracy": train_metrics["accuracy"], "val_loss": val_metrics["loss"],
               "val_accuracy": val_metrics["accuracy"], "seconds": time.perf_counter() - epoch_start}
        history.append(row)
        save_json(output / "history.json", history)
        print(f"Epoch {epoch}/{args.epochs}: train loss={row['train_loss']:.4f}, acc={row['train_accuracy']:.4f}; "
              f"val loss={row['val_loss']:.4f}, acc={row['val_accuracy']:.4f}; {row['seconds']:.1f}s", flush=True)
        if val_metrics["loss"] < best_loss:
            best_loss, best_epoch, wait_count = val_metrics["loss"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": config,
                        "best_epoch": best_epoch, "validation_loss": best_loss}, output / "best_model.pt")
        else:
            wait_count += 1
            if wait_count >= args.patience:
                print("Early stopping berdasarkan validation loss.", flush=True)
                break
    training_seconds = time.perf_counter() - started
    checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    validation, _ = run_epoch(model, val_loader, loss_fn, device, args.image_size)
    # Test tidak disentuh oleh loop training/early stopping.
    test_loader = make_loader("test")
    test, probabilities = run_epoch(model, test_loader, loss_fn, device, args.image_size, collect=True)
    save_json(output / "metrics.json", {"best_epoch": best_epoch, "actual_epochs": len(history),
                                       "training_seconds": training_seconds,
                                       "validation": validation, "test": test})
    plot_history(history, output / "training_curves.png")
    plot_confusion(test, output / "confusion_matrix.png")
    interpretations = {}
    if args.interpretations_file:
        interpretations = json.loads(args.interpretations_file.read_text(encoding="utf-8"))
        if not isinstance(interpretations, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in interpretations.items()):
            raise ValueError("File interpretasi harus JSON dengan path gambar sebagai key dan teks sebagai value.")
    examples = export_predictions(splits["test"], probabilities, root, output, interpretations)
    write_report(output, config, audit, architecture, splits, history, validation, test,
                 examples, best_epoch, training_seconds)
    print(f"Selesai. Test accuracy={test['accuracy']:.4f}, macro F1={test['macro_f1']:.4f}. Laporan: {output / 'laporan.md'}", flush=True)


if __name__ == "__main__":
    main()
