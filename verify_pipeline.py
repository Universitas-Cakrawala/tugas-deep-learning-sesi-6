"""Pengujian integritas pipeline; seluruh gambar dibuat di direktori sementara."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from train import (CatsDogsDataset, SmallCNN, audit_dataset, check_batch,
                   export_predictions, metrics_from_predictions, set_seed, split_dataset, verify_architecture)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_seed(42, 1)
        cls.temporary = tempfile.TemporaryDirectory(prefix="dl006-verification-")
        cls.root = Path(cls.temporary.name) / "PetImages"
        rng = np.random.default_rng(13)
        for name in ("Cat", "Dog"):
            folder = cls.root / name
            folder.mkdir(parents=True)
            for index in range(12):
                pixels = rng.integers(0, 256, size=(18, 23, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{index:02d}.png")
        (cls.root / "Cat" / "copy.png").write_bytes((cls.root / "Cat" / "00.png").read_bytes())
        for name in ("Cat", "Dog"):
            Image.fromarray(np.full((18, 23, 3), 77, dtype=np.uint8)).save(cls.root / name / "conflict.png")
        (cls.root / "Cat" / "broken.jpg").write_bytes(b"not an image")
        (cls.root / "Cat" / "notes.txt").write_text("non-image")
        cls.rows, cls.audit = audit_dataset(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_corrupt_duplicate_and_conflicting_labels(self):
        summary = self.audit["summary"]
        self.assertEqual(summary["corrupt_count"], 1)
        self.assertEqual(summary["duplicate_count"], 1)
        self.assertEqual(summary["conflicting_label_count"], 2)
        self.assertEqual(summary["usable_per_class"], {"Cat": 12, "Dog": 12})
        self.assertEqual(summary["ignored_count"], 1)
        self.assertEqual(self.audit["rejected"][0]["path"], "Cat/broken.jpg")

    def test_reproducible_stratified_disjoint_split(self):
        first = split_dataset(self.rows, 42, 0.15, 0.15)
        repeated = split_dataset(list(reversed(self.rows)), 42, 0.15, 0.15)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, split_dataset(self.rows, 43, 0.15, 0.15))
        self.assertEqual([len(first[name]) for name in ("train", "validation", "test")], [20, 2, 2])
        fingerprints = [{row["pixel_sha256"] for row in split} for split in first.values()]
        self.assertFalse(fingerprints[0] & fingerprints[1])
        self.assertFalse(fingerprints[0] & fingerprints[2])
        self.assertFalse(fingerprints[1] & fingerprints[2])
        for rows in first.values():
            self.assertEqual(sum(row["label"] == 0 for row in rows), sum(row["label"] == 1 for row in rows))

    def test_preprocessing_and_batch_of_one(self):
        for cache in (False, True):
            dataset = CatsDogsDataset(self.root, self.rows[:2], 16, cache)
            image, label = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 16, 16))
            self.assertEqual(image.dtype, torch.float32)
            self.assertEqual(label.dtype, torch.float32)
            self.assertGreaterEqual(float(image.min()), 0)
            self.assertLessEqual(float(image.max()), 1)
            images, labels = image.unsqueeze(0), label.unsqueeze(0)
            model = SmallCNN(16)
            logits = model(images)
            self.assertEqual(tuple(logits.shape), (1,))
            check_batch(images, labels, logits, 16)
            loss = nn.BCEWithLogitsLoss()(logits, labels)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            with self.assertRaisesRegex(ValueError, "Shape logit"):
                check_batch(images, labels.unsqueeze(1), logits, 16)
            with self.assertRaisesRegex(ValueError, "Label BCE"):
                check_batch(images, torch.tensor([2.0]), logits, 16)
        gray = Path(self.temporary.name) / "gray.png"
        Image.fromarray(np.full((7, 9), 255, dtype=np.uint8)).save(gray)
        dataset = CatsDogsDataset(gray.parent, [{"path": gray.name, "label": 0}], 16, False)
        image, _ = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 16, 16))
        self.assertTrue(torch.all(image == 1))

    def test_manual_parameters_and_odd_image_size(self):
        for size in (16, 64, 65):
            model = SmallCNN(size)
            rows = verify_architecture(model, size)
            self.assertTrue(all(row["verified"] for row in rows))
            if size == 64:
                self.assertEqual(sum(row["manual_parameters"] for row in rows), 529505)
            self.assertEqual(tuple(model(torch.zeros(1, 3, size, size)).shape), (1,))

    def test_metrics_with_known_predictions(self):
        metrics = metrics_from_predictions([0, 0, 1, 1], [0, 1, 1, 1], 0.5)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["per_class"]["Cat"]["precision"], 1.0)
        self.assertEqual(metrics["per_class"]["Cat"]["recall"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 0.8) / 2)

    def test_visual_interpretation_export(self):
        destination = Path(self.temporary.name) / "interpretations"
        destination.mkdir()
        first_path = self.rows[0]["path"]
        note = "Catatan hasil inspeksi manual, bukan prediksi penyebab otomatis"
        examples = export_predictions(self.rows[:3], [0.9, 0.9, 0.9], self.root,
                                      destination, {first_path: note})
        self.assertEqual(len(examples), 3)
        self.assertEqual(examples[0]["interpretation"], note)
        self.assertNotIn("interpretation", examples[1])
        data = json.loads((destination / "misclassified_examples.json").read_text())
        self.assertEqual(data["total_errors"], 3)
        self.assertEqual(data["examples"][0]["interpretation"], note)
        self.assertTrue((destination / "misclassified_examples.png").is_file())

    def test_end_to_end_checkpoint_and_artifacts(self):
        output = Path(self.temporary.name) / "run"
        script = Path(__file__).resolve().with_name("train.py")
        result = subprocess.run([
            sys.executable, str(script), "--data-dir", str(self.root),
            "--output-dir", str(output), "--image-size", "16", "--epochs", "1",
            "--batch-size", "8", "--threads", "1", "--device", "cpu", "--no-cache-images",
        ], capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in ("best_model.pt", "laporan.md", "metrics.json", "training_curves.png",
                     "confusion_matrix.png", "splits.json", "dataset_audit.json", "test_predictions.csv"):
            self.assertTrue((output / name).is_file(), name)
        report = (output / "laporan.md").read_text()
        analysis_links = re.findall(r"\[ANALISIS\.md\]\(([^)]+)\)", report)
        self.assertEqual(len(analysis_links), 1)
        target = (output / analysis_links[0].strip("<>")).resolve()
        self.assertTrue(target.is_file(), f"Link analisis laporan tidak valid: {target}")
        self.assertEqual(target, script.with_name("ANALISIS.md"))
        metrics = json.loads((output / "metrics.json").read_text())
        self.assertEqual(metrics["test"]["samples"], 2)
        checkpoint = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["best_epoch"], 1)
        model = SmallCNN(16)
        model.load_state_dict(checkpoint["state_dict"])
        self.assertTrue(torch.isfinite(model(torch.zeros(1, 3, 16, 16))).all())
        # Direktori berisi hasil tidak boleh tertimpa.
        result = subprocess.run([sys.executable, str(script), "--output-dir", str(output)],
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output sudah berisi file", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
