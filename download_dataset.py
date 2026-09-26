"""Unduh dataset resmi Kaggle; arsip dan gambar hanya disimpan lokal."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DATASET_URL = "https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset"
DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/shaunthesheep/microsoft-catsvsdogs-dataset?datasetVersionNumber=1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data"))
    parser.add_argument("--archive", type=Path, help="ZIP dari halaman Kaggle jika unduhan API tidak tersedia")
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "download_metadata.json"
    if (root / "PetImages" / "Cat").is_dir() and (root / "PetImages" / "Dog").is_dir():
        print(f"Dataset sudah ada: {root / 'PetImages'}; metadata lama dipertahankan.")
        return
    archive = args.archive.resolve() if args.archive else root / "cats_vs_dogs.zip"
    downloaded_at = None
    if not args.archive:
        partial = archive.with_suffix(".zip.part")
        print("Mengunduh versi 1 Microsoft Cats vs Dogs dari Kaggle...", flush=True)
        try:
            request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "dl006-praktikum/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as target:
                total = 0
                last_progress = 0
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
                    total += len(chunk)
                    if total - last_progress >= 100 * 1024 * 1024:
                        print(f"Terunduh {total / 1024**2:.0f} MiB", flush=True)
                        last_progress = total
            if not zipfile.is_zipfile(partial):
                raise ValueError("Respons Kaggle bukan arsip ZIP; unduh lewat halaman dataset.")
            partial.replace(archive)
            downloaded_at = datetime.now(timezone.utc).isoformat()
        except (urllib.error.URLError, OSError, ValueError) as error:
            partial.unlink(missing_ok=True)
            raise SystemExit(f"Unduhan gagal: {error}. Unduh dari {DATASET_URL}, lalu gunakan --archive PATH.zip") from error
    if not zipfile.is_zipfile(archive):
        raise SystemExit(f"ZIP tidak valid: {archive}")
    # Validasi semua tujuan sebelum ekstraksi, termasuk entri yang tidak dibutuhkan.
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            target = (root / entry.filename).resolve()
            if not target.is_relative_to(root):
                raise SystemExit(f"Path arsip tidak aman: {entry.filename}")
        zipped.extractall(root)
    if not (root / "PetImages" / "Cat").is_dir() or not (root / "PetImages" / "Dog").is_dir():
        raise SystemExit("Arsip tidak memiliki struktur PetImages/Cat dan PetImages/Dog.")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "dataset_url": DATASET_URL,
        "dataset_version": 1 if not args.archive else "lihat versi pada halaman unduhan manual",
        "download_url": DOWNLOAD_URL if not args.archive else None,
        "downloaded_at_utc": downloaded_at,
        "download_date_note": None if downloaded_at else "Unduhan manual: isi tanggal unduh sebenarnya; waktu ekstraksi bukan tanggal unduh.",
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": digest,
        "archive_bytes": archive.stat().st_size,
        "method": "Kaggle public download API" if not args.archive else "arsip lokal dari Kaggle",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset siap: {root / 'PetImages'}; provenance: {metadata_path}")


if __name__ == "__main__":
    main()
