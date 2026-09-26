# Tugas Praktik Deep Learning Sesi 6

CNN kecil untuk klasifikasi **Cat/Dog**, berdasarkan instruksi `981bb4f8-fd8f-4633-b119-bc58967e18e4_dl006.pdf`. Source code berupa `.py`; model dilatih dari awal. Jawaban pertanyaan analisis **hanya nomor 1 dan 2** ada di [ANALISIS.md](ANALISIS.md).

## Environment dan cara menjalankan

Gunakan Python 3.11 atau lebih baru; environment pengujian menggunakan Python 3.12 dan PyTorch 2.8.0 CPU. Dependensi: PyTorch, NumPy, Pillow, Matplotlib. Tidak memerlukan torchvision, notebook, atau GPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python download_dataset.py
python train.py --device cpu --output-dir outputs/baseline
```

Untuk GPU, pasang build PyTorch yang sesuai perangkat melalui petunjuk resmi PyTorch dan gunakan `--device cuda`. Direktori output harus kosong; pilih nama baru untuk setiap eksperimen agar hasil lama tidak tertimpa.

Dataset wajib: [Microsoft Cats vs Dogs oleh shaunthesheep di Kaggle](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset). Script mengunduh versi 1 melalui API publik Kaggle. Jika endpoint membutuhkan autentikasi atau tidak tersedia, unduh ZIP dari halaman dataset lalu jalankan:

```bash
python download_dataset.py --archive /path/arsip-kaggle.zip
```

Struktur lokal yang diperlukan:

```text
data/
├── download_metadata.json
└── PetImages/
    ├── Cat/*.jpg
    └── Dog/*.jpg
```

Dataset baseline diunduh pada **26 September 2026** (18:23 WIB). Tanggal unduh aktual, sumber, ukuran, dan SHA-256 arsip dicatat pada `data/download_metadata.json`, lalu disalin ke `outputs/baseline/run_config.json` serta laporan. Untuk ZIP yang diunduh manual, isi `downloaded_at_utc` dengan tanggal sebenarnya; script tidak menyamakan tanggal ekstraksi dengan tanggal unduh. Dataset dan checkpoint dikecualikan dari Git; hanya preview tiga contoh kesalahan yang disertakan untuk laporan.

## Konfigurasi baseline

RGB **64×64**, resize bilinear, normalisasi `/255`, Cat=0/Dog=1. Dua blok Conv3×3–ReLU–MaxPool2×2 dengan 16 dan 32 filter, flatten, dense 64, dropout 0,25, dan satu logit. Total parameter **529.505**. Adam, learning rate 0,001, batch size 64, maksimum 8 epoch, patience 3, seed 42. Gunakan `python train.py --help` untuk opsi lengkap.

Split **70% train / 15% validation / 15% test**, stratifikasi per kelas setelah gambar rusak dan duplikat piksel dikeluarkan; manifest split disimpan. Jumlah aktual dapat berbeda sedikit karena pembulatan. Checkpoint dipilih berdasarkan validation loss dan test dievaluasi setelah pemilihan model, dengan threshold 0,5 yang telah ditentukan sebelumnya. Jangan mengubah konfigurasi berdasarkan hasil test.

Cache gambar uint8 64×64 memerlukan sekitar 300 MiB untuk dataset lengkap; training juga memerlukan memori tambahan. Gunakan `--no-cache-images` jika RAM terbatas, dengan konsekuensi decode gambar berulang. Default workers=0 agar dapat berjalan di berbagai environment.

## Pemeriksaan tanpa training

```bash
python train.py --shape-only
python train.py --audit-only --output-dir outputs/audit
python verify_pipeline.py
```

Verifikasi mencakup gambar rusak, duplikat dan label konflik, pemisahan split, normalisasi/label, shape batch satu gambar, jumlah parameter, metrik, serta training dan pembacaan checkpoint pada dataset sintetis sementara. Data sintetis hanya untuk pengujian pipeline dan tidak digunakan sebagai hasil praktikum.

## Artefak

[Laporan baseline](outputs/baseline/laporan.md) memuat pemeriksaan dataset, konfigurasi, perhitungan shape/parameter dibandingkan PyTorch, metrik validation/test, kurva training, confusion matrix, dan contoh kesalahan. Hasil numerik tersedia di `metrics.json`, `history.json`, `run_config.json`, `architecture.json`, dan `audit_summary.json` dalam direktori output.

Manifest `splits.json`, audit rinci `dataset_audit.json`, prediksi `test_predictions.csv`, dan `best_model.pt` hanya disimpan lokal. Untuk eksperimen baru, interpretasi visual dalam laporan otomatis perlu diisi setelah melihat preview gambar; jangan mengklaim penyebab kesalahan hanya dari probabilitas model. Pertanyaan analisis nomor 3–6 diserahkan kepada anggota kelompok lain, sedangkan artefak praktik tetap disediakan lengkap.

Baseline pada dataset asli selesai dalam **218,94 detik**, sebanyak 8 epoch. Checkpoint epoch **7** dipilih berdasarkan validation loss; validation accuracy **79,78%**, test accuracy **79,51%**, dan test macro F1 **0,7947** pada 3.744 gambar test. Tujuh pengujian integritas pipeline berhasil.

Interpretasi tiga preview baseline sudah diisi. Untuk menyertakan catatan visual pada laporan eksperimen dengan split/model yang sama, gunakan `--interpretations-file outputs/baseline/misclassification_interpretations.json`; catatan hanya dimasukkan jika path contoh kesalahan cocok.
