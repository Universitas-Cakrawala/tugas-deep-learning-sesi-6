# Review Kesesuaian Tugas Praktik Deep Learning Sesi 6

Review dilakukan pada 26 September 2026 terhadap seluruh isi `981bb4f8-fd8f-4633-b119-bc58967e18e4_dl006.pdf`, source code, dataset lokal asli, checkpoint, dan artefak baseline. Ruang lingkup jawaban analisis tetap hanya nomor 1 dan 2 sesuai permintaan; nomor 3–6 merupakan bagian anggota kelompok lain.

## Kesimpulan

Delapan instruksi praktik dan tiga kategori artefak wajib sudah terpenuhi pada baseline yang disimpan. Tidak ditemukan ketidaksesuaian pada arsitektur, preprocessing, pembagian data, pemilihan checkpoint, perhitungan parameter, ataupun hasil evaluasi. Dua masalah pada cara menjalankan ulang dan tautan laporan ditemukan serta diperbaiki. Bagian analisis nomor 1–2 sudah lengkap; penyerahan kelompok secara keseluruhan masih perlu menggabungkan jawaban nomor 3–6 dari teman.

## Temuan dan perbaikan

1. **Sedang — perintah README gagal pada repository yang sudah berisi baseline.** Perintah awal menggunakan `--output-dir outputs/baseline`, sedangkan folder tersebut sudah menyimpan hasil yang terlacak Git. Kode menolak folder output yang tidak kosong, sehingga perintah berhenti sebelum audit/training. Masalah dibuktikan dengan menjalankan perintah persis: exit code 1 dan pesan `Output sudah berisi file`. Perintah README, contoh pada source, dan default CLI sekarang menggunakan `outputs/run`. Hasil baseline tetap disimpan; proteksi terhadap penimpaan hasil tetap aktif.
2. **Rendah — tautan analisis tidak berlaku untuk semua lokasi output.** Generator laporan menghardcode `../../ANALISIS.md`. Tautan itu benar untuk `outputs/baseline`, tetapi salah untuk direktori sementara, output satu tingkat, atau lokasi custom lain. Generator sekarang menghitung path relatif dari lokasi laporan menuju file analisis sebenarnya. Pengujian eksekusi lengkap diperluas untuk memastikan target tautan ada dan mengarah ke `ANALISIS.md` proyek.

Kedua perbaikan tidak mengubah arsitektur, split, optimizer, hyperparameter, bobot checkpoint, atau hasil baseline.

## Pemetaan delapan instruksi praktik

| Instruksi PDF | Bukti pengerjaan | Hasil review |
|---|---|---|
| 1. Unduh dataset Kaggle dan periksa struktur, jumlah file, ukuran, serta gambar rusak sebelum training | `download_dataset.py` memakai API Kaggle versi 1. `audit_dataset()` memeriksa Cat/Dog, ekstensi, verify, decode penuh, ukuran, mode, duplikat, dan konflik label sebelum split/training. Audit ulang seluruh 25.000 gambar cocok dengan baseline. | Terpenuhi |
| 2. Pembagian train/validation/test reproducible; test tidak untuk memilih model | Split stratifikasi dengan seed 42 dan rasio 70/15/15. Manifest dan hash split cocok; path dan hash piksel antarsplit tidak beririsan. Audit ulang menghasilkan manifest identik. Loop training hanya memakai train dan validation; checkpoint dipilih dari validation loss. | Terpenuhi |
| 3. Resize, normalisasi, dan label konsisten | Semua split memakai fungsi preprocessing yang sama: koreksi EXIF, RGB, resize bilinear 64×64, C,H,W, float32 /255. Label Cat=0/Dog=1 berbentuk N untuk output satu logit dan BCEWithLogitsLoss. | Terpenuhi |
| 4. Hitung shape/parameter conv sebelum menjalankan model dan bandingkan dengan framework | Perhitungan independen dijalankan sebelum dummy forward. Conv1 menghasilkan 16×64×64 dengan 448 parameter; Conv2 menghasilkan 32×32×32 dengan 4.640 parameter. Tabel manual/PyTorch cocok dengan arsitektur checkpoint; total 529.505. | Terpenuhi |
| 5. CNN kecil dengan minimal dua blok convolution, activation, pooling, dan classifier biner | Dua blok Conv3×3–ReLU–MaxPool2×2, dilanjutkan flatten, dense 64, ReLU, dropout, dan satu output logit. Model dilatih dari awal. | Terpenuhi |
| 6. Catat seluruh konfigurasi dan waktu training | `run_config.json`, `metrics.json`, dan laporan mencatat arsitektur, input shape, parameter, Adam, BCEWithLogitsLoss, LR 0,001, CPU, batch 64, maksimum/aktual 8 epoch, waktu 218,94 detik, dan seed 42. | Terpenuhi |
| 7. Kurva training, hasil validation, confusion matrix, dan minimal tiga salah klasifikasi beserta interpretasi | Laporan menampilkan kurva loss/accuracy, tabel validation/test, confusion matrix, dan tiga contoh Cat→Dog dengan path, probabilitas, serta interpretasi hasil inspeksi gambar. Ketiganya cocok dengan prediksi test yang tersimpan. | Terpenuhi |
| 8. Jelaskan kemungkinan mismatch/error dan cara memverifikasinya | Laporan menjelaskan H,W,C versus N,C,H,W, shape logit/label N, label float32 0/1, serta verify/decode gambar rusak. Pemeriksaan batch dan pengujian mencakup shape salah, label salah, batch satu gambar, dan file rusak. | Terpenuhi |

## Dataset dan hasil yang diverifikasi

- URL sumber dan tanggal unduh tercantum di README, laporan, serta metadata run. Tanggal unduh: 26 September 2026, sekitar 18:23 WIB. SHA-256 arsip lokal sesuai provenance dalam `run_config.json`.
- Dari 25.000 gambar: 2 gagal dibaca, 26 duplikat piksel dihapus, dan 6 gambar dengan label konflik dikeluarkan. Tersisa 24.966: Cat 12.480 dan Dog 12.486. Ada dua peringatan decoder pada satu file yang tetap lolos decode; peringatan tidak disamakan dengan kegagalan baca.
- Jumlah aktual: train 17.478, validation 3.744, test 3.744. Pembulatan rasio per kelas dijelaskan pada laporan.
- Checkpoint epoch 7 sesuai minimum validation loss pada history. Epoch 8 memiliki validation accuracy lebih tinggi tetapi loss sedikit lebih tinggi; memilih epoch 7 tetap benar karena kriteria selection yang ditetapkan adalah loss.
- Seluruh 3.744 baris CSV memiliki path dan label sesuai manifest test. Prediksi mengikuti threshold 0,5. Accuracy, precision, recall, F1, balanced accuracy, serta confusion matrix dihitung ulang dari CSV dan cocok dengan `metrics.json`.
- Confusion matrix test: `[[1575, 297], [470, 1402]]`, baris aktual dan kolom prediksi dalam urutan Cat/Dog. Total benar 2.977; accuracy 2.977/3.744 = 79,5139%. Total kesalahan 767.
- Validation accuracy checkpoint terpilih: 79,7810%; test accuracy: 79,5139%; test macro F1: 0,7947006.
- Evaluasi ulang validation memakai checkpoint dan 3.744 gambar asli menghasilkan seluruh metrik yang sama. Test BCE dihitung independen dari probabilitas CSV: 0,4285070033, cocok dengan loss tersimpan 0,4285070022 dalam toleransi 0,000001.
- Seluruh 7 pengujian pipeline lulus setelah perbaikan, termasuk training singkat, pembacaan checkpoint, proteksi output, dan target tautan laporan. Pemeriksaan sintaks serta `git diff --check` lulus.

## Artefak dan jawaban analisis

Source [train.py](train.py) dan [download_dataset.py](download_dataset.py) berupa `.py`, disertai [requirements.txt](requirements.txt) dan [README.md](README.md). [Laporan baseline](outputs/baseline/laporan.md) berisi shape/parameter, tabel metrik, visualisasi, dan confusion matrix. Dataset lengkap serta checkpoint tidak terlacak Git; tiga preview gambar dipakai untuk memenuhi laporan.

[ANALISIS.md](ANALISIS.md) hanya mempunyai jawaban nomor 1 dan 2. Nomor 1 membahas struktur spasial, konektivitas lokal, berbagi bobot, fitur hierarkis, dan batas perbandingan CNN/MLP. Nomor 2 mencakup keenam operasi yang ditanyakan dan hubungan dengan alur classifier aktual. Tidak ada klaim bahwa keunggulan akurasi atas MLP telah diuji.

Input 32×32×3 pada pertanyaan analisis nomor 3 bukan ketentuan ukuran wajib untuk praktik. Oleh karena itu, pemakaian RGB 64×64 yang dinyatakan konsisten di kode dan laporan memenuhi instruksi preprocessing. PDF tidak menetapkan accuracy minimum, optimizer tertentu, kewajiban augmentasi, jumlah epoch tertentu, atau penggunaan transfer learning. Tiga contoh seluruhnya dari kelas Cat juga diperbolehkan karena instruksi menyebut Cat **atau** Dog.

## Batas kesimpulan review

Reproducibility split telah diverifikasi pada arsip dan environment saat ini. Deduplikasi hanya mendeteksi piksel identik; near-duplicate belum dibuktikan tidak ada. Dependencies selain PyTorch memiliki rentang versi, sehingga kesamaan angka training pada environment berbeda tidak dijamin; versi baseline dicatat lengkap dalam konfigurasi dan laporan.

Contoh pertama memuat kucing dan anjing sekaligus sehingga label biner Cat ambigu. Dua interpretasi lainnya membahas pose, detail yang hilang, dan background sebagai hipotesis visual, bukan penyebab keputusan model yang telah dibuktikan. Batas ini sudah dijelaskan dalam laporan. Untuk eksperimen baru, interpretasi laporan perlu diperiksa dan diisi sesuai gambar yang benar-benar terpilih; placeholder otomatis belum memenuhi instruksi 7 sampai dilengkapi. Baseline yang dikumpulkan tidak memiliki placeholder tersebut.
