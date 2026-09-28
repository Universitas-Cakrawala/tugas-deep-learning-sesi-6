# Jawaban Tugas Praktik Deep Learning - Sesi 6

## Membangun Convolutional Neural Network (CNN) untuk Klasifikasi Cats vs Dogs

- **Mata Kuliah**: Deep Learning (SDA2110)
- **Sub-CPMK**: Membangun Convolutional Neural Network
- **Topik**: CNN untuk Data Citra (Klasifikasi Biner Cat vs Dog)
- **Kelompok**: Kelompok 5

### Identitas Anggota Kelompok:
| No. | Nama Lengkap | NIM / ID | Peran |
|:---:|:---|:---:|:---:|
| 1 | Tita Noviana | 24120500011 | Ketua |
| 2 | Titanio Yudista | 24120500031 | Anggota |
| 3 | Suci Fransisca Sisilia R | 24120500008 | Anggota |
| 4 | Fajar Dwiharjo | 24130500010 | Anggota |
| 5 | Rafli Ramadhan | 24130500001 | Anggota |

---

## 1. Mengapa CNN lebih sesuai daripada MLP untuk mengklasifikasikan gambar kucing dan anjing pada dataset ini?

CNN memanfaatkan struktur spasial yang memang ada pada citra. Pixel yang letaknya berdekatan biasanya membentuk pola tertentu, misalnya tepi, tekstur, atau bagian dari suatu objek. Konvolusi bekerja pada lingkungan lokal seperti ini, sedangkan MLP biasa menerima gambar yang sudah diratakan menjadi vektor. Akibatnya, MLP tidak memiliki struktur koneksi khusus untuk mempertahankan hubungan dua dimensi tersebut. Flatten sendiri tidak menghilangkan nilai pixel, tetapi setelah flatten MLP tetap harus mempelajari hubungan spasial tersebut melalui bobot dense. Pada CNN, filter yang sama digunakan berulang pada berbagai posisi, sehingga detektor pola dapat dipakai di seluruh bagian gambar. Ketika layer ditumpuk, representasi yang terbentuk juga bisa menjadi semakin kompleks. Prinsip konektivitas lokal dan berbagi bobot ini dijelaskan dalam [materi CNN Stanford CS231n](https://cs231n.github.io/convolutional-networks/).

Pada dataset Cat/Dog, arsitektur seperti ini memungkinkan model memanfaatkan petunjuk visual, misalnya tekstur bulu atau kontur telinga dan wajah, walaupun objeknya muncul di posisi yang berbeda. Ini menjadi salah satu alasan mengapa CNN dipilih untuk praktikum ini. Namun, fitur yang benar-benar dipelajari oleh model belum diperiksa melalui visualisasi aktivasi, jadi bagian tersebut masih berupa penjelasan berdasarkan cara kerja arsitektur. Pooling juga membantu merangkum respons lokal dan dapat memberi toleransi terhadap pergeseran kecil. Meski begitu, pooling tidak menjamin invariansi terhadap semua pose, rotasi, atau skala.

Kalau dibandingkan langsung dari konfigurasi kode, input 64×64×3 memiliki 12.288 nilai. Satu dense layer MLP dengan 64 neuron membutuhkan `(12.288+1)×64 = 786.496` parameter. Sementara itu, Conv pertama pada model hanya membutuhkan `(3×3×3+1)×16 = 448` parameter karena menggunakan 16 filter 3×3 dan bobotnya dipakai bersama di seluruh posisi. Perbandingan ini lebih tepat dibaca sebagai gambaran efisiensi berbagi bobot, bukan sebagai bukti bahwa kedua model memiliki kapasitas atau akurasi yang setara. CNN yang digunakan di sini tetap mempunyai 529.505 parameter, terutama karena adanya dense layer setelah flatten. Jadi, CNN tidak otomatis selalu lebih kecil daripada setiap kemungkinan MLP.

Karena struktur CNN lebih sesuai dengan pola lokal pada citra dan mengurangi kebutuhan untuk mempelajari detektor yang berbeda pada setiap posisi, CNN menjadi pilihan yang lebih sesuai untuk praktikum ini. Tetapi ada satu hal yang perlu diperhatikan: keunggulan akurasi dibandingkan MLP belum dibuktikan melalui eksperimen pembanding pada dataset ini. Selain itu, jika data training memiliki pola tertentu pada background, model juga bisa saja mempelajari petunjuk tersebut meskipun sebenarnya bukan ciri objek yang ingin dikenali.

---

## 2. Bagaimana resize, normalisasi, kernel, padding, stride, dan pooling mengubah representasi gambar sebelum masuk ke classifier?

Tahap preprocessing pada dasarnya membuat bentuk dan skala input menjadi konsisten. Dalam implementasi ini, gambar terlebih dahulu dikonversi ke RGB, lalu di-resize dengan metode bilinear menjadi 64×64. Dengan begitu, jumlah pixel setiap gambar menjadi sama dan gambar dapat dibentuk menjadi batch. Konsekuensinya, pengecilan ukuran dapat menghilangkan sebagian detail, sementara resize langsung ke bentuk persegi juga bisa mengubah proporsi gambar. Pada metode bilinear, nilai pixel baru ditentukan melalui interpolasi. Penjelasan mengenai metode ini tersedia dalam [dokumentasi resize Pillow](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize).

Normalisasi yang dilakukan di kode cukup sederhana. Nilai `uint8` diubah menjadi `float32`, kemudian setiap pixel dibagi dengan 255. Jadi, rentang nilai 0–255 menjadi 0–1 tanpa mengubah ukuran tensor. Ini perlu dibedakan dari standardisasi yang menggunakan mean dan standard deviation. Operasi yang sama diterapkan pada train, validation, dan test, sehingga tidak ada statistik normalisasi yang perlu dipelajari dari test set. Selain itu, array H,W,C diubah menjadi C,H,W agar sesuai dengan format input PyTorch.

Kernel dapat dipahami sebagai kumpulan bobot yang dipelajari oleh model. Kernel 3×3 mengambil area lokal dan menggabungkannya dengan seluruh channel input untuk menghasilkan respons fitur. Setiap filter menghasilkan satu feature map. Pada model ini, Conv pertama mengubah tiga channel RGB menjadi 16 channel fitur, sedangkan conv kedua mengubah 16 channel menjadi 32. Padding menambahkan batas di sekitar input. Dengan zero padding 1, kernel 3×3, dan stride 1, tinggi dan lebar tetap dipertahankan sekaligus memungkinkan area di tepi gambar ikut diproses. Nilai batas ini merupakan nilai buatan, sehingga tetap dapat memengaruhi respons fitur. Stride menentukan seberapa jauh kernel bergeser setiap kali melakukan operasi. Jika stride diperbesar, jumlah posisi output berkurang dan beberapa detail dapat terlewat. Untuk dilation 1, ukuran spasial mengikuti `floor((ukuran_input + 2×padding − kernel)/stride)+1`. Definisi dan bentuk tensor dijelaskan pada [dokumentasi Conv2d PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html).

Max pooling 2×2 dengan stride 2 mengambil nilai respons maksimum dari setiap wilayah pada masing-masing channel. Akibatnya, tinggi dan lebar masing-masing menjadi setengah, sementara jumlah channel tetap. Pooling sendiri tidak memiliki bobot yang dilatih. Representasi yang dihasilkan menjadi lebih ringkas, tetapi sebagian informasi posisi dan respons yang bukan nilai maksimum ikut hilang. Operasi ini sesuai dengan [dokumentasi MaxPool2d PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MaxPool2d.html).

Alur aktual dalam kode, dengan urutan **C,H,W** dan tanpa dimensi batch, adalah:

```text
RGB dan resize:         3×64×64
Conv1 + ReLU:           16×64×64
MaxPool1:               16×32×32
Conv2 + ReLU:           32×32×32
MaxPool2:               32×16×16
Flatten:                8.192 fitur
Dense + ReLU:           64 fitur
Dropout:                64 fitur, aktif saat training
Output:                 1 logit
```

ReLU menambahkan nonlinieritas tanpa mengubah shape. Setelah itu, classifier menerima feature map yang sudah diratakan dan menghasilkan satu logit. Saat training, logit tersebut langsung digunakan oleh `BCEWithLogitsLoss`, yang menggabungkan sigmoid dan binary cross entropy secara numerik stabil. Saat evaluasi, sigmoid mengubah logit menjadi P(Dog). Nilai ≥0,5 diprediksi sebagai Dog, sedangkan nilai yang lebih kecil diprediksi sebagai Cat. Penjelasan loss tersebut dapat dilihat pada [dokumentasi BCEWithLogitsLoss PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html).

---

## 3. Hitung output shape dan parameter untuk input 32×32×3, 16 filter 3×3, padding 1, dan stride 1, lalu jelaskan hubungan contoh tersebut dengan preprocessing dataset.

Dengan menggunakan rumus yang sama seperti yang diimplementasikan pada `manual_architecture()` di [train.py](train.py), yaitu `Hout=floor((Hin+2P−K)/S)+1` (berlaku sama untuk W), serta `parameter=(K×K×Cin+1)×Cout` termasuk bias:

- **Output shape**: `Hout = floor((32 + 2×1 − 3)/1) + 1 = floor(31) + 1 = 32`. Karena padding 1 dengan kernel 3 dan stride 1 adalah konfigurasi "same padding" (`P=(K−1)/2`), ukuran spasial tetap 32×32. Dengan 16 filter, output berbentuk **16×32×32** (C,H,W).
- **Parameter**: `(3×3×3+1)×16 = (27+1)×16 = 448`.

Perhitungan ini diverifikasi langsung terhadap PyTorch, bukan hanya manual, dengan menjalankan:

```bash
python train.py --shape-only --image-size 32
```

Hasil aktual pada baris `conv1` menunjukkan bahwa output manual `[16, 32, 32]` sama dengan output PyTorch, dan parameter manual `448` juga sama dengan parameter PyTorch. Jadi, keduanya cocok. Hasil ini konsisten dengan cara `verify_architecture()` memvalidasi setiap layer melalui dummy forward sebelum training (Instruksi #4).

**Hubungan dengan preprocessing dataset**: baseline repo ini tidak menggunakan 32×32, melainkan me-resize seluruh gambar Cat/Dog ke **64×64** (lihat `preprocess()` dan konfigurasi baseline di README). Ketika menjalankan `python train.py --shape-only --image-size 64`, conv1 tetap menghasilkan **448 parameter**, sama seperti pada hasil 32×32. Hal ini terjadi karena jumlah parameter pada konvolusi hanya bergantung pada ukuran kernel dan jumlah channel. Bobotnya dibagi/shared di seluruh posisi spasial, **bukan** berdasarkan resolusi input. Jadi, contoh soal 32×32×3 di atas tetap menggunakan rumus yang sama dengan pipeline 64×64, hanya ukuran inputnya yang berbeda.

Walaupun begitu, ukuran resize pada preprocessing tetap sangat berpengaruh. Pengaruhnya bukan pada conv1, melainkan pada **dense layer setelah flatten**. Hal ini terlihat jika dua run aktual dibandingkan langsung:

| Ukuran input | Flatten (setelah 2× conv+pool) | Parameter dense | Total parameter model |
|---|---:|---:|---:|
| 32×32×3 (contoh soal) | 32×8×8 = 2.048 | (2.048+1)×64 = 131.136 | **136.289** |
| 64×64×3 (baseline repo, `outputs/baseline/architecture.json`) | 32×16×16 = 8.192 | (8.192+1)×64 = 524.352 | **529.505** |

Jadi keputusan resize pada tahap preprocessing (Instruksi #3) hampir tidak memengaruhi jumlah parameter conv1/conv2 (tetap 448 dan 4.640), tetapi mengubah total parameter model hampir 4× lipat karena efeknya berlipat pada dimensi flatten sebelum dense layer. Ini adalah alasan mengapa laporan wajib mencantumkan ukuran resize secara eksplisit dan menghitung shape/parameter "sebelum menjalankan model" (Instruksi #4): mismatch antara ukuran yang dilaporkan dan yang benar-benar dipakai akan mengubah total parameter secara signifikan tanpa terlihat dari conv layer saja.

---

## 4. Mengapa accuracy saja mungkin belum cukup untuk mengevaluasi klasifikasi Cat versus Dog, terutama jika jumlah gambar atau kesalahan antar-kelas tidak seimbang?

Berdasarkan hasil aktual `outputs/baseline/metrics.json` (checkpoint epoch 7, test accuracy 79,51%, 3.744 sampel test dengan support seimbang 1.872 Cat / 1.872 Dog):

**Dataset ini nyaris seimbang jumlah gambarnya** (audit: 12.480 Cat vs 12.486 Dog gambar unik valid; `outputs/baseline/audit_summary.json`), sehingga argumen klasik "trivial classifier menebak kelas mayoritas" tidak langsung berlaku di sini secara persis. Namun accuracy tunggal (79,51%) tetap menyembunyikan masalah penting: **kesalahan antar-kelas tidak seimbang**, meskipun jumlah gambarnya seimbang.

Confusion matrix test aktual (baris = aktual, kolom = prediksi, urutan Cat, Dog):

```
              Prediksi Cat   Prediksi Dog
Aktual Cat        1575           297
Aktual Dog         470          1402
```

Dari confusion matrix tersebut terlihat bahwa recall tiap kelas cukup berbeda: **recall Cat = 1575/1872 = 0,8413**, sedangkan **recall Dog = 1402/1872 = 0,7489**, dengan selisih 9,2 poin. Artinya, 470 dari 1.872 gambar Dog (25,1%) salah diklasifikasikan sebagai Cat. Sebaliknya, ada 297 dari 1.872 gambar Cat (15,9%) yang salah diklasifikasikan sebagai Dog. Jadi, model lebih sering gagal mengenali Dog dibanding Cat. Perbedaan ini tidak akan terlihat jika yang dilaporkan hanya satu angka accuracy, yaitu 79,51%.

Ironisnya, tiga contoh kesalahan yang didokumentasikan pada bagian Nomor 5 di atas (dipilih deterministik, bukan berdasarkan confidence) kebetulan semuanya kasus **Cat→Dog**, padahal kategori kesalahan yang secara jumlah lebih besar justru **Dog→Cat** (470 vs 297). Ini menunjukkan bahwa bahkan pemeriksaan kualitatif atas beberapa contoh kesalahan (Instruksi #7) bisa memberi kesan yang menyesatkan tentang kelas mana yang sebenarnya lebih sering salah, jika tidak dicek silang dengan confusion matrix lengkap.

Secara umum, accuracy juga bisa menjadi menyesatkan pada kondisi yang lebih ekstrem daripada dataset ini. Misalnya, jika jumlah gambar antar kelas benar-benar tidak seimbang, seperti 90% Cat dan 10% Dog, model yang selalu menebak "Cat" sudah bisa mendapatkan accuracy 90% tanpa pernah mengenali satu pun Dog (recall Dog = 0%). Angkanya memang tinggi, tetapi performanya jelas tidak cukup untuk tugas klasifikasi tersebut. Karena itu, `metrics_from_predictions()` pada `train.py` menghitung beberapa metrik tambahan yang dapat membantu melihat masalah seperti ini:

- **Confusion matrix** dan **precision/recall/F1 per kelas**: membantu melihat kelas mana yang lebih sering salah, seperti yang sudah dianalisis di atas.
- **Balanced accuracy** (rata-rata recall per kelas, 0,7951 pada test): pada dataset yang jumlah kelasnya seimbang seperti ini, nilainya hampir sama dengan accuracy biasa (79,51% vs 79,51%). Jika jumlah gambar antar kelas benar-benar tidak seimbang, nilainya dapat berbeda jauh dari accuracy biasa.
- **Macro F1** (0,7947 pada test): merupakan rata-rata tak berbobot F1 antar kelas. Dengan begitu, performa yang lebih buruk pada satu kelas, dalam kasus ini Dog, tetap terlihat dan tidak "ditenggelamkan" oleh kelas lain yang jumlah sampelnya lebih besar.

Kesimpulannya, accuracy tetap berguna sebagai ringkasan satu angka, tetapi sebaiknya selalu dibaca bersama confusion matrix dan metrik per kelas. Hal ini bukan hanya penting ketika jumlah gambar antar kelas timpang. Pada kasus ini pun jumlah gambar seimbang, tetapi **kesalahan model antar kelas tidak seimbang**.

---

## Nomor 5 - Analisis Kesalahan Klasifikasi

Berdasarkan hasil aktual dari run `outputs/baseline`, model menggunakan arsitektur CNN dengan 529.505 parameter, input 64×64×3, test accuracy 79,5%, dan macro F1 0,795.

Dari total 3.744 sampel test, terdapat **767 kesalahan klasifikasi** (error rate ≈20,5%, konsisten dengan test accuracy 79,5%). Sistem mengekspor tiga contoh konkret pada `outputs/baseline/misclassified_examples.json`, dan ketiganya merupakan kasus **Cat diprediksi sebagai Dog**:

### 1. [`example_1.jpg`](https://github.com/Universitas-Cakrawala/tugas-deep-learning-sesi-6/blob/main/outputs/baseline/misclassified/example_1.jpg) (`Cat/10181.jpg`) - probability Dog = 0,729 (kesalahan paling percaya diri)

Pada gambar ini terdapat kucing di bagian depan, tetapi ada juga anjing yang cukup besar di latar belakang. Ada kemungkinan model menangkap fitur anjing yang secara visual cukup dominan. Karena itu, prediksi tersebut memang "salah" jika dibandingkan dengan label folder, tetapi masih cukup masuk akal jika dilihat dari isi citranya. Contoh ini menunjukkan adanya **ambiguitas label** pada skema klasifikasi biner single-label ketika sebuah citra sebenarnya berisi lebih dari satu objek. Jadi, kasus ini belum tentu murni menunjukkan kegagalan model dalam mengenali objek.

### 2. [`example_2.jpg`](https://github.com/Universitas-Cakrawala/tugas-deep-learning-sesi-6/blob/main/outputs/baseline/misclassified/example_2.jpg) (`Cat/1947.jpg`) - probability Dog = 0,626

Pada contoh ini, kucing berada dalam pose yang tidak umum: berbaring, kepala menengadah, mulut terbuka, dan satu kaki terangkat dekat wajah. Pose seperti ini dapat membuat kontur wajah dan telinga menjadi lebih sulit dibedakan, padahal keduanya biasanya menjadi salah satu fitur pembeda kucing dan anjing. Selain itu, tekstur bulu cukup menyatu dengan karpet di background, sementara resize ke 64×64 juga mengurangi detail halus pada wajah. Jadi, kombinasi **pose non-frontal + resolusi rendah + background bertekstur mirip** dapat menjadi hipotesis penyebabnya. Namun, hipotesis ini belum dibuktikan melalui analisis aktivasi/Grad-CAM.

### 3. [`example_3.jpg`](https://github.com/Universitas-Cakrawala/tugas-deep-learning-sesi-6/blob/main/outputs/baseline/misclassified/example_3.jpg) (`Cat/8414.jpg`) - probability Dog = 0,601

Pada gambar ini terdapat dua kucing dalam satu frame. Salah satunya sebagian tertutup (occlusion) oleh kucing yang lain, dan background-nya juga cukup kompleks, terdiri dari daun, bunga, pot, pagar, dan selang. Pada resolusi 64×64, batas objek dan detail wajah bisa saja menjadi kurang terlihat karena bercampur dengan tekstur background. Karena itu, **occlusion + scene clutter + downsampling** menjadi salah satu kandidat penyebab kesalahan.

### Pola Umum

Semua error di atas menghasilkan prediksi Dog dengan probabilitas yang tidak terlalu ekstrem, yaitu sekitar 0,60–0,73 dan masih cukup dekat dengan threshold 0,5. Ini menunjukkan bahwa model masih "ragu-ragu", bukan membuat kesalahan dengan tingkat keyakinan yang sangat tinggi. Hal ini juga konsisten dengan recall Cat yang lebih tinggi, yaitu 0,84, dibanding recall Dog yang 0,75 pada test set. Jadi, kesalahan Cat→Dog memang lebih jarang, tetapi tetap muncul pada beberapa kasus yang cukup ambigu seperti contoh di atas.

### Eksperimen Lanjutan yang Diusulkan

Langkah berikutnya yang bisa dilakukan adalah menerapkan **Grad-CAM / saliency map** pada ketiga contoh tersebut dan beberapa sampel error lainnya. Tujuannya untuk memeriksa secara empiris, "apakah model memang "melihat" ke area anjing/background alih-alih fitur wajah kucing". Dari sini dapat dilihat apakah dugaan masalahnya memang berasal dari interferensi objek lain dan hilangnya detail akibat resize, atau justru ada faktor lain, misalnya bias tekstur/warna, yang belum terlihat. Hasil analisis tersebut kemudian dapat digunakan sebagai dasar untuk menentukan perbaikan berikutnya.

---

## Nomor 6 - Menjaga Test Set Tidak Bocor ke Proses Pemilihan Model

Ada beberapa lapisan pengamanan yang diterapkan di [train.py](train.py). Tujuannya adalah menjaga agar test set tetap digunakan sebagai evaluasi akhir, bukan ikut memengaruhi proses pemilihan model:

1. **Split dilakukan sekali di awal, sebelum training apa pun**, menggunakan `split_dataset()` dengan seed tetap (42) dan proporsi train 70% / val 15% / test 15%, dengan stratifikasi per kelas.
2. **Deduplikasi berbasis pixel-hash dilakukan sebelum split**. Sebanyak 26 gambar duplikat ditemukan saat audit dan dihapus terlebih dahulu. Dengan cara ini, gambar identik/near-identik tidak masuk ke split yang berbeda, yang bisa menyebabkan kebocoran informasi melalui duplikasi, bukan hanya melalui test set secara langsung.
3. **Verifikasi eksplisit anti-kebocoran**: setelah split dibuat, kode mengecek irisan `pixel_sha256` dan `path` antar ketiga split. Jika ada yang tumpang tindih, program langsung `raise RuntimeError("Kebocoran data...")`. Jadi, kebocoran tidak hanya dihindari secara prosedural, tetapi juga divalidasi secara programatik.
4. **Test loader baru dibuat setelah model final dipilih**. Selama loop training, hanya `train_loader` dan `val_loader` yang digunakan. Early stopping dan pemilihan `best_model.pt` sepenuhnya berdasarkan **validation loss**, bukan performa di test set (lihat komentar eksplisit di kode: *"Test tidak disentuh oleh loop training/early stopping"*).
5. **Test set baru dievaluasi satu kali**, setelah checkpoint terbaik berdasarkan val loss di-load kembali. Dengan begitu, angka test accuracy (79,5%) digunakan sebagai estimasi generalisasi akhir dan tidak ikut memengaruhi pemilihan epoch/hyperparameter.
6. **Reproducibility & audit trail**: manifest split disimpan (`splits.json`) dan di-hash (`split_manifest_sha256` di `run_config.json`). Dengan begitu, siapa pun dapat memverifikasi bahwa split yang digunakan saat training sama dengan yang dilaporkan dan tidak diubah-ubah di antara percobaan.

Dengan kombinasi tersebut, test set tetap berfungsi sebagai estimator akhir generalisasi. Test set tidak digunakan untuk memilih arsitektur, optimizer, atau hyperparameter, karena keputusan tersebut dibuat berdasarkan performa pada validation set.