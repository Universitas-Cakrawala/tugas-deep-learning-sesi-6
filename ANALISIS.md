# Jawaban Tugas Praktik Deep Learning — Sesi 6

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

CNN memanfaatkan struktur spasial citra. Hubungan pixel yang berdekatan membentuk pola seperti tepi, tekstur, dan bagian objek. Konvolusi memproses lingkungan lokal, sedangkan MLP biasa menerima gambar yang telah diratakan menjadi vektor dan tidak memiliki struktur koneksi khusus untuk hubungan dua dimensi tersebut. Flatten tidak menghapus nilai pixel, tetapi MLP harus mempelajari hubungan spasial melalui bobot dense. Filter CNN digunakan berulang pada berbagai posisi sehingga detektor pola dapat dipakai di seluruh gambar. Penumpukan layer membentuk representasi yang semakin kompleks. Prinsip konektivitas lokal dan berbagi bobot ini dijelaskan dalam [materi CNN Stanford CS231n](https://cs231n.github.io/convolutional-networks/).

Pada dataset Cat/Dog, arsitektur tersebut memungkinkan pembelajaran petunjuk seperti tekstur bulu atau kontur telinga dan wajah, meskipun objek muncul pada posisi berbeda. Ini merupakan alasan pemilihan arsitektur; jenis fitur yang benar-benar dipelajari model ini belum diperiksa dengan visualisasi aktivasi. Pooling merangkum respons lokal dan dapat memberi toleransi terhadap pergeseran kecil, tetapi tidak menjamin invariansi terhadap seluruh pose, rotasi, atau skala.

Sebagai perbandingan yang dihitung dari konfigurasi kode, input 64×64×3 memiliki 12.288 nilai. Satu dense layer MLP dengan 64 neuron memerlukan `(12.288+1)×64 = 786.496` parameter. Conv pertama pada model menggunakan 16 filter 3×3 dan memerlukan `(3×3×3+1)×16 = 448` parameter karena bobot digunakan bersama di seluruh posisi. Perbandingan ini menggambarkan efisiensi berbagi bobot, bukan perbandingan kapasitas atau akurasi dua model yang setara. Seluruh CNN tetap mempunyai 529.505 parameter, terutama pada dense layer setelah flatten; CNN tidak otomatis selalu lebih kecil daripada setiap kemungkinan MLP.

Karena struktur CNN sesuai dengan pola lokal pada citra dan mengurangi kebutuhan mempelajari detektor terpisah untuk setiap posisi, CNN merupakan pilihan yang lebih tepat untuk praktikum ini. Namun, keunggulan akurasi atas MLP pada dataset ini belum dibuktikan melalui eksperimen pembanding. Model juga dapat mempelajari petunjuk background yang keliru jika data training mengandung pola tersebut.

---

## 2. Bagaimana resize, normalisasi, kernel, padding, stride, dan pooling mengubah representasi gambar sebelum masuk ke classifier?

Tahap preprocessing membuat bentuk dan skala input konsisten. Pada implementasi ini, gambar dikonversi ke RGB lalu di-resize bilinear ke 64×64. Resize menyamakan jumlah pixel sehingga gambar dapat dibentuk menjadi batch; pengecilan dapat menghilangkan detail, sedangkan resize langsung ke persegi dapat mengubah proporsi gambar. Bilinear menentukan nilai pixel baru melalui interpolasi; metode ini tersedia dalam [dokumentasi resize Pillow](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize).

Normalisasi pada kode berarti mengubah uint8 menjadi float32 dan membagi pixel dengan 255. Rentang 0–255 berubah menjadi 0–1, tanpa mengubah ukuran tensor. Ini bukan standardisasi menggunakan mean dan standard deviation. Operasi yang sama diterapkan pada train, validation, dan test, sehingga tidak ada statistik normalisasi yang perlu dipelajari dari test set. Array H,W,C juga diubah menjadi C,H,W sesuai format input PyTorch.

Kernel adalah kumpulan bobot yang dipelajari. Kernel 3×3 menggabungkan nilai pada area lokal beserta seluruh channel input menjadi respons fitur; tiap filter menghasilkan satu feature map. Conv pertama mengubah tiga channel RGB menjadi 16 channel fitur, dan conv kedua mengubah 16 menjadi 32. Padding menambahkan batas di sekitar input; zero padding 1 pada kernel 3×3 dan stride 1 mempertahankan tinggi/lebar sekaligus memungkinkan pemrosesan lokasi tepi. Nilai batas tersebut buatan dan dapat memengaruhi respons fitur. Stride menentukan jarak pergeseran kernel: stride lebih besar mengurangi jumlah posisi output dan dapat melewatkan detail. Untuk dilation 1, ukuran spasial mengikuti `floor((ukuran_input + 2×padding − kernel)/stride)+1`. Definisi dan bentuk tensor dijelaskan pada [dokumentasi Conv2d PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html).

Max pooling 2×2 dengan stride 2 mengambil respons maksimum tiap wilayah untuk setiap channel secara terpisah. Tinggi dan lebar masing-masing menjadi setengah, sedangkan jumlah channel tetap. Pooling tidak memiliki bobot terlatih; representasinya lebih ringkas tetapi sebagian informasi posisi dan respons yang tidak maksimum hilang. Operasi tersebut sesuai dengan [dokumentasi MaxPool2d PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MaxPool2d.html).

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

ReLU menambahkan nonlinieritas tanpa mengubah shape. Classifier menerima feature map yang diratakan, kemudian menghasilkan satu logit. Saat training, logit digunakan langsung oleh `BCEWithLogitsLoss`, yang menggabungkan sigmoid dan binary cross entropy secara numerik stabil. Saat evaluasi, sigmoid mengubah logit menjadi P(Dog); nilai ≥0,5 diprediksi Dog dan nilai lebih kecil diprediksi Cat. Lihat [dokumentasi BCEWithLogitsLoss PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html).

---

## 3. Hitung output shape dan parameter untuk input 32×32×3, 16 filter 3×3, padding 1, dan stride 1, lalu jelaskan hubungan contoh tersebut dengan preprocessing dataset.

Menggunakan rumus yang sama dengan yang diimplementasikan pada `manual_architecture()` di [train.py](train.py), `Hout=floor((Hin+2P−K)/S)+1` (berlaku sama untuk W), dan `parameter=(K×K×Cin+1)×Cout` termasuk bias:

- **Output shape**: `Hout = floor((32 + 2×1 − 3)/1) + 1 = floor(31) + 1 = 32`. Karena padding 1 dengan kernel 3 dan stride 1 adalah konfigurasi "same padding" (`P=(K−1)/2`), ukuran spasial tetap 32×32. Dengan 16 filter, output berbentuk **16×32×32** (C,H,W).
- **Parameter**: `(3×3×3+1)×16 = (27+1)×16 = 448`.

Perhitungan ini diverifikasi langsung terhadap PyTorch, bukan hanya manual, dengan menjalankan:

```bash
python train.py --shape-only --image-size 32
```

Hasil aktual (baris `conv1`): output manual `[16, 32, 32]` = output PyTorch, parameter manual `448` = parameter PyTorch — keduanya cocok, konsisten dengan cara `verify_architecture()` memvalidasi setiap layer melalui dummy forward sebelum training (Instruksi #4).

**Hubungan dengan preprocessing dataset**: baseline repo ini tidak menggunakan 32×32, melainkan me-resize seluruh gambar Cat/Dog ke **64×64** (lihat `preprocess()` dan konfigurasi baseline di README). Menjalankan `python train.py --shape-only --image-size 64` menunjukkan conv1 tetap menghasilkan **448 parameter** — identik dengan hasil pada 32×32. Ini karena parameter konvolusi hanya bergantung pada ukuran kernel dan jumlah channel (bobot dibagi/shared di seluruh posisi spasial), **bukan** pada resolusi input; contoh soal 32×32×3 di atas berlaku dengan rumus persis sama seperti yang dipakai pipeline pada 64×64, hanya beda skala.

Namun ukuran resize preprocessing tetap sangat berpengaruh — bukan pada conv1, melainkan pada **dense layer setelah flatten**. Dibandingkan langsung dari dua run aktual:

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

Dari sini, recall per kelas berbeda cukup jauh: **recall Cat = 1575/1872 = 0,8413**, sedangkan **recall Dog = 1402/1872 = 0,7489** — selisih 9,2 poin. Artinya 470 dari 1.872 gambar Dog (25,1%) salah diklasifikasikan sebagai Cat, sementara hanya 297 dari 1.872 gambar Cat (15,9%) salah diklasifikasikan sebagai Dog. Model ini secara sistematis lebih lemah mengenali Dog dibanding Cat — informasi yang sepenuhnya hilang jika hanya melaporkan satu angka accuracy 79,51%.

Ironisnya, tiga contoh kesalahan yang didokumentasikan pada bagian Nomor 5 di atas (dipilih deterministik, bukan berdasarkan confidence) kebetulan semuanya kasus **Cat→Dog**, padahal kategori kesalahan yang secara jumlah lebih besar justru **Dog→Cat** (470 vs 297). Ini menunjukkan bahwa bahkan pemeriksaan kualitatif atas beberapa contoh kesalahan (Instruksi #7) bisa memberi kesan yang menyesatkan tentang kelas mana yang sebenarnya lebih sering salah, jika tidak dicek silang dengan confusion matrix lengkap.

Secara umum, accuracy juga tetap berisiko menyesatkan pada skenario yang lebih ekstrem daripada dataset ini: jika jumlah gambar antar kelas benar-benar tidak seimbang (misalnya 90% Cat, 10% Dog), model yang selalu menebak "Cat" akan mencapai accuracy 90% tanpa pernah mengenali satu pun Dog (recall Dog = 0%) — accuracy tinggi tetapi model tidak berguna. Karena itu, `metrics_from_predictions()` pada `train.py` menghitung metrik tambahan yang tahan terhadap kedua bentuk ketidakseimbangan ini:

- **Confusion matrix** dan **precision/recall/F1 per kelas** — mengungkap kelas mana yang lebih sering salah, seperti dianalisis di atas.
- **Balanced accuracy** (rata-rata recall per kelas, 0,7951 pada test) — pada dataset yang jumlah kelasnya seimbang seperti ini nilainya hampir sama dengan accuracy biasa (79,51% vs 79,51%), tetapi akan menyimpang jauh dari accuracy biasa jika jumlah gambar antar kelas benar-benar tidak seimbang.
- **Macro F1** (0,7947 pada test) — rata-rata tak berbobot F1 antar kelas, sehingga performa buruk pada satu kelas (di sini Dog) tetap tercermin dan tidak "ditenggelamkan" oleh kelas lain yang jumlah sampelnya lebih besar.

Kesimpulannya, accuracy tetap berguna sebagai ringkasan satu angka, tetapi harus selalu didampingi confusion matrix dan metrik per kelas — bukan hanya saat jumlah gambar antar kelas timpang, tetapi juga seperti pada kasus ini, ketika jumlah gambar seimbang namun **kesalahan model antar kelas tidak seimbang**.

---

## Nomor 5 — Analisis Kesalahan Klasifikasi

Berdasarkan hasil aktual run `outputs/baseline` (arsitektur CNN 529.505 parameter, input 64×64×3, test accuracy 79,5%, macro F1 0,795).

Dari total 3.744 sampel test, model menghasilkan **767 kesalahan klasifikasi** (error rate ≈20,5%, konsisten dengan test accuracy 79,5%). Tiga contoh konkret yang diekspor sistem (`outputs/baseline/misclassified_examples.json`), semuanya kasus **Cat diprediksi sebagai Dog**:

### 1. [`example_1.jpg`](https://github.com/Universitas-Cakrawala/tugas-deep-learning-sesi-6/blob/main/outputs/baseline/misclassified/example_1.jpg) (`Cat/10181.jpg`) — probability Dog = 0,729 (kesalahan paling percaya diri)

Gambar berisi kucing di bagian depan, namun ada anjing berukuran cukup besar di latar belakang. Model kemungkinan mendeteksi fitur anjing yang justru dominan secara visual, sehingga prediksinya "salah" terhadap label folder tapi sebenarnya masuk akal terhadap konten citra. Ini menunjukkan **ambiguitas label** pada skema klasifikasi biner single-label untuk citra yang sebenarnya multi-objek sehingga bukan murni kegagalan model mengenali objek.

### 2. [`example_2.jpg`](https://github.com/Universitas-Cakrawala/tugas-deep-learning-sesi-6/blob/main/outputs/baseline/misclassified/example_2.jpg) (`Cat/1947.jpg`) — probability Dog = 0,626

Kucing dalam pose tidak umum: berbaring, kepala menengadah, mulut terbuka, satu kaki terangkat dekat wajah. Pose ini mendistorsi kontur wajah dan telinga yang biasanya jadi fitur pembeda utama kucing vs anjing. Ditambah tekstur bulu yang menyatu dengan karpet di background, dan resize paksa ke 64×64 yang memangkas detail halus wajah. Kombinasi **pose non-frontal + resolusi rendah + background bertekstur mirip** adalah hipotesis penyebab, meski belum dibuktikan lewat analisis aktivasi/Grad-CAM.

### 3. [`example_3.jpg`](https://github.com/Universitas-Cakrawala/tugas-deep-learning-sesi-6/blob/main/outputs/baseline/misclassified/example_3.jpg) (`Cat/8414.jpg`) — probability Dog = 0,601

Ada dua kucing dalam satu frame dengan satu kucing sebagian tertutup (occlusion) oleh kucing lain, ditambah background kompleks (daun, bunga, pot, pagar, selang). Pada resolusi 64×64, batas objek dan detail wajah kemungkinan tenggelam oleh tekstur background yang ramai, sehingga  **occlusion + scene clutter + downsampling** jadi kandidat penyebab.

### Pola Umum

Semua error di atas mengarah ke prediksi Dog dengan probabilitas yang tidak ekstrem (0,60–0,73, dekat threshold 0,5), menandakan model masih "ragu-ragu" dan bukan suatu kesalahan total. Konsisten dengan recall Cat yang justru lebih tinggi (0,84) dibanding recall Dog (0,75) pada test set, sehingga kesalahan arah Cat→Dog memang lebih jarang tapi tetap muncul di kasus-kasus ambigu di atas.

### Eksperimen Lanjutan yang Diusulkan

Terapkan **Grad-CAM / saliency map** pada ketiga (dan sampel error lain) untuk memvalidasi hipotesis di atas secara empiris, "apakah model memang "melihat" ke area anjing/background alih-alih fitur wajah kucing". Ini akan mengonfirmasi apakah akar masalah benar interferensi objek lain dan hilangnya detail akibat resize, atau ada faktor lain (misalnya bias tekstur/warna) yang belum teridentifikasi dan hasilnya bisa mengarahkan perbaikan berikutnya.

---

## Nomor 6 — Menjaga Test Set Tidak Bocor ke Proses Pemilihan Model

Ada beberapa lapisan pengamanan yang diterapkan di [train.py](train.py):

1. **Split dilakukan sekali di awal, sebelum training apa pun**, menggunakan `split_dataset()` dengan seed tetap (42) dan proporsi train 70% / val 15% / test 15%, stratifikasi per kelas.
2. **Deduplikasi berbasis pixel-hash dilakukan sebelum split** — gambar duplikat (26 ditemukan saat audit) dihapus terlebih dahulu agar gambar identik/near-identik tidak jatuh ke split berbeda (yang bisa menyebabkan kebocoran informasi via duplikasi, bukan cuma lewat test set langsung).
3. **Verifikasi eksplisit anti-kebocoran**: setelah split dibuat, kode mengecek irisan `pixel_sha256` dan `path` antar ketiga split — jika ada yang tumpang tindih, program langsung `raise RuntimeError("Kebocoran data...")`. Jadi kebocoran bukan cuma dihindari secara prosedural, tapi divalidasi secara programatik.
4. **Test loader baru dibuat setelah model final dipilih** — dalam loop training, hanya `train_loader` dan `val_loader` yang dipakai. Early stopping dan pemilihan `best_model.pt` sepenuhnya berdasarkan **validation loss**, bukan performa di test set (lihat komentar eksplisit di kode: *"Test tidak disentuh oleh loop training/early stopping"*).
5. **Test set baru dievaluasi satu kali**, setelah checkpoint terbaik (berdasarkan val loss) di-load kembali — sehingga angka test accuracy (79,5%) adalah estimasi generalisasi yang jujur, bukan hasil yang ikut memengaruhi pemilihan epoch/hyperparameter.
6. **Reproducibility & audit trail**: manifest split disimpan (`splits.json`) dan di-hash (`split_manifest_sha256` di `run_config.json`), sehingga siapa pun bisa memverifikasi ulang bahwa split yang dipakai saat training sama dengan yang dilaporkan, dan tidak diubah-ubah di antara percobaan.

Dengan kombinasi ini, test set berfungsi murni sebagai estimator akhir generalisasi, bukan alat untuk memilih arsitektur, optimizer, atau hyperparameter (yang semuanya diputuskan berdasarkan performa di validation set).