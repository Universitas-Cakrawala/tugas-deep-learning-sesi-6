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
RGB dan resize:       3×64×64
Conv1 + ReLU:       16×64×64
MaxPool1:           16×32×32
Conv2 + ReLU:       32×32×32
MaxPool2:           32×16×16
Flatten:              8.192 fitur
Dense + ReLU:            64 fitur
Dropout:                 64 fitur, aktif saat training
Output:                   1 logit
```

ReLU menambahkan nonlinieritas tanpa mengubah shape. Classifier menerima feature map yang diratakan, kemudian menghasilkan satu logit. Saat training, logit digunakan langsung oleh `BCEWithLogitsLoss`, yang menggabungkan sigmoid dan binary cross entropy secara numerik stabil. Saat evaluasi, sigmoid mengubah logit menjadi P(Dog); nilai ≥0,5 diprediksi Dog dan nilai lebih kecil diprediksi Cat. Lihat [dokumentasi BCEWithLogitsLoss PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html).

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

Ada beberapa lapisan pengamanan yang diterapkan di `train.py`:

1. **Split dilakukan sekali di awal, sebelum training apa pun**, menggunakan `split_dataset()` dengan seed tetap (42) dan proporsi train 70% / val 15% / test 15%, stratifikasi per kelas.
2. **Deduplikasi berbasis pixel-hash dilakukan sebelum split** — gambar duplikat (26 ditemukan saat audit) dihapus terlebih dahulu agar gambar identik/near-identik tidak jatuh ke split berbeda (yang bisa menyebabkan kebocoran informasi via duplikasi, bukan cuma lewat test set langsung).
3. **Verifikasi eksplisit anti-kebocoran**: setelah split dibuat, kode mengecek irisan `pixel_sha256` dan `path` antar ketiga split — jika ada yang tumpang tindih, program langsung `raise RuntimeError("Kebocoran data...")`. Jadi kebocoran bukan cuma dihindari secara prosedural, tapi divalidasi secara programatik.
4. **Test loader baru dibuat setelah model final dipilih** — dalam loop training, hanya `train_loader` dan `val_loader` yang dipakai. Early stopping dan pemilihan `best_model.pt` sepenuhnya berdasarkan **validation loss**, bukan performa di test set (lihat komentar eksplisit di kode: *"Test tidak disentuh oleh loop training/early stopping"*).
5. **Test set baru dievaluasi satu kali**, setelah checkpoint terbaik (berdasarkan val loss) di-load kembali — sehingga angka test accuracy (79,5%) adalah estimasi generalisasi yang jujur, bukan hasil yang ikut memengaruhi pemilihan epoch/hyperparameter.
6. **Reproducibility & audit trail**: manifest split disimpan (`splits.json`) dan di-hash (`split_manifest_sha256` di `run_config.json`), sehingga siapa pun bisa memverifikasi ulang bahwa split yang dipakai saat training sama dengan yang dilaporkan, dan tidak diubah-ubah di antara percobaan.

Dengan kombinasi ini, test set berfungsi murni sebagai estimator akhir generalisasi, bukan alat untuk memilih arsitektur, optimizer, atau hyperparameter (yang semuanya diputuskan berdasarkan performa di validation set).