# Jawaban Pertanyaan Analisis — Nomor 1 dan 2

Dokumen ini hanya menjawab dua nomor yang menjadi bagian tugas saya. Konfigurasi yang dirujuk adalah baseline dalam `train.py`: RGB 64×64, dua blok Conv–ReLU–MaxPool, dan classifier biner tanpa transfer learning.

## 1. Mengapa CNN lebih sesuai daripada MLP untuk mengklasifikasikan gambar kucing dan anjing pada dataset ini?

CNN memanfaatkan struktur spasial citra. Hubungan pixel yang berdekatan membentuk pola seperti tepi, tekstur, dan bagian objek. Konvolusi memproses lingkungan lokal, sedangkan MLP biasa menerima gambar yang telah diratakan menjadi vektor dan tidak memiliki struktur koneksi khusus untuk hubungan dua dimensi tersebut. Flatten tidak menghapus nilai pixel, tetapi MLP harus mempelajari hubungan spasial melalui bobot dense. Filter CNN digunakan berulang pada berbagai posisi sehingga detektor pola dapat dipakai di seluruh gambar. Penumpukan layer membentuk representasi yang semakin kompleks. Prinsip konektivitas lokal dan berbagi bobot ini dijelaskan dalam [materi CNN Stanford CS231n](https://cs231n.github.io/convolutional-networks/).

Pada dataset Cat/Dog, arsitektur tersebut memungkinkan pembelajaran petunjuk seperti tekstur bulu atau kontur telinga dan wajah, meskipun objek muncul pada posisi berbeda. Ini merupakan alasan pemilihan arsitektur; jenis fitur yang benar-benar dipelajari model ini belum diperiksa dengan visualisasi aktivasi. Pooling merangkum respons lokal dan dapat memberi toleransi terhadap pergeseran kecil, tetapi tidak menjamin invariansi terhadap seluruh pose, rotasi, atau skala.

Sebagai perbandingan yang dihitung dari konfigurasi kode, input 64×64×3 memiliki 12.288 nilai. Satu dense layer MLP dengan 64 neuron memerlukan `(12.288+1)×64 = 786.496` parameter. Conv pertama pada model menggunakan 16 filter 3×3 dan memerlukan `(3×3×3+1)×16 = 448` parameter karena bobot digunakan bersama di seluruh posisi. Perbandingan ini menggambarkan efisiensi berbagi bobot, bukan perbandingan kapasitas atau akurasi dua model yang setara. Seluruh CNN tetap mempunyai 529.505 parameter, terutama pada dense layer setelah flatten; CNN tidak otomatis selalu lebih kecil daripada setiap kemungkinan MLP.

Karena struktur CNN sesuai dengan pola lokal pada citra dan mengurangi kebutuhan mempelajari detektor terpisah untuk setiap posisi, CNN merupakan pilihan yang lebih tepat untuk praktikum ini. Namun, keunggulan akurasi atas MLP pada dataset ini belum dibuktikan melalui eksperimen pembanding. Model juga dapat mempelajari petunjuk background yang keliru jika data training mengandung pola tersebut.

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
