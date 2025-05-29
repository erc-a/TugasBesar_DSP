# Tugas Besar Mata Kuliah Digital Processing Signal (IF3024)

<p align="center">
  <img src="docs/images/if3024_logo.png" alt="IF3024 Logo" width="200"/>
</p>

## Dosen Pengampu: **Martin Clinton Tosima Manullang, S.T., M.T..**

---

## **Anggota Kelompok**

| **Nama**                    | **NIM**   | **ID GITHUB**                                                               |
| --------------------------- | --------- | --------------------------------------------------------------------------- |
| Eric Arwido Damanik         | 122140157 | <a href="https://github.com/erc-a">@erc-a</a> |

---

## **Deskripsi Proyek**

Proyek ini merupakan tugas akhir dari mata kuliah "Pengolahan Sinyal Digital IF(3024)" yang dapat
digunakan untuk memperoleh sinyal respirasi dan sinyal remote-photopletysmography (rPPG) dari input
video web-cam secara real-time.

---

## **Teknologi yang Digunakan**

* **Bahasa Pemrograman:** Python 3.9
* **Utama:**
    * **OpenCV:** Untuk akuisisi dan pemrosesan gambar dari webcam.
    * **MediaPipe:** Untuk deteksi pose (estimasi gerakan bahu untuk sinyal pernapasan) dan deteksi wajah (ROI untuk sinyal rPPG).
    * **NumPy:** Untuk operasi numerik, terutama dalam pemrosesan sinyal.
    * **SciPy:** Untuk fungsi pemrosesan sinyal seperti filter (Butterworth) dan pencarian puncak.
    * **PyQt6:** Untuk membangun antarmuka pengguna grafis (GUI) aplikasi.
    * **PyQtGraph:** Untuk menampilkan plot sinyal secara *real-time* di GUI.
    * **Pandas:** Untuk manipulasi data, terutama saat membaca dan menulis file CSV.
    * **Matplotlib:** Untuk menghasilkan plot ringkasan analisis sinyal yang disimpan sebagai gambar.

----

## **Library yang Digunakan**

Berikut adalah daftar library utama yang tercantum dalam `requirements.txt` dan `environment.yml`:

* `opencv-python`
* `mediapipe`
* `numpy`
* `pandas`
* `matplotlib`
* `pyqt6`
* `pyqtgraph`
* `scipy`

---

## **Fitur**

* **Akuisisi Video Real-time:** Mengambil input video langsung dari webcam.
* **Deteksi Sinyal Pernapasan:**
    * Menggunakan MediaPipe Pose untuk melacak gerakan bahu.
    * Menghitung posisi rata-rata vertikal bahu sebagai sinyal mentah pernapasan.
    * Memfilter sinyal pernapasan menggunakan filter *low-pass* Butterworth.
    * Menghitung laju pernapasan (RR) menggunakan FFT dari sinyal yang telah difilter.
* **Deteksi Sinyal rPPG (Detak Jantung):**
    * Menggunakan MediaPipe Face Detection untuk mendeteksi wajah dan menentukan *Region of Interest* (ROI), terutama pada dahi.
    * Mengekstraksi nilai rata-rata channel RGB dari ROI wajah.
    * Menggunakan metode POS (*Plane-Orthogonal-to-Skin*) untuk mengkonversi sinyal RGB menjadi sinyal PPG mentah.
    * Alternatifnya, menggunakan sinyal G-B (*Green minus Blue*) yang dinormalisasi untuk plot *live* di GUI.
    * Sinyal PPG kemudian di-*detrend* dan difilter menggunakan *bandpass filter* Butterworth.
    * Menghitung detak jantung (BPM) dengan mencari puncak pada sinyal PPG yang telah diproses.
* **Antarmuka Pengguna Grafis (GUI):**
    * Menampilkan *feed* video dari webcam secara langsung.
    * Menampilkan plot sinyal pernapasan (setelah filter) dan sinyal rPPG (G-B terfilter atau POS terfilter) secara *real-time* menggunakan PyQtGraph.
    * Menampilkan estimasi RR dan BPM yang diperbarui secara periodik.
    * Tombol untuk memulai dan menghentikan pengambilan data.
    * Menampilkan durasi perekaman.
* **Penyimpanan Data:**
    * Menyimpan data mentah sinyal (timestamp, posisi bahu, nilai mean R, G, B dari ROI) ke dalam file CSV di folder `ouput/`. (Contoh nama file: `data_realtime_signal_YYYYMMDD_HHMMSS.csv`)
    * Menyimpan plot ringkasan analisis sinyal pernapasan dan rPPG (menggunakan Matplotlib) sebagai file gambar di folder `ouput/`.
* **Pemrosesan Sinyal Modular:** Fungsi-fungsi pemrosesan sinyal (filter, POS, FFT, deteksi puncak) diorganisir dalam file `signal_proccesing.py`.

---
## **Cara Menjalankan Proyek**

1.  **Persiapan Environment:**
    * Pastikan Python 3.10 terinstal.
    * Buat environment virtual (disarankan).
    * Instal semua dependensi yang dibutuhkan. Anda bisa menggunakan salah satu dari cara berikut:
        * Menggunakan Conda: `conda env create -f environment.yml` lalu `conda activate realtime_rppg_rr`
        * Menggunakan Pip: `pip install -r requirements.txt`

2.  **Menjalankan Aplikasi:**
    * Buka terminal atau command prompt.
    * Arahkan ke direktori utama proyek.
    * Jalankan skrip utama: `python main_app.py`

3.  **Menggunakan Aplikasi:**
    * Setelah aplikasi terbuka, tekan tombol "Mulai Pengambilan Data".
    * Posisikan wajah dan bahu Anda di depan webcam.
    * Aplikasi akan mulai menampilkan sinyal dan menghitung RR serta BPM.
    * Tekan "Hentikan Pengambilan Data" untuk berhenti. Data CSV dan plot ringkasan akan disimpan di folder `ouput/`.

---
## **Logbook**

### Minggu 1
Membuat Repository GitHub untuk proyek ini.

### Minggu 2
Implementasi fungsionalitas face tracking dan ekstraksi nilai RGB.
Implementasi pemrosesan sinyal rPPG menggunakan algoritma POS dan *bandpass filtering*.

### Minggu 3
Pengembangan fungsi untuk sinyal respirasi.
Penggabungan fungsionalitas rPPG dan respirasi dalam satu skrip utama.

### Minggu 4
Pengembangan Antarmuka Pengguna Grafis (GUI).
Penambahan fungsi-fungsi pemrosesan sinyal untuk analisis respirasi dan detak jantung ke dalam modul terpisah.
Memperbaiki struktur repository.
Pembaruan `README.md` untuk meningkatkan deskripsi fitur, metode pemrosesan sinyal, dan menambahkan `environment.yml` dan 'requirements.txt'.
Pembuataan laporan akhir.

---

## Referensi
* Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. *IEEE Transactions on Biomedical Engineering, 64*(7), 1479-1491. (Digunakan dalam implementasi metode POS)
* Dokumentasi MediaPipe: [https://mediapipe.dev/](https://mediapipe.dev/)
* Dokumentasi OpenCV: [https://opencv.org/](https://opencv.org/)
* Dokumentasi NumPy: [https://numpy.org/doc/](https://numpy.org/doc/)
* Dokumentasi SciPy: [https://docs.scipy.org/doc/scipy/](https://docs.scipy.org/doc/scipy/)
* Dokumentasi PyQt6: [https://www.riverbankcomputing.com/static/Docs/PyQt6/]
* Dokumentasi PyQtGraph: [https://www.pyqtgraph.org/](https://www.pyqtgraph.org/)
* Dokumentasi Pandas: [https://pandas.pydata.org/pandas-docs/stable/](https://pandas.pydata.org/pandas-docs/stable/)
* Dokumentasi Matplotlib: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)

---
