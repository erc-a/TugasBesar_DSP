# Tugas Besar Mata Kuliah Digital Processing Signal (IF3024)

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

## Teknologi yang Digunakan

----

## Library yang Digunakan

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

## Logbook
### Minggu 1
Membuat Repository GitHub untuk proyek ini.


### Minggu 2
### Minggu 3
### Minggu 4

---

## Referensi

---
