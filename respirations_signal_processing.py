import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import numpy as np

# --- PENGATURAN ---
FILE_CSV_INPUT = 'data_respirasi.csv'
FILE_GRAFIK_OUTPUT = 'grafik_sinyal_respirasi.png'
# --------------------

def filter_sinyal(data, cutoff_freq=0.8, sample_rate=30, order=2):
    """
    Fungsi untuk memfilter sinyal menggunakan low-pass filter Butterworth.
    Ini membantu menghaluskan data dan menghilangkan noise frekuensi tinggi.
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def hitung_frekuensi_napas(sinyal, timestamps):
    """
    Menghitung frekuensi napas dalam napas per menit (BPM - Breaths Per Minute).
    """
    # Menghitung puncak (puncak napas)
    fft_result = np.fft.fft(sinyal - np.mean(sinyal))
    freqs = np.fft.fftfreq(len(sinyal), d=(timestamps[-1] - timestamps[0]) / len(sinyal))
    
    # Cari frekuensi dominan di rentang napas normal (0.1-0.8 Hz)
    idx = np.where((freqs > 0.1) & (freqs < 0.8))
    if len(idx[0]) == 0:
        return 0
    
    dominant_freq_idx = idx[0][np.argmax(np.abs(fft_result[idx]))]
    dominant_freq = freqs[dominant_freq_idx]
    
    # Konversi dari Hz ke napas per menit
    napas_per_menit = dominant_freq * 60
    return napas_per_menit

try:
    # Baca data dari file CSV
    df = pd.read_csv(FILE_CSV_INPUT)
except FileNotFoundError:
    print(f"Error: File '{FILE_CSV_INPUT}' tidak ditemukan.")
    print("Pastikan kamu sudah menjalankan 'tangkap_sinyal.py' terlebih dahulu.")
    exit()

if df.empty:
    print("File CSV kosong. Tidak ada data untuk diproses.")
    exit()
    
# Ambil data waktu dan posisi y rata-rata bahu
timestamps = df['timestamp'].values
posisi_y = df['avg_shoulder_y'].values

# Hitung sample rate aktual dari data
sample_rate = 1 / np.mean(np.diff(timestamps))

# Filter sinyal untuk memperhalus grafik
sinyal_terfilter = filter_sinyal(posisi_y, sample_rate=sample_rate)

# Hitung laju pernapasan
laju_napas = hitung_frekuensi_napas(sinyal_terfilter, timestamps)

# Membuat plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 6))

# Plot data asli (mentah)
ax.plot(timestamps, posisi_y, 'o-', color='lightblue', markersize=3, label='Sinyal Asli (Mentah)')

# Plot data yang sudah difilter
ax.plot(timestamps, sinyal_terfilter, '-', color='royalblue', linewidth=2, label='Sinyal Terfilter (Pernapasan)')

# Pengaturan Grafik
ax.set_title(f'Analisis Sinyal Pernapasan\nEstimasi Laju Napas: {laju_napas:.2f} napas/menit', fontsize=16)
ax.set_xlabel('Waktu (detik)', fontsize=12)
# Sumbu Y dibalik karena koordinat piksel diukur dari atas ke bawah
ax.set_ylabel('Posisi Vertikal Bahu (piksel)', fontsize=12)
ax.invert_yaxis() 
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Menyimpan grafik ke file
plt.savefig(FILE_GRAFIK_OUTPUT)
print(f"Grafik sinyal disimpan di: {FILE_GRAFIK_OUTPUT}")

# Menampilkan grafik
plt.show()