import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

# --- FUNGSI-FUNGSI UNTUK PEMROSESAN SINYAL ---

def cpu_POS(rgb_signal, fps):
    """
    POS method on CPU using Numpy. This converts raw RGB signal to a pulse signal.
    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016).
    """
    # Memastikan ada cukup data untuk diproses
    if rgb_signal.shape[1] < 2:
        return np.zeros(rgb_signal.shape[1])
        
    w = int(1.6 * fps)  # Panjang window, direkomendasikan 1.6 detik
    if rgb_signal.shape[1] < w:
        w = rgb_signal.shape[1]

    # Temporal normalization
    X = rgb_signal.T
    mean_color = np.mean(X, axis=0)
    # Hindari pembagian dengan nol jika channel berwarna hitam
    diag_mean_color = np.diag(1 / (mean_color + 1e-9))
    Xn = np.matmul(X, diag_mean_color)
    Xn = Xn.T

    # Projection
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    S = np.dot(P, Xn)
    
    # Tuning
    H = np.zeros(S.shape[1])
    for i in range(S.shape[1] - w):
        S_window = S[:, i:i+w]
        S_std = np.std(S_window, axis=1)
        
        # Hindari pembagian dengan nol
        if S_std[1] < 1e-9:
            alpha = 0
        else:
            alpha = S_std[0] / S_std[1]
            
        H_window = S_window[0, :] + alpha * S_window[1, :]
        H[i:i+w] = H[i:i+w] + (H_window - np.mean(H_window))
        
    return H

def filter_sinyal_respirasi(data, sample_rate, cutoff_freq=0.8, order=2):
    """Filter sinyal respirasi menggunakan low-pass filter."""
    if len(data) < 20:
        return np.array(data)
    b, a = butter(order, cutoff_freq, btype='low', fs=sample_rate)
    return filtfilt(b, a, data)

def hitung_laju_napas(sinyal, sample_rate):
    """Menghitung laju napas dari sinyal menggunakan FFT."""
    if len(sinyal) < int(2 * sample_rate):
        return "N/A"
    
    sinyal_tanpa_dc = sinyal - np.mean(sinyal)
    fft_result = np.fft.fft(sinyal_tanpa_dc)
    freqs = np.fft.fftfreq(len(sinyal), 1.0 / sample_rate)
    
    idx = np.where((freqs >= 0.1) & (freqs <= 0.8))
    if len(idx[0]) == 0:
        return "N/A"
    
    dominant_freq_idx = idx[0][np.argmax(np.abs(fft_result[idx]))]
    dominant_freq = freqs[dominant_freq_idx]
    
    return f"{dominant_freq * 60:.1f} napas/menit"

def bandpass_filter_rppg(data, sample_rate, lowcut=0.7, highcut=2.5, order=4):
    """Bandpass filter sinyal rPPG."""
    if len(data) < int(2 * sample_rate):
        return np.array(data)
    b, a = butter(order, [lowcut, highcut], btype='band', fs=sample_rate)
    return filtfilt(b, a, data)

def hitung_detak_jantung(raw_rgb_signal, fps):
    """
    Menghitung detak jantung dari sinyal RGB mentah.
    Langkah: RGB -> POS -> Bandpass Filter -> Cari Puncak.
    """
    if raw_rgb_signal.shape[1] < int(2 * fps):
        return "N/A"

    # 1. Ubah sinyal RGB mentah menjadi sinyal denyut menggunakan POS
    pulse_signal = cpu_POS(raw_rgb_signal, fps)
    
    # 2. Filter sinyal denyut
    filtered_pulse = bandpass_filter_rppg(pulse_signal, fps)
    
    # 3. Cari puncak (detak jantung)
    min_height = np.std(filtered_pulse) * 0.5
    peaks, _ = find_peaks(filtered_pulse, height=min_height, distance=fps / 2.5)
    
    # 4. Hitung BPM
    durasi_detik = raw_rgb_signal.shape[1] / fps
    if durasi_detik == 0: return "N/A"
    
    bpm = len(peaks) * (60 / durasi_detik)
    return f"{bpm:.1f} BPM"