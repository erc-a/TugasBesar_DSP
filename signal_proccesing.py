import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, detrend

def detrend_signal(signal_data, type='linear'):
    """
    Menghilangkan tren dari sinyal.
    'linear': Menghilangkan tren linear.
    'constant': Sama seperti mengurangi mean.
    """
    if signal_data.ndim > 1:
        detrended_signal = np.zeros_like(signal_data)
        for i in range(signal_data.shape[0]):
            detrended_signal[i, :] = detrend(signal_data[i, :], type=type)
        return detrended_signal
    else:
        return detrend(signal_data, type=type)

def cpu_POS(rgb_signal, fps):
    """
    POS method on CPU using Numpy. This converts raw RGB signal to a pulse signal.
    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016).
    """
    if rgb_signal.shape[1] < 2:
        return np.zeros(rgb_signal.shape[1])
        
    w = int(1.6 * fps)
    if rgb_signal.shape[1] < w:
        w = rgb_signal.shape[1]

    X = rgb_signal.T
    mean_color = np.mean(X, axis=0)
    diag_mean_color = np.diag(1 / (mean_color + 1e-9))
    Xn = np.matmul(X, diag_mean_color)
    Xn = Xn.T

    P = np.array([[0, 1, -1], [-2, 1, 1]])
    S = np.dot(P, Xn)
    
    H = np.zeros(S.shape[1])
    for i in range(S.shape[1] - w):
        S_window = S[:, i:i+w]
        S_std = np.std(S_window, axis=1)
        
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

    nyquist = 0.5 * sample_rate
    if not (0 < cutoff_freq < nyquist):
        return np.array(data)
        
    b, a = butter(order, cutoff_freq, btype='low', fs=sample_rate)
    return filtfilt(b, a, data)

def hitung_laju_napas(sinyal, sample_rate):
    """Menghitung laju napas dari sinyal menggunakan FFT."""
    if len(sinyal) < int(2 * sample_rate) or sample_rate <= 0:
        return "N/A"
    
    sinyal_tanpa_dc = sinyal - np.mean(sinyal)
    fft_result = np.fft.fft(sinyal_tanpa_dc)
    freqs = np.fft.fftfreq(len(sinyal), 1.0 / sample_rate)
    
    idx = np.where((freqs >= 0.1) & (freqs <= 0.8)) 
    if len(idx[0]) == 0:
        return "N/A (Tidak ada frekuensi dominan di rentang napas)"
    
    dominant_freq_idx = idx[0][np.argmax(np.abs(fft_result[idx]))]
    dominant_freq = freqs[dominant_freq_idx]
    
    return f"{dominant_freq * 60:.1f} BPM"

def bandpass_filter_rppg(data, sample_rate, lowcut=0.7, highcut=2.5, order=4):
    """Bandpass filter sinyal rPPG."""
    if len(data) < int(2 * sample_rate * (order + 1)) or sample_rate <=0: 
        return np.array(data)
    
    nyquist = 0.5 * sample_rate
    if not (0 < lowcut < nyquist and 0 < highcut < nyquist and lowcut < highcut):
        return np.array(data)

    b, a = butter(order, [lowcut, highcut], btype='band', fs=sample_rate)
    return filtfilt(b, a, data)

def hitung_detak_jantung(raw_rgb_signal, fps):
    """
    Menghitung detak jantung dari sinyal RGB mentah.
    Langkah: RGB -> Detrend -> POS -> Bandpass Filter -> Cari Puncak.
    """
    if raw_rgb_signal.shape[1] < int(2 * fps) or fps <= 0:
        return "N/A (Kurang data atau FPS tidak valid)"

    detrended_rgb_signal = detrend_signal(raw_rgb_signal, type='linear')
    pulse_signal = cpu_POS(detrended_rgb_signal, fps)
    
    if len(pulse_signal) < int(2 * fps):
        return "N/A (Sinyal POS tidak cukup)"

    filtered_pulse = bandpass_filter_rppg(pulse_signal, fps)
    
    if len(filtered_pulse) < int(fps / 2.5) +1:
         return "N/A (Sinyal terfilter tidak cukup)"

    if np.std(filtered_pulse) > 1e-6:
        normalized_pulse = (filtered_pulse - np.mean(filtered_pulse)) / np.std(filtered_pulse)
        min_height_normalized = 0.5
        peaks, _ = find_peaks(normalized_pulse, height=min_height_normalized, distance=fps / 2.5)
    else:
        peaks = []

    durasi_detik_efektif = len(filtered_pulse) / fps
    if durasi_detik_efektif == 0: 
        return "N/A (Durasi efektif nol)"
    
    if len(peaks) > 1:
        bpm = (len(peaks) -1) / ((peaks[-1] - peaks[0]) / fps) * 60
    elif len(peaks) == 1 and durasi_detik_efektif > 0:
        bpm = 1 * (60 / durasi_detik_efektif)
    else:
        bpm = 0.0
        
    return f"{bpm:.1f} BPM"