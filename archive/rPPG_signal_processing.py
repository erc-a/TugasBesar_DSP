import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal
import os

def cpu_POS(signal, **kargs):
    """
    POS method on CPU using Numpy.

    The dictionary parameters are: {'fps':float}.

    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
    """
    # Run the pos algorithm on the RGB color signal c with sliding window length wlen
    # Recommended value for wlen is 32 for a 20 fps camera (1.6 s)
    eps = 10**-9
    X = signal
    e, c, f = X.shape            # e = #estimators, c = 3 rgb ch., f = #frames
    w = int(1.6 * kargs['fps'])   # window length

    # stack e times fixed mat P
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)

    # Initialize (1)
    H = np.zeros((e, f))
    for n in np.arange(w, f):
        # Start index of sliding window (4)
        m = n - w + 1
        # Temporal normalization (5)
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (np.mean(Cn, axis=2)+eps)
        M = np.expand_dims(M, axis=2)  # shape [e, c, w]
        Cn = np.multiply(M, Cn)

        # Projection (6)
        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1)    # remove 3-th dim

        # Tuning (7)
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
        # Overlap-adding (8)
        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)

    return H

def bandpass_filter(signal_in, lowcut, highcut, fs, order=5):
    """
    Bandpass filter the signal using a Butterworth filter.

    Parameters:
    - signal_in: The input signal to be filtered.
    - lowcut: The lower cutoff frequency.
    - highcut: The upper cutoff frequency.
    - fs: The sampling frequency of the signal.
    - order: The order of the filter.

    Returns:
    - filtered_signal: The bandpass filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, signal_in)
    return filtered_signal

def main():
    # Load the mean rgb value for POS rppg algorithm
    rgb_path = os.path.join('rppg_data', 'rppg_data_20250523_154558.csv')
    rgb_df = pd.read_csv(rgb_path)
    rgb_signal = rgb_df[['R', 'G', 'B']].values.T
    fps = 30

    # Store RGB means for plotting
    mean_r = rgb_signal[0, :]
    mean_g = rgb_signal[1, :]
    mean_b = rgb_signal[2, :]

    # Reshape for POS algorithm
    rgb_signal = rgb_signal.reshape(1, 3, -1)  # Reshape to (1, 3, f)
    
    # Run the POS algorithm
    pos_signal = cpu_POS(rgb_signal, fps=fps)
    print(f"Shape of pos_signal: {pos_signal.shape}")
    
    # Reshape pos_signal for easier handling
    pos_signal = pos_signal.flatten()
    
    # Apply bandpass filter
    lowcut = 0.8  # Hz
    highcut = 2.5  # Hz
    filtered_pos = bandpass_filter(pos_signal, lowcut, highcut, fps)
    
    # Find peaks in the filtered signal
    peaks, _ = signal.find_peaks(filtered_pos, height=0.01, distance=fps/2)
    
    # Calculate heart rate
    peak_count = len(peaks)
    estimated_hr = peak_count * (60 / (len(filtered_pos) / fps))
    
    print(f"Number of peaks detected: {peak_count}")
    print(f"Estimated heart rate: {estimated_hr:.1f} BPM")
    
    # Convert peak indices to time values
    peak_times = peaks / fps
    peak_values = filtered_pos[peaks]
    
    # Create time array
    time = np.arange(len(pos_signal)) / fps
    
    # Create figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Plot 1: Mean RGB signals
    axs[0].plot(time, mean_r, 'r-', label='Red', alpha=0.8)
    axs[0].plot(time, mean_g, 'g-', label='Green', alpha=0.8)
    axs[0].plot(time, mean_b, 'b-', label='Blue', alpha=0.8)
    axs[0].set_title('RGB Channel Mean Values')
    axs[0].set_ylabel('Pixel Value')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: POS signal
    axs[1].plot(time, pos_signal, 'b-', label='POS Signal')
    axs[1].set_title('POS rPPG Signal')
    axs[1].set_ylabel('Amplitude')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, alpha=0.3)
    
    # Plot 3: Filtered POS signal with peaks
    axs[2].plot(time, filtered_pos, 'g-', label=f'Filtered POS ({lowcut:.1f}-{highcut:.1f} Hz)')
    axs[2].plot(peak_times, peak_values, 'bo', markersize=6, label=f'Detected Peaks ({peak_count})')
    axs[2].set_title(f'Bandpass Filtered POS Signal - {peak_count} peaks detected ({estimated_hr:.1f} BPM)')
    axs[2].set_xlabel('Time (seconds)')
    axs[2].set_ylabel('Amplitude')
    axs[2].legend(loc='upper right')
    axs[2].grid(True, alpha=0.3)
    
    # Add common title and adjust layout
    plt.suptitle('RGB, POS and Filtered POS Signals with Peak Detection', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

if __name__ == "__main__":
    main()