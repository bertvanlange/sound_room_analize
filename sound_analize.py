import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fftpack as fft
import librosa
import librosa.display

# Load the audio file
filename = "clap.wav"  # Replace with your file
y, sr = librosa.load(filename, sr=None)  # Keep original sample rate

# Plot the waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform of Clap Sound")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# Compute FFT
N = len(y)
freqs = np.fft.rfftfreq(N, 1/sr)
fft_vals = np.abs(fft.rfft(y))

plt.figure(figsize=(10, 4))
plt.plot(freqs, fft_vals)
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim([0, 5000])  # Focus on lower frequencies
plt.show()

# Compute autocorrelation
autocorr = np.correlate(y, y, mode='full')
lags = np.arange(-len(y)+1, len(y))

plt.figure(figsize=(10, 4))
plt.plot(lags/sr, autocorr)
plt.title("Autocorrelation of Clap Sound")
plt.xlabel("Time Lag (s)")
plt.ylabel("Correlation")
plt.show()

# Find reflection peaks
peaks, _ = signal.find_peaks(autocorr, height=np.max(autocorr)*0.2, distance=sr*0.01)  # 10ms min separation

# Convert peak lags to time
reflection_times = lags[peaks] / sr
reflection_times = reflection_times[reflection_times > 0]  # Ignore negative lags

print("Estimated reflection times (s):", reflection_times)
