import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

filename = "clap.wav" 

# Load the audio file
sample_rate, audio_data = wavfile.read(filename)

# Convert stereo to mono if necessary
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)

# Normalize audio
audio_data = audio_data / np.max(np.abs(audio_data))

# Compute the spectrogram using scipy.signal.spectrogram
nperseg = 1024  # You can try adjusting this if your audio is short
frequencies, times, Sxx = spectrogram(audio_data, sample_rate, nperseg=nperseg)

# Avoid log10 of zero by adding a small constant to the spectrogram
Sxx = np.where(Sxx == 0, 1e-10, Sxx)

# Compute Fast Autocorrelation via FFT
N = len(audio_data)
fft_vals = np.fft.fft(audio_data, n=2 * N)  # Zero-pad to avoid circular correlation
power_spectrum = np.abs(fft_vals) ** 2  # Compute power spectrum
autocorr = np.fft.ifft(power_spectrum).real[:N]  # Compute inverse FFT and take real part

# Create time lag axis
lags = np.arange(N) / sample_rate  # Convert sample indices to time

# Convert autocorrelation to dB scale
eps = 1e-6
autocorr_db = 20 * np.log10(np.maximum(autocorr, eps))

speed_of_sound = 343  # m/s

# Convert time to distance (Distance = Time * Speed of Sound)
distance = times * speed_of_sound
distance_lags = lags * speed_of_sound

# Create subplots to display both plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot the spectrogram
ax1.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='auto')
ax1.set_ylabel('Frequency [Hz]')
ax1.set_xlabel('Time [sec]')
ax1.set_title('Spectrogram')
fig.colorbar(ax1.collections[0], ax=ax1, label='Power [dB]')
# fig.colorbar(ax2.collections[0], ax=ax2, label='Power [dB]')

# Plot the autocorrelation
ax2.plot(lags, autocorr_db)
ax2.set_title("Autocorrelation of Clap Sound (via FFT)")
ax2.set_xlabel("Lag Time (s)")
ax2.set_ylabel("Correlation")
ax2.set_ylim(-40, 50)
fig.colorbar(ax1.collections[0], ax=ax2, label='Power [dB]')

# fig.add_axes([0.91, 0.1, 0.03, 0.8])
# Adjust layout and show the plots
plt.tight_layout()

plt.savefig("sound_analysis_plot.png")  # Save with high resolution

plt.show()
