import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio file
y, sr = librosa.load('P:/archive/Actor_02/03-01-01-01-01-01-02.wav')

D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

# Display the spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.tight_layout()
plt.show()