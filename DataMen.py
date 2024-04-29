import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio file
y, sr = librosa.load('P:/archive/Actor_01/03-01-01-01-01-01-01.wav')

# Extract Mel-frequency cepstral coefficients (MFCCs)
mfccs = librosa.feature.mfcc(y=y, sr=sr)

# Extract Chroma feature
chroma_shift = librosa.feature.chroma_stft(y=y,sr=sr)

# Extract Spectral Constrast
spec_contrast = librosa.feature.spectral_contrast(y=y,sr=sr)

# Extract Zero Crossing Rate
zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)

# Extract Energy
rmse = librosa.feature.rms(y=y)
print("MFCCs:",mfccs)
print("Chroma Shift",chroma_shift)
print("Spectral Contrast:",spec_contrast)
print("Zero Crossing Rate:",zero_crossing_rate)
print("RMSE:",rmse)
librosa.display.specshow(mfccs,x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

