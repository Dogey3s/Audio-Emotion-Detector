import librosa
import numpy as np
import matplotlib.pyplot as plt\

result = np.array([])

y ,sr = librosa.load("P:/archive/Actor_24/03-01-02-02-02-01-24.wav")
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
mfccs_processed = np.mean(mfccs.T, axis=0)
result = np.hstack((result, mfccs_processed))

print("MFCCs:",mfccs)
print("MFCCs Processed:",mfccs_processed)
print("Result:",result)

plt.plot(y)
plt.xlabel('Time(Samples)')
plt.ylabel('Amplitude')
plt.title("Waveform")
plt.show()