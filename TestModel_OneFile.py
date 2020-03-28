import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.metrics import confusion_matrix

#DataFlair - Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

observed_emotions=['neutral', 'happy', 'sad', 'angry', 'surprised']

#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
            result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

# Load MLPC Model
FileModelName = 'finalized_model5.sav'
loaded_model = pickle.load(open(FileModelName, 'rb'))

# Select waw File
file = 'TestFile.wav'
file_name=os.path.basename(file)

#DataFlair - Parametr to Extract
ExtractFeaturesMfcc = True
ExtractFeaturesChroma = True
ExtractFeaturesMel = True

# Get Fetures from selected file
feature=extract_feature(file_name, mfcc=ExtractFeaturesMfcc, chroma=ExtractFeaturesChroma, mel=ExtractFeaturesMel)
feature = feature.reshape(1,-1)

# Predict emotion 
y_pred = loaded_model.predict_proba(feature)

# Show predict emotion
x = 0
for emotion in observed_emotions:
    print(emotion + " = " + str(y_pred[0][x]))
    x += 1

