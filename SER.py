import librosa
import soundfile
import os, glob, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix


#Extract features (mfcc, chroma, mel) from a sound file
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

#Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',   #neutralny
  '02':'calm',      #spokojny
  '03':'happy',     #szczesliwy
  '04':'sad',       #smutny
  '05':'angry',     #zly
  '06':'fearful',   #bajazliwy
  '07':'disgust',   #obrzydzony
  '08':'surprised'  #zaskoczony
}

#Emotions to observe
#observed_emotions=['neutral', 'calm','happy', 'sad' , 'angry', 'fearful','disgust','surprised']

observed_emotions=[ 'fearful', 'sad', 'angry', 'surprised']

#Parametr to Extract
ExtractFeaturesMfcc = True
ExtractFeaturesChroma = True
ExtractFeaturesMel = True

#Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("C:\\Users\\Kamil\\Desktop\\Sztuczna Inteligencja - Projekt\\Project\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=ExtractFeaturesMfcc,
                                chroma=ExtractFeaturesChroma,
                                mel=ExtractFeaturesMel)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

#Parametr test size
LoadDataTestSize = 0.25

#Split the dataset
x_train,x_test,y_train,y_test=load_data(test_size=LoadDataTestSize) #parametr test_size

#Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

#Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

#Parametr to Multi Layer Perceptron Classifier
MLPC_hidden_layer_sizes = (300,)
MLPC_batch_size = 256
MLPC_epsilon = 1e-08 #Only for Solver Adam
MLPC_alpha = 0.01
MLPC_learning_rate = 'adaptive'
MLPC_max_iter = 500
MLPC_solver = 'adam'

#Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=MLPC_alpha, batch_size=MLPC_batch_size, epsilon=MLPC_epsilon, hidden_layer_sizes=MLPC_hidden_layer_sizes, learning_rate=MLPC_learning_rate, max_iter=MLPC_max_iter) #parametr

#Train the model
model.fit(x_train,y_train)

# save the model to disk
filename = 'finalized_model' + str(licznik) + '.sav'
pickle.dump(model, open(filename, 'wb'))

#Predict for the test set
y_pred=model.predict(x_test)

#Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))

# Plot Confusion Matrix and Save them
plot_confusion_matrix(model,x_test,y_test,cmap=plt.cm.Blues,normalize='true').confusion_matrix
plt.savefig('4 Emotions - Test n.1' + str(licznik) + "- " + str(round(accuracy,2)) + ".png")
