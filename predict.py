import librosa
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from keras.models import load_model
from keras.utils import to_categorical

model = load_model('D:/Python Projects/lang/Language_Recognition/models/basicmodel.hdf5')
y = np.load('D:/Python Projects/lang/Language_Recognition/dataframes/y_numpy.npy')

le = LabelEncoder()
tc = to_categorical(le.fit_transform(y))

"""
take files and generate MFCCs
"""
def get_predict_mfcc(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)

    except Exception as e:
        print("Error encountered whilst parsing file: ", file_name)
        return None

    return np.array([mfccsscaled])

"""
Take file and predict the language of the file.
"""
def predict(file_name):
    prediction_feature = get_predict_mfcc(file_name)

    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector)
    print("The predicted class is: ", predicted_class[0], "\n")

    predicted_proba_vector = model.predict_proba(prediction_feature)
    predicted_proba = predicted_proba_vector[0]

    for i in range(len(predicted_proba)):
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t :", format(predicted_proba[i], '.3f'))