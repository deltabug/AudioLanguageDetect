from datetime import datetime
import librosa
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

"""
defining path for training and test files with example filename
"""
train_path = 'D:/Python Projects/lang/Language_Recognition/spoken-language-identification/spoken-language-identification/train/train/'
test_path = 'D:/Python Projects/lang/Language_Recognition/spoken-language-identification/spoken-language-identification/test/test/'
filename = 'fa_18569902.wav'

"""
function to extract Mel-frequency cepstral coefficients to represent audio file
"""
def get_mfcc(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)

    except Exception as e:
        print("Error encountered whilst parsing file: ", filename)
        return None

    return mfccsscaled

print("Beginning feature extraction from files")

"""
loop through training files, extract the MFCC, the language from the filename and create an array known as features.
This array is then converted to a pandas dataframe and saved.
"""
features = []
for filename in os.listdir(train_path):
    label=(filename[:2])

    data = get_mfcc((train_path + filename))
    features.append([data, label])

df = pd.DataFrame(features, columns=['features', 'label'])
df.to_pickle('D:/Python Projects/lang/Language_Recognition/dataframes/lang_rec_dataframe.pkl')

print('Finished feature extraction from ', len(df), ' files')

"""
take dataframe, seperate into features and label(language), generate numpy array and store as x and y. These
are then saved
"""
x = np.array(df.features.tolist())
np.save('D:/Python Projects/lang/Language_Recognition/dataframes/x_numpy.npy', x)
y = np.array(df.label.tolist())
np.save('D:/Python Projects/lang/Language_Recognition/dataframes/y_numpy.npy', y)

"""
encode target labels with values between 0 and 1. 
Then fit label encoder and return encoded lables into to_categorical that converts into binary class matrix
"""
le = LabelEncoder()
tc = to_categorical(le.fit_transform(y))

"""
split the files into train and test sets
"""
print("Beginning train test split")
x_train, x_test, y_train, y_test = train_test_split(x, tc, test_size=0.2, random_state=42)

"""
Prepare model layers for training.
"""
num_labels = tc.shape[1]

model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')