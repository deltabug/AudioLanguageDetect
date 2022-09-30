import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


x = np.load('D:/Python Projects/lang/Language_Recognition/dataframes/x_numpy.npy')
y = np.load('D:/Python Projects/lang/Language_Recognition/dataframes/y_numpy.npy')

le = LabelEncoder()
tc = to_categorical(le.fit_transform(y))

x_train, x_test, y_train, y_test = train_test_split(x, tc, test_size=0.2, random_state=42)

"""
Test the accuracy of the model trained
"""
def test_model(modelpath):
    loadedmodel = load_model(modelpath)
    score = loadedmodel.evaluate(x_train, y_train, verbose=0)
    print("Training accuracy: ", score[1])

    scoretest = loadedmodel.evaluate(x_test, y_test, verbose=0)
    print("Testing accuracy: ", scoretest[1])