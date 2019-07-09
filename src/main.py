import argparse
import numpy as np
from keras.models import Model
from keras.models import model_from_json
from emo_utils import *

parser = argparse.ArgumentParser(description="convert sentece to emojy")

parser.add_argument("-s", "--sentence", type=str, default="not feeling happy",
                    help="Input sentence")

args = parser.parse_args()

with open("./trained model/model.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("./trained model/model.h5", "r")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0] # number of training examples
    
    X_indices = np.zeros((m, max_len))
    
    for i in range(m): # loop over training examples
        sentence_words = X[i].lower().split() # make input sentece as lower case and tokenize
        
        j = 0
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j = j + 1
    
    return X_indices

word_to_index, _, _ = read_glove_vecs('../data/glove.6B.50d.txt')
maxLen = 10

x_test = np.array([args.sentence])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))