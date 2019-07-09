import numpy as np
from emo_utils import *
import emoji
import numpy as np
import json
import os
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

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


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1 # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0] # define dimensionality of your GloVe word vectors (= 50)
    
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


def Emojify(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = Input(input_shape, dtype='int32')
    
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    embeddings = embedding_layer(sentence_indices)   
    
    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(5)(X)
    X = Activation("softmax")(X)
    
    model = Model(inputs=sentence_indices, outputs=X)
        
    return model


# load raw datasets
X_train, Y_train = read_csv('../data/train_emoji.csv')
X_test, Y_test = read_csv('../data/tesss.csv')

# emoji mapping test
index = 3
print("emojy mapping test for index {}".format(index))
print("sentence : " + X_train[index])
print("emoji : " + label_to_emoji(Y_train[index]))

# convert raw data to one_hot
Y_oh_train = convert_to_one_hot(Y_train, C = 5)
Y_oh_test = convert_to_one_hot(Y_test, C = 5)

# load Glove data
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('../data/glove.6B.50d.txt')

# build model
maxLen = len(max(X_train, key=len).split()) # maximum length of input sentence
model = Emojify((maxLen,), word_to_vec_map, word_to_index)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# prepare training data
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = 5)

# train model
model.fit(X_train_indices, Y_train_oh, epochs = 100, batch_size = 16, shuffle=True)


if not os.path.exists("./trained model/"):
    os.makedirs("./trained model/")

# evaluate model
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
with open("./trained model/evaluate.txt", "w") as evaluate:
    evaluate.write("loss : " + str(loss) + "\n" + "Test accuracy : " + str(acc))

# save model
plot_model(model, to_file='./trained model/Emojify model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))

model_json = model.to_json()
with open("./trained model/model.json", "w") as json_file:
    json_file.write(model_json) # model architecture
model.save_weights("./trained model/model.h5") # model weights