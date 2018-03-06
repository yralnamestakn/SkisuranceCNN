import pickle

import os

from datetime import datetime
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import Dense, Flatten, Conv1D

from config import NUM_OF_INTENTS, SENTENCE_SHAPE
from preprocessing import preprocess_messages


def build_model(sentence_shape, num_of_intents):
    input_layer = Input(shape=sentence_shape)
    hidden_layer = Conv1D(filters=16, kernel_size=5)(input_layer)
    hidden_layer = Conv1D(filters=16, kernel_size=3)(hidden_layer)
    hidden_layer = Conv1D(filters=16, kernel_size=1)(hidden_layer)
    hidden_layer = Flatten()(hidden_layer)
    output_layer = Dense(num_of_intents, activation="softmax")(hidden_layer)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer="sgd", loss="categorical_crossentropy")  # literaturrecherche
    return model


def train_model():
    with open("intents.pickle", mode="rb") as file:
        vec_x, vec_y = pickle.load(file)
    with open("validation_intents.pickle", mode="rb") as file:
        val_vec_x, val_vec_y = pickle.load(file)
    model = build_model(SENTENCE_SHAPE, NUM_OF_INTENTS)
    dir_name = os.path.join('logs', '{:%Y-%m-%d-%H-%M}'.format(datetime.now()))
    tensor_board = TensorBoard(log_dir=dir_name)
    model.fit(vec_x, vec_y, epochs=5000, batch_size=16, validation_data=(val_vec_x, val_vec_y), callbacks=[tensor_board])
    return model

def save_model(model:Model):
    model.save_weights("model.weights")

def load_model():
    model = build_model(SENTENCE_SHAPE, NUM_OF_INTENTS)
    model.load_weights("model.weights")
    return model
