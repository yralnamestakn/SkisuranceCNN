import os
import json
import random
import nltk
import numpy as np
import pickle

from gensim.models import KeyedVectors

from config import INTENTS

os.chdir("C:/Users/Vincent/PycharmProjects/SkisuranceCNN")
model = KeyedVectors.load_word2vec_format('german.model', binary=True)


def replace_special_chars(word):
    #python dictionary zum Ersetzen von Umlauten
    char_map = {"ü": "ue", "ö": "oe", "ä": "ae", "Ü": "Ue", "Ö": "Oe", "Ä": "Ae", "ß": "ss"}
    for special_char in char_map.keys():
        word = word.replace(special_char, char_map[special_char])
    return word


def load_data_from_file(json_filename):
    with open(json_filename, encoding="utf-8-sig", mode="r") as file:
        #json (file) zu python object (raw_samples) umwandeln
        raw_samples = json.loads("".join(file.readlines()))
        #shuffle Trainingsdaten
        random.seed = 12
        random.shuffle(raw_samples)

    x = []
    y = []
    #Liste aus Nachrichten (x) und aus Intents/ Labels (y) erstellen
    for sample in raw_samples:
        x.append(sample["message"])  # message ist dictionary
        y.append(sample["intent"])
    return x, y

def tokenize_messages(messages):
    ret = []
    for message in messages:
        ret.append(nltk.word_tokenize(message))
    return ret

def preprocess_labels(labels, possible_labels):
    vec_y = []

    for intents in labels:
        #Ausgabevektoren in Trainingsdaten mit Länge der Anzahl der Labels erstellen und mit 0en füllen
        intent = np.zeros(len(possible_labels))  #
        #Ausgabevektoren in Trainingsdaten labeln (1 an der Stelle, an der das Label in config steht)
        intent[possible_labels.index(intents)] = 1.
        vec_y.append(intent)
    vec_y = np.array(vec_y)
    return vec_y


def get_possible_labels():
    return INTENTS


# erstelle matrix aus Nachrichten*Wörter*Vektor
def preprocess_messages(messages):
    vec_x = []
    for message in messages:
        message_vec = np.zeros(shape=(100, model.vector_size))
        # Alternative als list comprehension: vec_x = [[model[word] for word in message] for message in x]
        for index, word in enumerate(message):
            word = replace_special_chars(word)
            try:
                word_vec = model[word]
            except KeyError:
                word_vec = np.zeros(model.vector_size)
                print("Wort '{}' nicht erkannt".format(word))
            message_vec[index] = np.array(word_vec)
        vec_x.append(np.array(message_vec))
    vec_x = np.array(vec_x)
    return vec_x


def save_data(vec_x, vec_y, output_filename):
    with open(output_filename, "wb") as file:
        pickle.dump((vec_x, vec_y), file)


def preprocess_file(json_filename, output_filename):
    messages, labels = load_data_from_file(json_filename)
    messages = tokenize_messages(messages)
    labels = preprocess_labels(labels,get_possible_labels())
    messages = preprocess_messages(messages)
    save_data(messages, labels, output_filename)

preprocess_file("Intents.json.txt", "intents.pickle")
preprocess_file("Validation.json.txt", "validation_intents.pickle")
