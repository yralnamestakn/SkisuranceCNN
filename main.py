from config import INTENTS
from model import train_model, load_model, save_model
from preprocessing import preprocess_messages, tokenize_messages
import numpy as np

save_model(train_model())
model = load_model()

while True:

    sentence = input("Bitte gib deine Nachricht ein:")

    input_data = preprocess_messages(tokenize_messages([sentence]))
    prediction = model.predict(input_data)[0]
    index = np.argmax(prediction)
    print(prediction)
    print("Erkannter Intent für Satz '{}': {} ({:.2%})".format(sentence, INTENTS[index], prediction[index]))
