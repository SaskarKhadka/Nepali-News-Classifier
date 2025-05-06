import tensorflow as tf
import keras
import codecs
from app.config.config import settings
import json
import pickle
from app.config.preprocess import perform_preprocessing

def predict(model, tokenizer, label_encoder, text, input_sequence_length=256):
    text = perform_preprocessing(text)
    sequence = tokenizer.texts_to_sequences([text])[0]

    if len(sequence) > input_sequence_length:
        sequence = sequence[:input_sequence_length]
    else:
        sequence += [0] * (input_sequence_length - len(sequence))
    probs = model.predict(tf.convert_to_tensor([sequence], dtype=tf.int32))[0]
    pred_label_idx = probs.argmax()
    pred = label_encoder.inverse_transform([pred_label_idx])[0]

    return pred

def get_parameters():
    with codecs.open(settings.LSTM_PARAMETERS_PATH, encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_tokenizer():
    with codecs.open(settings.LSTM_TOKENIZER_PATH, encoding='utf-8') as f:
        data = json.load(f)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
    return tokenizer

def get_model():
    return keras.models.load_model(settings.LSTM_MODEL_PATH)

def get_label_encoder():
    with open(settings.LSTM_LABEL_ENCODER, 'rb') as f:
        le = pickle.load(f)
    return le

parameters = get_parameters()
model = get_model()
tokenizer = get_tokenizer()
label_encoder = get_label_encoder()