from transformers import BertTokenizer, BertForSequenceClassification
from app.config.config import settings
import pickle
from app.config.preprocess import perform_preprocessing
import torch

torch.classes.__path__ = []

parameters = {
    'NEWS_SEQUENCE_LENGTH': 256,
    'TRAIN_BATCH_SIZE': 32,
    'EVAL_BATCH_SIZE': 32,
    'EPOCHS': 3,
    'LEARNING_RATE': 5e-5,
    'WARMUP_STEPS': 1_000,
    'GRADIENT_ACCUMULATION_STEPS': 4,
    'L2_REG': 0.01,
}

def predict(model, tokenizer, label_encoder, text, input_sequence_length=256):
    text = perform_preprocessing(text)
    inputs = tokenizer(text, padding=True, truncation=True, max_length=input_sequence_length, return_tensors='pt')
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    pred_label_idx = probs.argmax()
    pred = label_encoder.inverse_transform([pred_label_idx.cpu().numpy()])[0]

    return pred

def get_model():
    return BertForSequenceClassification.from_pretrained(settings.BERT_MODEL_PATH)

def get_tokenizer():
    return BertTokenizer.from_pretrained(settings.BERT_TOKENIZER_PATH)

def get_label_encoder():
    with open(settings.BERT_LABEL_ENCODER, 'rb') as f:
        le = pickle.load(f)
    return le

model = get_model()
tokenizer = get_tokenizer()
label_encoder = get_label_encoder()