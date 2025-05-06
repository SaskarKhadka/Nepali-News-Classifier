import os

class Settings:
    APP_LEVEL_PATH: str = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    ROOT_LEVEL_PATH: str = os.path.split(APP_LEVEL_PATH)[0]

    BERT_TOKENIZER_PATH = os.path.join(ROOT_LEVEL_PATH, 'outputs/bert/BERT_nepali_news_classifier_tokenizer')
    LSTM_TOKENIZER_PATH = os.path.join(ROOT_LEVEL_PATH, 'outputs/lstm/tokenizer.json')

    BERT_MODEL_PATH = os.path.join(ROOT_LEVEL_PATH, 'outputs/bert/BERT_nepali_news_classifier_model')
    LSTM_MODEL_PATH = os.path.join(ROOT_LEVEL_PATH, 'outputs/lstm/LSTM_nepali_news_classifier_model.keras')

    BERT_LABEL_ENCODER = os.path.join(ROOT_LEVEL_PATH, 'outputs/bert/label_encoder.pkl')
    LSTM_LABEL_ENCODER = os.path.join(ROOT_LEVEL_PATH, 'outputs/lstm/label_encoder.pkl')

    LSTM_PARAMETERS_PATH = os.path.join(ROOT_LEVEL_PATH, 'outputs/lstm/parameters.json')


settings = Settings()