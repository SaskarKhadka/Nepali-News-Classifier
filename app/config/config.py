import os

class Settings:
    APP_LEVEL_PATH: str = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    ROOT_LEVEL_PATH: str = os.path.split(APP_LEVEL_PATH)[0]

    BERT_TOKENIZER_PATH = os.path.join(ROOT_LEVEL_PATH, 'outputs/bert/BERT_nepali_news_classifier_tokenizer')
    LSTM_TOKENIZER_PATH = os.path.join(ROOT_LEVEL_PATH, 'outputs/lstm/tokenizer.json')
    ML_TFIDF_PATH = os.path.join(ROOT_LEVEL_PATH, 'outputs/ml/tfidf_vectorizer.pkl')
    MLP_TOKENIZER_PATH = os.path.join(ROOT_LEVEL_PATH, 'outputs/mlp/tokenizer.json')

    BERT_MODEL_PATH = os.path.join(ROOT_LEVEL_PATH, 'outputs/bert/BERT_nepali_news_classifier_model')
    LSTM_MODEL_PATH = os.path.join(ROOT_LEVEL_PATH, 'outputs/lstm/LSTM_nepali_news_classifier_model.keras')
    ML_NAIVE_BAYES_MODEL_PATH = os.path.join(ROOT_LEVEL_PATH, 'outputs/ml/Naive_Bayes_Classifier.joblib')
    ML_GRADIENT_BOOSTING_MODEL_PATH = os.path.join(ROOT_LEVEL_PATH, 'outputs/ml/Gradient_Boosting_Classifier.joblib')
    ML_XG_BOOST_MODEL_PATH = os.path.join(ROOT_LEVEL_PATH, 'outputs/ml/XG_Boost_Classifier.joblib')
    MLP_MODEL_PATH = os.path.join(ROOT_LEVEL_PATH, 'outputs/mlp/MLP_nepali_news_classifier_model.keras')

    BERT_LABEL_ENCODER = os.path.join(ROOT_LEVEL_PATH, 'outputs/bert/label_encoder.pkl')
    LSTM_LABEL_ENCODER = os.path.join(ROOT_LEVEL_PATH, 'outputs/lstm/label_encoder.pkl')
    ML_LABEL_ENCODER = os.path.join(ROOT_LEVEL_PATH, 'outputs/ml/label_encoder.pkl')
    MLP_LABEL_ENCODER =  os.path.join(ROOT_LEVEL_PATH, 'outputs/mlp/label_encoder.pkl')

    LSTM_PARAMETERS_PATH = os.path.join(ROOT_LEVEL_PATH, 'outputs/lstm/parameters.json')
    ML_PARAMETERS_PATH = os.path.join(ROOT_LEVEL_PATH, 'outputs/ml/parameters.json')
    MLP_PARAMETERS_PATH = os.path.join(ROOT_LEVEL_PATH, 'outputs/mlp/parameters.json')


settings = Settings()