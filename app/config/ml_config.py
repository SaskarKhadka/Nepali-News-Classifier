import joblib, pickle, codecs, json
from app.config.preprocess import perform_preprocessing
from app.config.config import settings

def predict(model, tfidf, label_encoder, text, input_sequence_length=256):
    text = perform_preprocessing(text)
    vectorized = tfidf.transform([text])
    pred_label_idx = model.predict(vectorized)
    pred = label_encoder.inverse_transform(pred_label_idx)[0]
    return pred

def get_parameters():
    with codecs.open(settings.ML_PARAMETERS_PATH, encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_label_encoder():
    with open(settings.ML_LABEL_ENCODER, 'rb') as f:
        le = pickle.load(f)
    return le

def get_naive_bayes_model():
    return joblib.load(settings.ML_NAIVE_BAYES_MODEL_PATH)

def get_gradient_boosting_model():
    return joblib.load(settings.ML_GRADIENT_BOOSTING_MODEL_PATH)

def get_xgboost_model():
    return joblib.load(settings.ML_XG_BOOST_MODEL_PATH)

def get_tfidf_vectorizer():
    with open(settings.ML_TFIDF_PATH, 'rb') as f:
        tfidf = pickle.load(f)
    return tfidf

parameters = get_parameters()
label_encoder = get_label_encoder()
naive_bayes_model = get_naive_bayes_model()
gradient_boosting_model = get_gradient_boosting_model()
xgboost_model = get_xgboost_model()
tfidf = get_tfidf_vectorizer()