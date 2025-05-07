import streamlit as st
from app.config import bert_config as bert, lstm_config as lstm, ml_config as ml
from app.config.preprocess import perform_preprocessing
import asyncio

st.set_page_config(
    page_title="Summarization",
    layout="wide",
    initial_sidebar_state="collapsed",
)


async def main():
    st.title("Nepali News Classifier")

    st.text_area('Enter Nepali News:', height=200, key='news_input')
    if st.button('Predict Category'):
        news = st.session_state.news_input
        if news.strip() == '':
            st.error("Please enter some news", icon='🚨')
        
        elif len(news.split()) <= 10:
            st.error("News seems very short, please make sure it has atleast 30 words", icon='🚨')
        
        else:
            pred_bert = bert.predict(bert.model, bert.tokenizer, bert.label_encoder, st.session_state.news_input, bert.parameters['NEWS_SEQUENCE_LENGTH'])
            st.write(f"###### **BERT:**&nbsp;&nbsp;&nbsp;{pred_bert}")
            pred_lstm = lstm.predict(lstm.model, lstm.tokenizer, lstm.label_encoder, st.session_state.news_input, lstm.parameters['MAX_NEWS_LENGTH'])
            st.write(f"###### **Bi-LSTM:**&nbsp;&nbsp;&nbsp;{pred_lstm}")
            pred_nb = ml.predict(ml.naive_bayes_model, ml.tfidf, ml.label_encoder, st.session_state.news_input, lstm.parameters['MAX_NEWS_LENGTH'])
            st.write(f"###### **Naive Bayes:**&nbsp;&nbsp;&nbsp;{pred_nb}")
            pred_gb = ml.predict(ml.gradient_boosting_model, ml.tfidf, ml.label_encoder, st.session_state.news_input, lstm.parameters['MAX_NEWS_LENGTH'])
            st.write(f"###### **Gradient Boost:**&nbsp;&nbsp;&nbsp;{pred_gb}")
            pred_xgb = ml.predict(ml.gradient_boosting_model, ml.tfidf, ml.label_encoder, st.session_state.news_input, lstm.parameters['MAX_NEWS_LENGTH'])
            st.write(f"###### **XG Boost:**&nbsp;&nbsp;&nbsp;{pred_xgb}")

if __name__ == "__main__":
    asyncio.run(main())