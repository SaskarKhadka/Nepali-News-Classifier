import streamlit as st
from app.config import bert_config as bert, lstm_config as lstm
from app.config.preprocess import perform_preprocessing
import asyncio

st.set_page_config(
    page_title="Summarization",
    layout="wide",
    initial_sidebar_state="collapsed",
)


async def main():
    st.title("Nepali News Classifier")

    st.text_area('Enter Nepali News:', height=250, key='news_input')
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

if __name__ == "__main__":
    asyncio.run(main())



# if st.button("Predict"):

#     # Clean the input news
#     cleaned_news = preprocess_text(input_news)

#     # If the input news length is more than the length used to train the model, trim it
#     if len(cleaned_news.split()) > constants["max_news_length"]:
#         cleaned_news = " ".join(cleaned_news.split()[: constants["max_news_length"]])

#     # Convert text top sequences using the tokenizer
#     cleaned_seq = tokenizer.texts_to_sequences([cleaned_news])

#     # Pad the number sequences if the length is less than the length of sequence used to train the model
#     cleaned_pad_seq = tf.keras.preprocessing.sequence.pad_sequences(
#         cleaned_seq, maxlen=constants["max_news_length"], padding="post"
#     )

#     # Finally, use the model to generate a prediction
#     # The output of the model is probability distribution over different catgeoies
#     # Find the argmax of the distribution and use it to access the news catgeory from the categories.json file
#     pred = np.argmax(model.predict([cleaned_pad_seq])[0])
#     res = categories[str(pred)]

    # st.header(f"{res}")
