import pickle
import string
import codecs
import tensorflow as tf
import string
import re
from nltk.corpus import stopwords
import json
import streamlit as st
import numpy as np

nep_stopwrods = stopwords.words("nepali")


def get_constants():
    """
    Returns the CONSTANTS json file which consists the values of parameters used in the model
    """
    with codecs.open("../outputs/model 2/constants.json", encoding="utf-8") as const:
        CONSTANTS = json.load(const)
    return CONSTANTS


def get_tokenizer():
    """
    Returns the tokenizer used in converting the text to sequence of numbers
    """
    with codecs.open("../outputs/model 2/tokenizer.json", encoding="utf-8") as f:
        data = json.load(f)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
    return tokenizer


def get_cotegories():
    """
    Returns the json file which consits of numbers mapped to different catgeories
    """
    with codecs.open("../outputs/model 2/categories.json", encoding="utf-8") as f:
        data = json.load(f)
    return data


def remove_emojis_english_and_numbers(data):
    """
    Removes emojis, non-nepali texts and numbers from the given text
    """
    # Removes emoji from given data
    emoj = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        re.UNICODE,
    )
    res = re.sub(emoj, "", data)
    res = re.sub("[0-9]+", "", res)
    return re.sub("[a-zA-Z]", "", res)


def preprocess_text(data):
    """
    Cleans the given textual data
    Removes special characters, english texts, numbers, and stopwords
    """
    if type(data) == float:
        return data
    data = (
        data.replace("-", " ")
        .replace("—", " ")
        .replace("‘", " ")
        .replace("’", " ")
        .replace("।", " ")
        .replace("–", " ")
        .replace("“", " ")
        .replace("”", " ")
        .replace("\n", " ")
        .replace("–", " ")
        .replace("ः", " ")
    )
    no_extra_spaces = " ".join(data.split())
    no_emoji_english_numbers = remove_emojis_english_and_numbers(no_extra_spaces)
    no_punc = "".join(
        [char for char in no_emoji_english_numbers if char not in (string.punctuation)]
    )
    extra = " ".join(no_punc.split())
    no_num = "".join([char for char in extra if char not in "०१२३४५६७८९"])
    no_stopwords = [
        word.strip() for word in no_num.split() if word.strip() not in nep_stopwrods
    ]
    return " ".join(no_stopwords)


# Load the classifier model, tokenizer, categories and constants file
model = tf.keras.models.load_model("../outputs/model 2/NepaliNewsClassifier")
tokenizer = get_tokenizer()
categories = get_cotegories()
constants = get_constants()

# Create title for Web App
st.title("Nepali News Classifier")

# Create text area to input the news
input_news = st.text_area("Enter News: ")

if st.button("Predict"):

    # Clean the input news
    cleaned_news = preprocess_text(input_news)

    # If the input news length is more than the length used to train the model, trim it
    if len(cleaned_news.split()) > constants["max_news_length"]:
        cleaned_news = " ".join(cleaned_news.split()[: constants["max_news_length"]])

    # Convert text top sequences using the tokenizer
    cleaned_seq = tokenizer.texts_to_sequences([cleaned_news])

    # Pad the number sequences if the length is less than the length of sequence used to train the model
    cleaned_pad_seq = tf.keras.preprocessing.sequence.pad_sequences(
        cleaned_seq, maxlen=constants["max_news_length"], padding="post"
    )

    # Finally, use the model to generate a prediction
    # The output of the model is probability distribution over different catgeoies
    # Find the argmax of the distribution and use it to access the news catgeory from the categories.json file
    pred = np.argmax(model.predict([cleaned_pad_seq])[0])
    res = categories[str(pred)]

    st.header(f"{res}")
