# Nepali-News-Classifier

In this project I experiment with Attentive Seq2Seq, Encoder Transformer(BERT) and some Machine Learning models to classify Nepali news articles

Check out this <a href="report/Nepali News Classifier.pdf">document</a> to learn about this project

## Setup Instructions
- Run the setup.sh script
    - `setup.sh`

## Startup Instructions
- Inside the project folder
    - Create models and tokenizer folder
        - `mkdir models`
        - `mkdir tokenizer`

    - Add weights of seq2seq and transformer models and the tokenizer to their respective folders
        - Name of seq2seq model = seq2seq.weights.h5
        - Name of Transformer model = transformer.weights.h5
        - Tokenizer must have two files, summarization_50000.model, summarization_50000.vocab
- Run start.sh
    - `./start.sh`