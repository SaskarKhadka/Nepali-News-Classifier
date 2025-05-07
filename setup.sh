#!/bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11.11
git clone https://github.com/SaskarKhadka/Nepali-News-Classifier.git
cd Nepali-News-Classifier
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt