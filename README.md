# AURA — Automated User Review Analyzer

**AURA** is a compact deep-learning project for sentiment analysis of user reviews. It fine-tunes a BERT model to classify text as **Positive**, **Negative** or **Neutral**. The repository contains training, inference and a simple Flask API for serving predictions.

## Key features

* **High accuracy** using fine-tuned BERT.
* **Three-class classification**: positive / negative / neutral.
* **Modular & scalable**: organised for easy adaptation to new datasets or models.
* **API included**: lightweight Flask app (`app.py`) for serving predictions.

## Repository layout

```
AURA-Sentiment-Analyzer/
│
├── data/
├── saved_models/
├── notebooks/
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── engine.py
│   ├── model.py
│   └── train.py
│
├── app.py
├── predict.py
└── README.md
```

## Dataset
I have utilised Amazon's [Fine Food Reviews dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) from Kaggle. Consisting of 560K rows, this dataset was both hard to compile and exceedingly large to upload GitHub. Therefore, I could not upload this dataset and inserted the link for you.

## Installation

1. Clone the repo and enter the folder:

```bash
git clone https://github.com/anormalguy96/AURA_Sentiment_Analysis.git
cd AURA_Sentiment_Analysis
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Add the dataset: download the *Amazon Fine Food Reviews* dataset from Kaggle, extract `Reviews.csv` and place it in `data/`.

## Usage

Train the model:

```bash
python src/train.py
```

This preprocesses the data, trains the model and saves the best checkpoint to `saved_models/`.

Predict from the command line:

```bash
python predict.py --text "This product was absolutely fantastic; I would recommend it."
```

Run the web app:

```bash
python app.py
```

Then you can open `http://127.0.0.1:5000` to interact with the model.

## License
This project is licensed under [MIT License](#License).
