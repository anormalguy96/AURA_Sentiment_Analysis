import torch

MODEL_NAME = 'bert-base-uncased'
TOKENIZER = 'bert-base-uncased'

DATA_PATH = "data/Reviews.csv"
SAVED_MODEL_PATH = "saved_models/aura_model.bin"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 256
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

SENTIMENT_MAP = {
    1: 0,
    2: 0,
    3: 1,
    4: 2,
    5: 2
}
CLASS_NAMES = ['Negative', 'Neutral', 'Positive']