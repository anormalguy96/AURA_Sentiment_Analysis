# predict.py

import torch
import argparse
from transformers import BertTokenizer

from src import config
from src.model import AURA

def predict_sentiment(text: str):
    device = torch.device(config.DEVICE)
    
    model = AURA(n_classes=len(config.CLASS_NAMES))
    model.load_state_dict(torch.load(config.SAVED_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(config.TOKENIZER)

    encoded_review = tokenizer.encode_plus(
        text,
        max_length=config.MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)
    token_type_ids = encoded_review['token_type_ids'].to(device)

    with torch.no_grad():
        outputs = model(
            ids=input_ids,
            mask=attention_mask,
            token_type_ids=token_type_ids
        )
        _, prediction_idx = torch.max(outputs, dim=1)
        
    predicted_class = config.CLASS_NAMES[prediction_idx.item()]
    print(f'Review text: "{text}"')
    print(f'Predicted sentiment: {predicted_class}')
    return predicted_class

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict sentiment of a review.")
    parser.add_argument('--text', type=str, required=True, help='The review text to analyze.')
    args = parser.parse_args()
    predict_sentiment(args.text)