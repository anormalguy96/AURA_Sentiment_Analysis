# app.py

from flask import Flask, request, jsonify, render_template_string
import torch
from transformers import BertTokenizer

from src import config
from src.model import AURA

app = Flask(__name__)
device = torch.device(config.DEVICE)

model = AURA(n_classes=len(config.CLASS_NAMES))
model.load_state_dict(torch.load(config.SAVED_MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained(config.TOKENIZER)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AURA Sentiment Analyzer</title>
    <style>
        body { font-family: 'Helvetica Neue', Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #f4f4f9; margin: 0; }
        .container { text-align: center; background-color: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        textarea { width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ddd; font-size: 16px; margin-top: 20px; }
        input[type="submit"] { background-color: #007BFF; color: white; padding: 10px 20px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; margin-top: 20px; }
        input[type="submit"]:hover { background-color: #0056b3; }
        .result { margin-top: 20px; font-size: 20px; color: #333; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AURA: Sentiment Analyzer</h1>
        <form action="/" method="post">
            <textarea name="text" rows="5" cols="50" placeholder="Enter review text here...">{{text}}</textarea><br>
            <input type="submit" value="Analyze Sentiment">
        </form>
        {% if sentiment %}
        <div class="result">
            <p>Predicted Sentiment: <strong>{{sentiment}}</strong></p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = ""
    text = ""
    if request.method == 'POST':
        text = request.form['text']
        
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
            outputs = model(ids=input_ids, mask=attention_mask, token_type_ids=token_type_ids)
            _, prediction_idx = torch.max(outputs, dim=1)
            
        sentiment = config.CLASS_NAMES[prediction_idx.item()]

    return render_template_string(HTML_TEMPLATE, sentiment=sentiment, text=text)

if __name__ == '__main__':
    app.run(debug=True)