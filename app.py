from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForSequenceClassification
import json
import os

with open("goemotions_labels.json", "r") as f:
    GOEMOTIONS_LABELS = json.load(f)


# Load model and tokenizer
# reference: https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/
# reference: https://huggingface.co/monologg/bert-base-cased-goemotions-original/tree/main
model_name = "monologg/bert-base-cased-goemotions-original"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.eval()

# GoEmotions label list (get this from the goemotions_labels.json file that contains the labels)

with open("goemotions_labels.json", "r") as f:
    GOEMOTIONS_LABELS = json.load(f)

app = Flask(__name__)
CORS(app)


# define a route for the resource
# classify_text() function:
# tokenizes the input text, 
# runs it through the model, 
# applies sigmoid to get probabilities, 
# and filters emotions with probability > 0.3

@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.json
    text = data.get("text", "")

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Run through the model
    # 
    with torch.no_grad():
        outputs = model(**inputs)

    # apply sigmoid to get probabilities (since it's multi-label)
    probs = torch.sigmoid(outputs.logits).squeeze()

    # Filter emotions with probability > 0.3
    # this means that the model will only retrieve emotions with confidence of at least 30%
    threshold = 0.3
    result = [GOEMOTIONS_LABELS[i] for i, prob in enumerate(probs) if prob > threshold]

    return jsonify({ "tags": result })

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

