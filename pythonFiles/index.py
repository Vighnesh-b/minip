import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import BertTokenizer, BertModel
import torch
import requests
from bs4 import BeautifulSoup
import csv
import re
import pandas as pd
from torch import nn
from waitress import serve

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directory to store files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Helper function to remove emojis
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["  
        "\U0001F600-\U0001F64F"  
        "\U0001F300-\U0001F5FF"  
        "\U0001F680-\U0001F6FF"  
        "\U0001F1E0-\U0001F1FF"  
        "\U00002700-\U000027BF"  
        "\U000024C2-\U0001F251"  
        "\U0001F900-\U0001F9FF"  
        "\U0000200D"             
        "\U00002328-\U0000232A"  
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

# Scraping function to get reviews from an Amazon page
def scrape_amazon_reviews(url, output_csv="reviews.csv"):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            logger.error(f"Error: Unable to fetch data, HTTP Status Code: {response.status_code}")
            return

        soup = BeautifulSoup(response.text, "html.parser")
        review_elements = soup.select('[data-hook="review"]')

        # Extract review data
        reviews = []
        for review_element in review_elements:
            title = review_element.select_one('[data-hook="review-title"]')
            rating = review_element.select_one('.review-rating .a-icon-alt')
            text = review_element.select_one('.review-text-content span')
            verified = review_element.select_one('.a-size-mini.a-color-state')

            reviews.append({
                "Review Title": title.text.strip() if title else "N/A",
                "Review Text": remove_emojis(text.text.strip()) if text else "N/A",
                "Star Rating": float(rating.text.split()[0]) if rating else None,
                "Verified Purchase": bool(verified)
            })

        # Save to CSV
        output_path = os.path.join(BASE_DIR, output_csv)
        pd.DataFrame(reviews).to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Data successfully exported to {output_path}")

    except Exception as e:
        logger.error(f"An error occurred during scraping: {e}")

# Multimodal BERT model definition
class MultiModalBERT(nn.Module):
    def __init__(self, bert_model_name, num_numeric_features):
        super(MultiModalBERT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + num_numeric_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, numeric_features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        combined_features = torch.cat((bert_output, numeric_features), dim=1)
        return self.fc(combined_features)

# Load model and tokenizer
model_path = os.path.join(BASE_DIR, 'bert_multimodal_classifier')
model_config = torch.load(os.path.join(model_path, 'model_config.pth'))
model = MultiModalBERT(
    bert_model_name=model_config['bert_model_name'],
    num_numeric_features=model_config['num_numeric_features']
)
model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=device))
model = model.to(device)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Preprocessing function
def preprocess_input(review_title, review_text, rating, verified, tokenizer, max_length=128):
    text = f"{review_title}. {review_text}"
    encoding = tokenizer.encode_plus(
        text,
        max_length=max_length,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )
    numeric_features = torch.tensor([[rating, int(verified)]], dtype=torch.float)
    return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), numeric_features

# Predict review
def predict_review(review_title, review_text, rating, verified, model, tokenizer):
    model.eval()
    input_ids, attention_mask, numeric_features = preprocess_input(
        review_title, review_text, rating, verified, tokenizer
    )
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    numeric_features = numeric_features.to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, numeric_features=numeric_features)
        predicted_class = torch.argmax(logits, dim=1).item()
    return 'real' if predicted_class == 1 else 'fake'

# Routes
@app.route('/')
def home():
    return "Multimodal BERT backend is running!"

@app.route('/scrape_reviews', methods=['POST'])
def scrape_reviews():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({"error": "URL is required"}), 400

    scrape_amazon_reviews(url)
    return jsonify({"message": "Scraping and saving reviews to CSV completed!"})

@app.route('/predict_reviews', methods=['POST'])
def predict_reviews():
    try:
        reviews_df = pd.read_csv(os.path.join(BASE_DIR, "reviews.csv"))
        predictions = []
        for _, row in reviews_df.iterrows():
            prediction = predict_review(
                review_title=row["Review Title"],
                review_text=row["Review Text"],
                rating=row["Star Rating"],
                verified=row["Verified Purchase"],
                model=model,
                tokenizer=tokenizer
            )
            predictions.append(prediction)

        reviews_df["Prediction"] = predictions
        output_csv = os.path.join(BASE_DIR, "reviews_with_predictions.csv")
        reviews_df.to_csv(output_csv, index=False)
        return jsonify({"message": f"Predictions saved to {output_csv}"})

    except Exception as e:
        logger.error(f"Error predicting reviews: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict_single_review', methods=['POST'])
def predict_single_review():
    data = request.json
    review_title = data.get('review_title')
    review_text = data.get('review_text')
    rating = data.get('rating')
    verified = data.get('verified')

    if not all([review_title, review_text, rating is not None, verified is not None]):
        return jsonify({"error": "Missing required fields"}), 400

    prediction = predict_review(review_title, review_text, rating, verified, model, tokenizer)
    return jsonify({"prediction": prediction})

@app.route('/get_csv', methods=['GET'])
def get_csv():
    file = request.args.get('file', '')
    if file not in ['reviews.csv', 'reviews_with_predictions.csv']:
        return jsonify({"error": "Invalid file requested"}), 400

    return send_from_directory(BASE_DIR, file, as_attachment=False)

# Entry point
if __name__ == '__main__':
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5000))
    logger.info(f"Starting server at {HOST}:{PORT}")
    serve(app, host=HOST, port=PORT)


