from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
import torch
import requests
from bs4 import BeautifulSoup
import csv
import re
import pandas as pd
import os
from torch import nn

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper function to remove emojis
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["  # Emoji ranges
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0000200D"             # zero-width joiner
        "\U00002328-\U0000232A"  # misc technical
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

# Scraping function to get reviews from an Amazon page
def scrape_amazon_reviews(url, output_csv="reviews.csv"):
    try:
        # Send a GET request to the provided URL
        response = requests.get(url)

        # Check for a successful response
        if response.status_code != 200:
            print(f"Error: Unable to fetch data, HTTP Status Code: {response.status_code}")
            return

        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all review elements
        review_elements = soup.select('[data-hook="review"]')

        # Initialize lists to store extracted data
        review_titles_list = []
        review_texts_list = []
        review_ratings_list = []
        verified_purchase_list = []

        # Loop through each review element
        for review_element in review_elements:
            # Extract the review title
            title_element = review_element.select_one('[data-hook="review-title"]')
            review_titles_list.append(title_element.text.strip() if title_element else "N/A")

            # Extract the rating
            rating_element = review_element.select_one('.review-rating .a-icon-alt')
            if rating_element:
                rating_text = rating_element.text.strip().split()[0]
                review_ratings_list.append(float(rating_text))  # Convert to float
            else:
                review_ratings_list.append(None)

            # Extract the review text
            review_text_element = review_element.select_one('.review-text-content span')
            if review_text_element:
                review_text = review_text_element.text.strip()
                review_text = remove_emojis(review_text)  # Remove emojis
                review_texts_list.append(review_text)
            else:
                review_texts_list.append("N/A")

            # Extract "Verified Purchase" status
            verified_purchase_element = review_element.select_one('.a-size-mini.a-color-state')
            verified_purchase_list.append(True if verified_purchase_element else False)

        # Write the data to a CSV file
        with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Review Title", "Review Text", "Star Rating", "Verified Purchase"])

            for i in range(len(review_titles_list)):
                writer.writerow([review_titles_list[i], review_texts_list[i], review_ratings_list[i], verified_purchase_list[i]])

        print(f"Data successfully exported to {output_csv}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Define the multimodal model class
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
model_save_path = './bert_multimodal_classifier'
model_config = torch.load(os.path.join(model_save_path, 'model_config.pth'))
model = MultiModalBERT(
    bert_model_name=model_config['bert_model_name'],
    num_numeric_features=model_config['num_numeric_features']
)
model.load_state_dict(torch.load(os.path.join(model_save_path, 'pytorch_model.bin'), map_location=device))
model = model.to(device)
tokenizer = BertTokenizer.from_pretrained(model_save_path)

# Helper function to preprocess inputs
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
    return (
        encoding['input_ids'].squeeze(0),
        encoding['attention_mask'].squeeze(0),
        numeric_features
    )

# Prediction function
def predict_review(review_title, review_text, rating, verified, model, tokenizer):
    model.eval()
    input_ids, attention_mask, numeric_features = preprocess_input(
        review_title, review_text, rating, verified, tokenizer
    )
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    numeric_features = numeric_features.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, numeric_features=numeric_features)
        logits = outputs
        predicted_class = torch.argmax(logits, dim=1).item()
    return 'real' if predicted_class == 1 else 'fake'

@app.route('/')
def home():
    return "Multimodal BERT backend with scraping is running!"

@app.route('/scrape_reviews', methods=['POST'])
def scrape_reviews():
    try:
        data = request.json
        url = data.get('url', '')
        if not url:
            return jsonify({"error": "URL is required"}), 400

        scrape_amazon_reviews(url, "reviews.csv")
        return jsonify({"message": "Scraping and saving reviews to CSV completed!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_reviews', methods=['POST'])
def predict_reviews():
    try:
        reviews_df = pd.read_csv("reviews.csv")
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
        output_csv = "reviews_with_predictions.csv"
        reviews_df.to_csv(output_csv, index=False)
        return jsonify({"message": f"Predictions saved to {output_csv}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

    return (
        encoding['input_ids'].squeeze(0),
        encoding['attention_mask'].squeeze(0),
        numeric_features
    )

# API endpoint for single review prediction
@app.route('/predict_single_review', methods=['POST'])
def predict_single_review():
    try:
        data = request.json

        # Validate input
        review_title = data.get('review_title', '')
        review_text = data.get('review_text', '')
        rating = data.get('rating', None)
        verified = data.get('verified', None)

        if not review_title or not review_text or rating is None or verified is None:
            return jsonify({"error": "Missing required fields: 'review_title', 'review_text', 'rating', or 'verified'"}), 400

        # Preprocess the input
        input_ids, attention_mask, numeric_features = preprocess_input(
            review_title, review_text, rating, verified, tokenizer
        )

        # Predict
        model.eval()
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        numeric_features = numeric_features.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, numeric_features=numeric_features)
            predicted_class = torch.argmax(outputs, dim=1).item()

        return jsonify({"prediction": "real" if predicted_class == 1 else "fake"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
