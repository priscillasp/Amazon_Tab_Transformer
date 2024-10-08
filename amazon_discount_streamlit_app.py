import streamlit as st
import pickle
import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# Load TabTransformer model
model = load_model("/Users/priscillamorales/Desktop/CodeSpace/Personal_Projects/Amazon_Tab_Transformer/embedded_tabtransformer_model.keras")

# Load the SentenceTransformer model by specifying the path to the folder
sentence_model = SentenceTransformer("/Users/priscillamorales/Desktop/CodeSpace/Personal_Projects/Amazon_Transformer/sentence_transformer_model")

with open("/Users/priscillamorales/Desktop/CodeSpace/Personal_Projects/Amazon_Transformer/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# Define headers to mimic a real browser
ua = UserAgent()
headers = {
    "User-Agent": ua.random,
    "Accept-Language": "en-US,en;q=0.9"
}

# Scrape product data function
def scrape_product_data(url):
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to retrieve page. Status code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the product title
    title = soup.find("span", {"id": "productTitle"})
    if title:
        title = title.get_text(strip=True)
    else:
        title = "Product title not found"

    # Extract the price using the updated class
    price = soup.find("span", {"class": "a-price-whole"})
    if price:
        price_fraction = soup.find("span", {"class": "a-price-fraction"})
        price = price.get_text(strip=True)
        if price_fraction:
            price += price_fraction.get_text(strip=True)
        price = float(price)
    else:
        price = "Price not found"

    # Extract the original list price if available
    list_price = price  # If list price is not available, set it to the same as the price

    # Extract the rating
    rating = soup.find("span", {"class": "a-icon-alt"})
    if rating:
        rating = rating.get_text(strip=True)
        rating = rating.split()[0]  # This will take the first part before "out of 5 stars"
        rating = float(rating)
    else:
        rating = "Rating not found"

    # Extract the rating count
    rating_count = soup.find("span", {"id": "acrCustomerReviewText"})
    if rating_count:
        rating_count = rating_count.get_text(strip=True).replace(",", "").replace(" ratings", "")
        rating_count = int(rating_count)
    else:
        rating_count = "Rating count not found"
    
    product_data = {
        'product_name': title,
        'discounted_price_usd': price,
        'actual_price_usd': list_price,
        'rating': rating,
        'rating_count': rating_count
    }

    return product_data


def predict_discount(input_data, n_samples=100):
    try:
        print(f"Original input data: {input_data}")

        model.trainable = False  # Optional, to ensure the model isn't updating

        # Generate the sentence embedding for the product name using sentence_model
        product_name = input_data.get('product_name', '')
        product_name_embedding = sentence_model.encode([product_name])  # Encode the product name
        print(f"Product name embedding: {product_name_embedding}")

        # Apply log transformation to rating_count
        input_data['rating_count'] = np.log1p(input_data['rating_count'])

        # Ensure the numerical columns are correctly parsed
        numerical_columns = ['discounted_price_usd', 'actual_price_usd', 'rating', 'rating_count']
        numerical_data = []
        for col in numerical_columns:
            if col in input_data:
                try:
                    num_val = float(input_data[col])
                    numerical_data.append(num_val)
                except ValueError:
                    print(f"Invalid data for {col}: {input_data[col]}")
                    raise ValueError(f"Invalid data for {col}: {input_data[col]}")
            else:
                print(f"Missing value for {col}")
                raise ValueError(f"Missing value for {col}")

        # Reshape numerical data and scale
        numerical_data = np.array(numerical_data).reshape(1, -1)
        scaled_numerical_data = scaler.transform(numerical_data)
        
        # Combine the sentence embedding (categorical data) and scaled numerical data
        categorical_data = np.array(product_name_embedding).reshape(1, -1)  # Product name embedding

        # Combine categorical (embedding) and numerical data
        input_combined = [categorical_data, scaled_numerical_data]
        
        # Perform multiple predictions with dropout enabled at inference time
        dropout_predictions = []
        for _ in range(n_samples):
            prediction = model(input_combined, training=True)  # Enable dropout at inference
            dropout_predictions.append(prediction[0][0])

        # Compute the mean and standard deviation of the predictions (for uncertainty)
        dropout_predictions = np.array(dropout_predictions)
        mean_prediction = np.mean(dropout_predictions)
        std_prediction = np.std(dropout_predictions)

        # Compute confidence (lower std means higher confidence)
        confidence = 1 - std_prediction  # Adjust this formula based on your confidence threshold

        return mean_prediction, confidence

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None  # Always return a tuple



# Streamlit chatbot interface
st.title("Amazon Discount Predictor Chatbot")


# User input for Amazon URL
user_input = st.text_input("Enter Amazon Product URL:")

if user_input:
    # Scrape the product data
    product_data = scrape_product_data(user_input)
    
    if product_data:
        # Show scraped data to user
        st.write("Scraped Product Data:")
        st.json(product_data)

        # Process data and predict discount
        predicted_discount, confidence = predict_discount(product_data)
        
        if predicted_discount is not None:
            # Display the predicted discount and the bot's response
            st.write(f"Predicted Discount Percentage: {round(predicted_discount * 100, 2)}%")
            st.write(f"Confidence: {round (confidence * 100, 2)}%")
        else:
            st.error("Failed to predict the discount percentage.")