# Amazon Discount Prediction Using TabTransformer and Sentence Transformer

## Project Overview
This project set out to build a model that predicts **discount percentages** for Amazon products. The initial goal was to use **TabTransformer**, an AI model that learns from both **categorical** and **numerical data**, to predict **new prices after discount**. However, after experimenting with the data and debugging through various stages, I shifted focus to predicting the **discount percentage**, which provided better results.

The project involved:
- **Data annotation** and careful **feature selection**.
- Experimenting with **label encoders** and **semantic sentence embeddings**.
- **Scaling** and **troubleshooting** data inputs.
- Handling **real-time data** from Amazon by scraping product information.
- **User Interface Integration**: designing and developing an interactive front-end that allows users to input data, interact with the predictor model, and receive real-time predictions and confidence feedback. 

The result is Streamlit web app to input amazon link and get predict discount percentage using AI tabtransformer model and a comparison between two TabTransformer models in the main.ipynb. One using **label encoding** and the other using **sentence transformer embeddings** for product names, which produced promising results for discount prediction.

## Data Source
The dataset was sourced from Kaggle, which included approximately 1,500 Amazon product data points. This data contained a range of information, from product names and prices to ratings and reviews. The project initially used all available columns but later reduced the feature set to only include data that could be **scraped directly from Amazon**, making the model more practical for real-world application.

## Scraping Process
Scraping the data from Amazon was a trial and error process. I had to be specific about handling **special characters** and carefully examine Amazon’s HTML code to extract the correct data. Although I considered using **AWS APIs**, I decided to go with **BeautifulSoup** for scraping. One of the major challenges was handling the variability in Amazon listings, where some products displayed both the original price and a discounted price, while others only had one price. The scraping process also encountered errors when trying to find consistent data across all listings.

The **Streamlit app** performs best with products that **do not have a discount already applied**, allowing the model to accurately predict the discount percentage.

## Initial Challenges and Pivot
Initially, I aimed to predict the **new price after discount**, but the variability in product prices led to extremely high mean absolute error (MAE) and mean squared error (MSE). The large range of prices made it difficult for the model to generalize well, and the loss was exceedingly high.

After some experimentation, I decided that the **discount percentage** would make a better target variable, given its more predictable range. This pivot improved the model’s performance and led to better results.

## Feature Selection
While the original dataset contained many columns like **review titles**, **product descriptions**, and **categories**, I realized that these features would be **difficult to scrape reliably** from live Amazon pages. As a result, I focused on the following key features:
- **Product Name**
- **Discounted Price**
- **List Price**
- **Rating**
- **Rating Count**

## Troubleshooting and Key Learnings
This project was a **tedious debugging process** as I learned the importance of **ensuring the input data to the model matches the training data exactly** in both format and structure. Some key troubleshooting steps included:

1. **Scaling Numerical Data:** I ran into issues where predictions on unseen data were far too high. After some trial and error, I realized that the **scaler** needed adjustments to ensure numerical data was properly scaled before feeding it to the model.
   
2. **Product Name Encoding:** Initially, the model was assigning label encodings based on exact product name matches from the training data, which caused issues since product names scraped from Amazon were often different. At first, I assigned default values for unseen labels, but this led to inaccurate predictions.

3. **Semantic Embeddings:** To solve the product name issue, I added a **semantic transformer** model to **embed product names**. By using a **sentence transformer**, I could generate embeddings based on **similar words** in the product name, instead of exact matches. This improved the model’s ability to generalize and predict for unseen products.

## Model Comparison: Label Encoder vs. Sentence Transformer Embedding
The final product of this project compares two models:
1. **Encoded Model**: Used **label encoding** for product names.
2. **Embedded Model**: Used a **sentence transformer** to embed product names based on their meaning.

### Results:

| Model            | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | R2 Score   |
|------------------|--------------------------|--------------------------|------------|
| Encoded Model    | 0.152784                  | 0.038491                  | 0.154836   |
| Embedded Model   | 0.137171                  | 0.031129                  | 0.316479   |



**Metric Evaluation**: The embedded model achieved a higher R² score of 0.316, compared to 0.154 for the encoded model. This indicates that the embedded model explains a greater proportion of the variance in the target variable, suggesting that using sentence embeddings for product names leads to a better overall fit and more accurate predictions
Therefore, although the difference between the two models on **test data** is small, there was a notable improvement when the **Embedded Model** was used on **completely new data** scraped directly from Amazon.
### Tab Transformer Using Label Encoder
![5bbc72ee-87e7-4434-b30f-01c45491ac13](https://github.com/user-attachments/assets/4dd13424-4b4a-4cca-9e9d-1e27e9de0678)

### Tab Transformer Using Sentence Transformer Embeddings
![c72d8cc4-ae3f-4a46-bbad-049e45151d60](https://github.com/user-attachments/assets/bb9fcb03-c92d-4606-9276-dc4690f5aa82)

The graphs clearly show that the model using embeddings captures fluctuations in the data much more effectively. While the metrics provide an overall summary, the visuals tell a more detailed story, highlighting where the models differ, especially in handling a wide range of numerical data.


## Streamlit App Overview
Alongside model training, I created a **Streamlit web app** that allows users to input an **Amazon product URL** and get a predicted discount percentage along with a confidence score. This app interacts with the **embedded model** and performs **real-time scraping** of product data.

### Key Features:
- **Amazon Data Scraping**: The app scrapes product details like **price**, **list price**, **ratings**, and **rating count** using BeautifulSoup.
- **Prediction**: The app predicts the **discount percentage** for the product and provides a **confidence score** based on multiple predictions using dropout-enabled inference.
- **Confidence Score**: The app shows how confident the model is in its predictions, calculated based on the variability of the prediction outputs.

When you clone the repo, you can easily use the web app by inserting an Amazon link. The code scrapes the appropriate data, feeds it to the model, and returns a **predicted discount** and a **confidence score**.

![Screenshot 2024-10-07 at 3 56 13 PM](https://github.com/user-attachments/assets/26362d10-6539-408b-9b8d-f14973ca0597)



## Conclusion
This project highlights the **power of TabTransformer** models in handling different data types and achieving great **MSE** and **MAE** scores. It also underscores the importance of **data quality**, feature selection, and **embedding techniques** in improving prediction results. More importantly, it illustrates how **current, live data** can influence the model's performance, and the importance of adapting models to handle unseen and real-time data.

## Skills Highlighted:
- **Data Annotation and Cleaning**: Carefully curated the dataset by filtering out features that couldn't be reliably scraped.
- **Model Training, Fine Tuning and Debugging**: Extensive trial and error to refine the model’s architecture and feature set.
- **Feature Engineering**: Transformed categorical and numerical features to improve model performance.
- **Real-Time Data Handling**: Adapted the model to predict using live data scraped from Amazon, ensuring practicality for future use cases.
- **Comparison of AI Techniques**: Showcased the difference between **label encoding** and **sentence transformer embeddings** for product names.
- **User Interaction and Real-Time Prediction:** Developed a user-facing Streamlit app that scrapes live product data from Amazon, passes it through the model to generate a predicted discount percentage, and provides a confidence score for real-time feedback to users.

