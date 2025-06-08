import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import numpy as np
from difflib import get_close_matches

# Set your OpenAI API Key
openai.api_key = "your-openai-key-here"  # Replace with HF secret or env var

# Load product data
df = pd.read_excel("products.xlsx")  # Upload this file in the same repo
df.columns = df.columns.str.strip()
df['Product_Description'] = df['Product_Description'].fillna('')
product_names = df['Product_Name'].tolist()
descriptions = df['Product_Description'].tolist()

# Vectorize descriptions
vectorizer = TfidfVectorizer()
description_vectors = vectorizer.fit_transform(descriptions)

def generate_reason(product, desc):
    prompt = (
        f"Product: {product}\n"
        f"Benefits: {desc}\n"
        f"Write a short, trustworthy 1-line reason why it's a good fit for the user's concern."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful Ayurveda product recommender."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=40,
            temperature=0.5
        )
        return response['choices'][0]['message']['content'].strip()
    except:
        return "This product is effective for the mentioned concern."

def recommend_product(user_input):
    input_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(input_vector, description_vectors)
    best_index = similarity_scores.argmax()
    product = product_names[best_index]
    desc = descriptions[best_index]
    reason = generate_reason(product, desc)
    return f"✅ Recommended Product: {product}\n\n💡 Why: {reason}"

iface = gr.Interface(
    fn=recommend_product,
    inputs=gr.Textbox(lines=2, placeholder="Enter your health concern..."),
    outputs="text",
    title="Sasi Nutraceuticals Product Recommender",
    description="Enter your concern (e.g., 'hair fall', 'acne', etc.) to get the best product recommendation."
)

if __name__ == "__main__":
    iface.launch()
