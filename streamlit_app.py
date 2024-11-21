import streamlit as st
import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import cosine_similarity
from tiktoken import get_encoding

# Set Streamlit page config
st.set_page_config(page_title="ULM Chatbot", page_icon="🤖")

# Define constants
MAX_TOKENS = 500
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"

# Load processed data (from your scraping pipeline)
@st.cache_data
def load_data():
    # Ensure the scraped and processed CSV is saved in the `processed/` folder
    return pd.read_csv("processed/scrapped.csv")

# Calculate embeddings for all chunks of text
@st.cache_data
def compute_embeddings(df, api_key):
    openai.api_key = api_key
    embeddings = []
    for text in df["text"]:
        response = openai.Embedding.create(input=text, model=EMBEDDING_MODEL)
        embeddings.append(response["data"][0]["embedding"])
    df["embedding"] = embeddings
    return df

# Function to find the most relevant chunk based on the query
def find_relevant_chunk(query, df, api_key):
    openai.api_key = api_key
    query_embedding = openai.Embedding.create(input=query, model=EMBEDDING_MODEL)["data"][0]["embedding"]
    df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(x, query_embedding))
    most_relevant = df.sort_values("similarity", ascending=False).iloc[0]
    return most_relevant["text"]

# Streamlit UI
st.title("ULM Chatbot 🤖")
st.write("Ask me anything about the content of the ULM's website!")

# User inputs OpenAI API key
api_key = st.text_input("Enter your OpenAI API Key", type="password")

if api_key:
    openai.api_key = api_key
    
    # Load and process scraped data
    df = load_data()
    if "embedding" not in df.columns:
        df = compute_embeddings(df, api_key)

    # Chat UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input field for new messages
    if user_input := st.chat_input("What do you want to know?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Find relevant context
        with st.spinner("Thinking..."):
            context = find_relevant_chunk(user_input, df, api_key)

            # Generate response
            completion = openai.ChatCompletion.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"}
                ],
            )
            assistant_response = completion["choices"][0]["message"]["content"]

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

else:
    st.info("Please provide your OpenAI API key to continue.")
