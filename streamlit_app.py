import streamlit as st
import pandas as pd
import tiktoken
from openai import OpenAI

# Load the cl100K_base tokenizer which is designed to work with ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens=500):
    sentences = text.split(". ")
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    chunks = []
    tokens_so_far = 0
    chunk = []

    for sentence, token in zip(sentences, n_tokens):
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0
        
        if token > max_tokens:
            continue
        
        chunk.append(sentence)
        tokens_so_far += token + 1
    
    if chunk:
        chunks.append(". ".join(chunk) + ".")
    
    return chunks

# Load and preprocess the data
@st.cache_data
def load_data():
    df = pd.read_csv('processed/scrapped.csv', index_col=0)
    df.columns = ['title', 'text']
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    
    shortened = []
    for _, row in df.iterrows():
        if row['text'] is None:
            continue
        if row['n_tokens'] > 500:
            shortened += split_into_many(row['text'])
        else:
            shortened.append(row['text'])
    
    return shortened

# Streamlit UI
st.title("ULM Chatbot")
st.write(
    "I am a simple ULM Chatbot. "
    "I can answer your questions related to ULMs."
)

# Load the data
data = load_data()

# Ask user for their OpenAI API key
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Create an OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Create a session state variable to store the chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response using the OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in ULMs (Unified Language Models). Use the provided context to answer questions."},
                {"role": "user", "content": f"Context: {' '.join(data[:5])}"},
                *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            ],
            stream=True,
        )

        # Stream the response
        with st.chat_message("assistant"):
            content = st.write_stream(response)
        st.session_state.messages.append({"role": "assistant", "content": content})

print("Streamlit app is running!")
