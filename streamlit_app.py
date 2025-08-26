import openai
from openai import OpenAI
import streamlit as st
import pandas as pd
import numpy as np
import ast
import re
import os

# Set maximum tokens for text chunking
max_tokens = 500

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def validate_api_key(api_key):
    """Validate OpenAI API key format"""
    if not api_key:
        return False
    # OpenAI API keys start with 'sk-' and contain alphanumeric characters
    pattern = r'^sk-[A-Za-z0-9]{20,}$'
    return bool(re.match(pattern, api_key))

def get_api_key():
    """Securely retrieve API key from secrets, environment, or user input"""
    # Try to get from Streamlit secrets first
    try:
        if hasattr(st, 'secrets') and 'api_keys' in st.secrets and 'openai' in st.secrets['api_keys']:
            api_key = st.secrets['api_keys']['openai']
            if validate_api_key(api_key):
                return api_key
    except Exception:
        pass
    
    # Try to get from environment variable
    env_api_key = os.getenv('OPENAI_API_KEY')
    if env_api_key and validate_api_key(env_api_key):
        return env_api_key
    
    # Fall back to user input
    return None

def clear_api_key():
    """Clear API key from memory"""
    # Clear from session state if it exists
    if 'api_key' in st.session_state:
        st.session_state.api_key = None
    if 'openai_client' in st.session_state:
        st.session_state.openai_client = None

def get_openai_client(api_key):
    """Get OpenAI client with proper error handling"""
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"❌ Error creating OpenAI client: {str(e)}")
        return None

def split_into_chunks(text, max_tokens=500):
    sentences = text.split(". ")
    chunks = []
    chunk = []
    tokens = 0
    
    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        if tokens + sentence_tokens > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens = 0
        chunk.append(sentence)
        tokens += sentence_tokens
    
    if chunk:
        chunks.append(". ".join(chunk) + ".")
    
    return chunks

def generate_embeddings(data, client):
    embeddings = []
    try:
        for text in data:
            response = client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            embeddings.append(response.data[0].embedding)
    except Exception as e:
        st.error(f"❌ Error generating embeddings: {str(e)}")
        # Don't expose API key in error messages
        if "api_key" in str(e).lower():
            st.error("Please check your API key configuration.")
        raise e
    return embeddings

def find_relevant_context(query, df, client):
    try:
        response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = response.data[0].embedding
        
        df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(query_embedding, x))
        most_relevant = df.sort_values(by='similarity', ascending=False).iloc[0]
        return most_relevant['text']
    except Exception as e:
        st.error(f"❌ Error finding relevant context: {str(e)}")
        # Don't expose API key in error messages
        if "api_key" in str(e).lower():
            st.error("Please check your API key configuration.")
        raise e

# Streamlit UI
st.title("Web-Scraped Content Chatbot")
st.write("Ask me anything about the website!")

# Secure API Key handling
api_key = get_api_key()

if not api_key:
    st.info("📝 **API Key Configuration**")
    st.write("You can provide your OpenAI API key in one of these ways:")
    st.write("1. 📁 **Recommended**: Create `.streamlit/secrets.toml` (see `.streamlit/secrets.toml.example`)")
    st.write("2. 🌍 **Environment Variable**: Set `OPENAI_API_KEY` environment variable")
    st.write("3. 🔒 **Manual Input**: Enter your API key below (will not be saved)")
    
    user_input_key = st.text_input("Enter OpenAI API Key (temporary)", type="password", key="manual_api_key")
    
    if user_input_key:
        if validate_api_key(user_input_key):
            api_key = user_input_key
            st.success("✅ Valid API key provided!")
        else:
            st.error("❌ Invalid API key format. OpenAI API keys start with 'sk-' followed by alphanumeric characters.")
            st.stop()
    else:
        st.warning("🔑 Please provide your OpenAI API key to continue.")
        st.stop()

# Set the API key securely
if api_key:
    try:
        client = get_openai_client(api_key)
        if client is None:
            st.error("❌ Failed to create OpenAI client")
            st.stop()
        # Store client and API key in session state for cleanup
        st.session_state.api_key = api_key
        st.session_state.openai_client = client
    except Exception as e:
        st.error(f"❌ Error setting up OpenAI client: {str(e)}")
        clear_api_key()
        st.stop()

    # File uploader for CSV with embeddings
    uploaded_file = st.file_uploader("Choose a CSV file with embeddings", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
        st.success("Embeddings loaded successfully!")

        # Chat UI
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_input := st.chat_input("Ask a question"):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Process input
            input_chunks = split_into_chunks(user_input)

            with st.spinner("Finding relevant context..."):
                context = find_relevant_context(user_input, df, client)
            
            with st.spinner("Generating response..."):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"}
                ]
                
                for chunk in input_chunks:
                    try:
                        completion = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=messages + [{"role": "user", "content": chunk}]
                        )
                        chunk_response = completion.choices[0].message.content
                        messages.append({"role": "assistant", "content": chunk_response})
                    except Exception as e:
                        st.error(f"❌ Error generating response: {str(e)}")
                        # Don't expose API key in error messages
                        if "api_key" in str(e).lower():
                            st.error("Please check your API key configuration.")
                        break
                
                assistant_response = " ".join([msg['content'] for msg in messages if msg['role'] == 'assistant'])
            
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    else:
        st.warning("Please upload a CSV file with embeddings.")

    # Add a section for processing new text and generating embeddings (within API key scope)
    st.sidebar.header("Process New Text")
    new_text = st.sidebar.text_area("Enter text to process")
    if st.sidebar.button("Process Text"):
        if new_text:
            with st.spinner("Processing text and generating embeddings..."):
                chunks = split_into_chunks(new_text)
                df_new = pd.DataFrame({'text': chunks})
                df_new['embedding'] = generate_embeddings(df_new['text'].tolist(), client)
                st.sidebar.download_button(
                    label="Download processed data",
                    data=df_new.to_csv(index=False),
                    file_name="processed_embeddings.csv",
                    mime="text/csv"
                )
            st.sidebar.success("Text processed and embeddings generated!")
        else:
            st.sidebar.warning("Please enter some text to process.")

# Add cleanup on session end
if st.button("🔒 Clear API Key", help="Remove API key from memory for security"):
    clear_api_key()
    st.success("API key cleared from memory!")
    st.experimental_rerun()

