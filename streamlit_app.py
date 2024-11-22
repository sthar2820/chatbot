import openai
import streamlit as st
import pandas as pd
from openai.embeddings_utils import cosine_similarity
import ast

# Set maximum tokens for text chunking
max_tokens = 500

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

def generate_embeddings(data):
    embeddings = []
    for text in data:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embeddings.append(response['data'][0]['embedding'])
    return embeddings

def find_relevant_context(query, df):
    query_embedding = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )['data'][0]['embedding']
    
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(query_embedding, x))
    most_relevant = df.sort_values(by='similarity', ascending=False).iloc[0]
    return most_relevant['text']

# Streamlit UI
st.title("Web-Scraped Content Chatbot")
st.write("Ask me anything about the website!")

# API Key Input
api_key = st.text_input("Enter OpenAI API Key", type="password")
if api_key:
    openai.api_key = api_key

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
                context = find_relevant_context(user_input, df)
            
            with st.spinner("Generating response..."):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"}
                ]
                
                for chunk in input_chunks:
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages + [{"role": "user", "content": chunk}]
                    )
                    chunk_response = completion['choices'][0]['message']['content']
                    messages.append({"role": "assistant", "content": chunk_response})
                
                assistant_response = " ".join([msg['content'] for msg in messages if msg['role'] == 'assistant'])
            
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    else:
        st.warning("Please upload a CSV file with embeddings.")
else:
    st.warning("Please enter your OpenAI API key to continue.")

# Add a section for processing new text and generating embeddings
st.sidebar.header("Process New Text")
new_text = st.sidebar.text_area("Enter text to process")
if st.sidebar.button("Process Text"):
    if new_text:
        with st.spinner("Processing text and generating embeddings..."):
            chunks = split_into_chunks(new_text)
            df_new = pd.DataFrame({'text': chunks})
            df_new['embedding'] = generate_embeddings(df_new['text'].tolist())
            st.sidebar.download_button(
                label="Download processed data",
                data=df_new.to_csv(index=False),
                file_name="processed_embeddings.csv",
                mime="text/csv"
            )
        st.sidebar.success("Text processed and embeddings generated!")
    else:
        st.sidebar.warning("Please enter some text to process.")

