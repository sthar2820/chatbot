import streamlit as st
from streamlit_react import st_react

st.set_page_config(layout="wide")

st.title("ULM Chatbot")

st_react(ULMChatbot)

