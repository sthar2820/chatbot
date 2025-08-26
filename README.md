# 💬 Chatbot template

A simple Streamlit app that shows how to build a chatbot using OpenAI's GPT-3.5.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-template.streamlit.app/)

## 🔐 API Key Security

This application implements secure API key handling with multiple configuration options:

### 1. **Recommended: Streamlit Secrets** 
Create `.streamlit/secrets.toml` (see `.streamlit/secrets.toml.example` for template):
```toml
[api_keys]
openai = "sk-your-actual-api-key-here"
```

### 2. **Environment Variable**
Set the `OPENAI_API_KEY` environment variable:
```bash
export OPENAI_API_KEY="sk-your-actual-api-key-here"
```

### 3. **Manual Input**
Enter your API key directly in the app (not saved between sessions)

### Security Features:
- ✅ API key validation and format checking
- ✅ Secure storage using Streamlit secrets
- ✅ Environment variable support
- ✅ No API key exposure in error messages
- ✅ Manual API key cleanup option
- ✅ Session-based key management

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Configure your OpenAI API key (see API Key Security section above)

3. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
