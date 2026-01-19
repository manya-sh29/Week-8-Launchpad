import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Quantised LLM",
    layout="wide",
)

st.markdown(
    """
    <style>
    body {
        background-color: black;
    }

    .stApp {
        background: linear-gradient(180deg, #ffffff 0%, black 100%);
    }

    h1, h2, h3 {
        color: #0b5ed7;
    }

    .stSidebar {
        background-color: grey;
    }

    .stSidebar * {
        color: white !important;
    }

    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
    }

    .stChatMessage.user {
        background-color: #e7f1ff;
    }

    .stChatMessage.assistant {
        background-color: #ffffff;
        border: 1px solid #d0e2ff;
    }

    button[kind="primary"] {
        background-color: white;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h1 style="text-align:center;">Quantised LLM Chat</h1>
    <p style="text-align:center; color:#4a6fa5;">
        Local LLM • Streaming • Chat & Generate
    </p>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("# Generation Controls")

system_prompt = st.sidebar.text_area(
    "System Prompt",
    placeholder="You are a helpful assistant...",
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
top_k = st.sidebar.slider("Top-k", 1, 100, 40, 1)
max_tokens = st.sidebar.slider("Max Tokens", 64, 1024, 256, 32)

mode = st.sidebar.radio("Mode", ["Chat", "Generate"])

if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append(("user", user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    payload = {
        "prompt": user_input,
        "system_prompt": system_prompt if system_prompt else None,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
    }

    if mode == "Chat":
        payload["chat_id"] = st.session_state.chat_id
        endpoint = "/chat"
    else:
        endpoint = "/generate"

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        with requests.post(
            f"{BACKEND_URL}{endpoint}",
            json=payload,
            stream=True,
        ) as response:
            buffer = ""

            for chunk in response.iter_lines(decode_unicode=True):
                if not chunk:
                    continue

                buffer += chunk

                if len(buffer) >= 10:
                    full_response += buffer
                    placeholder.markdown(full_response)
                    buffer = ""

            if buffer:
                full_response += buffer
                placeholder.markdown(full_response)

    st.session_state.messages.append(("assistant", full_response))

    if mode == "Chat":
        try:
            data = response.json()
            st.session_state.chat_id = data.get("chat_id")
        except Exception:
            pass

st.sidebar.markdown("---")
if st.sidebar.button("Clear Chat"):
    st.session_state.chat_id = None
    st.session_state.messages = []
    st.rerun()
