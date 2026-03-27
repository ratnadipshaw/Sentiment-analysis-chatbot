import streamlit as st
from Backend_1 import (
    preprocess_text,
    typo_manage,
    get_bot_response,
    gibberish_check,
    empty_check
)

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Sentiment AI", page_icon="✨")

st.title("✨ Sentiment AI Chatbot")
st.caption("Smart sentiment detection with NLP")

# ----------------------------
# Session State
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "step" not in st.session_state:
    st.session_state.step = "input"

if "cleaned" not in st.session_state:
    st.session_state.cleaned = ""

if "corrected" not in st.session_state:
    st.session_state.corrected = ""

if "final_text" not in st.session_state:
    st.session_state.final_text = ""

# ----------------------------
# Show Chat History
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# STEP 1: INPUT
# ----------------------------
if st.session_state.step == "input":

    user_input = st.chat_input("Type your message...")

    if user_input:

        # Store user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        if user_input.strip() == "":
            st.session_state.messages.append({
                "role": "assistant",
                "content": "⚠️ Please enter a message."
            })
            st.rerun()

        cleaned = preprocess_text(user_input)
        corrected = typo_manage(cleaned)

        st.session_state.cleaned = cleaned
        st.session_state.corrected = corrected

        if cleaned != corrected:
            st.session_state.step = "suggestion"
        else:
            st.session_state.final_text = cleaned
            st.session_state.step = "process"

        st.rerun()

# ----------------------------
# STEP 2: TYPO SUGGESTION
# ----------------------------
elif st.session_state.step == "suggestion":

    with st.chat_message("assistant"):
        st.markdown(f"✏️ Did you mean:\n\n**{st.session_state.corrected}** ?")

        col1, col2 = st.columns(2)

        if col1.button("✅ Yes"):
            st.session_state.final_text = st.session_state.corrected
            st.session_state.step = "process"
            st.rerun()

        if col2.button("❌ No"):
            st.session_state.final_text = st.session_state.cleaned
            st.session_state.step = "process"
            st.rerun()

# ----------------------------
# STEP 3: PROCESS
# ----------------------------
elif st.session_state.step == "process":

    final_text = st.session_state.final_text
    response = None  # ✅ FIX: always define

    with st.chat_message("assistant"):

        if empty_check(final_text):
            response = "⚠️ Please try something meaningful."
            st.markdown(response)

        elif gibberish_check(final_text):
            response = "🤔 I didn't quite catch that. Could you rephrase?"
            st.markdown(response)

        else:
            with st.spinner("Analyzing sentiment..."):
                response = get_bot_response(final_text)

            st.markdown(f"💬 {response}")

    # ✅ Store bot message safely
    if response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    # Reset flow
    st.session_state.step = "input"
    st.rerun()