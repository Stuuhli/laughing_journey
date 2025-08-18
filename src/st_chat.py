import streamlit as st
from streamlit_chat import message

from answer_generation import generate_answer_stream
from vector_db import get_vector_store

K = 10
EMBEDDING_MODEL = "granite-embedding:278m"
GEN_MODEL = "gemma3n:e4b"


def _init_session_state():
    """Initialize session state variables."""
    st.session_state.setdefault("past", [])
    st.session_state.setdefault("generated", [])
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = get_vector_store(
            embedding_model_name=EMBEDDING_MODEL
        )


def _on_clear():
    """Clear the chat history."""
    st.session_state.past = []
    st.session_state.generated = []


def main():
    st.title("Simple Chatbot")

    _init_session_state()

    chat_container = st.container()
    with chat_container:
        for i in range(len(st.session_state.generated)):
            message(st.session_state.past[i], is_user=True, key=f"{i}_user")
            message(st.session_state.generated[i], key=str(i))

    if prompt := st.chat_input("Your message:"):
        st.session_state.past.append(prompt)
        with chat_container:
            message(prompt, is_user=True, key=f"{len(st.session_state.past)-1}_user")
            bot_placeholder = st.empty()
        response = ""
        for chunk in generate_answer_stream(
            model=GEN_MODEL,
            query=prompt,
            k_values=K,
            vector_store=st.session_state["vector_store"],
            embedding_model_name=EMBEDDING_MODEL,
            use_full_chapters=False,
        ):
            response += chunk
            bot_placeholder.markdown(response)
        st.session_state.generated.append(response)
        st.experimental_rerun()

    st.button("Clear chat", on_click=_on_clear)


if __name__ == "__main__":
    main()
