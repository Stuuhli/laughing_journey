import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from answer_generation import generate_answer_stream
from vector_db import get_vector_store

K = 10
EMBEDDING_MODEL = "granite-embedding:278m"
GEN_MODEL = "gemma3n:e4b"

# Chat Memory
history = StreamlitChatMessageHistory(key="chat_memory")


def _init_session_state():
    """Initialize session state variables."""
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = get_vector_store(
            embedding_model_name=EMBEDDING_MODEL
        )


def _on_clear():
    """Clear the chat history."""
    history.clear()


def main():
    st.set_page_config(
        page_title="Retrieval-Augmented Generation (RAG) Demo",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _init_session_state()

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("Settings")
        st.button("Clear chat", on_click=_on_clear)

    st.title("Retrieval-Augmented Generation Demo")

    # --- Chat Verlauf aus Memory laden ---
    for msg in history.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

    # --- Eingabe ---
    if prompt := st.chat_input("Your message:"):
        # User speichern + anzeigen
        history.add_user_message(prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Bot-Antwort mit Streaming + Markdown
        with st.chat_message("assistant"):
            with st.status("Generiere Antwort...", expanded=True) as status:
                response = st.write_stream(
                    generate_answer_stream(
                        model=GEN_MODEL,
                        query=prompt,
                        k_values=K,
                        vector_store=st.session_state["vector_store"],
                        embedding_model_name=EMBEDDING_MODEL,
                        use_full_chapters=True,
                    )
                )
                status.update(label="Antwort fertig ✅", state="complete")

        # Antwort ins Memory speichern
        history.add_ai_message(response)


if __name__ == "__main__":
    main()
