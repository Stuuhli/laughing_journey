import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from answer_generation import generate_answer_stream
from vector_db import get_vector_store
from preflight import run_full_pipeline_for_new_doc

K = 10
EMBEDDING_MODEL_CHAT = "granite-embedding:278m"
GEN_MODEL_CHAT = "gemma3n:e2b"

# Chat Memory
history = StreamlitChatMessageHistory(key="chat_memory")


def _init_session_state():
    """Initialize session state variables."""
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = get_vector_store(
            embedding_model_name=EMBEDDING_MODEL_CHAT
        )


def _on_clear():
    """Clear the chat history."""
    history.clear()


def _on_submit(uploaded_files):
    if uploaded_files:
        for file in uploaded_files:
            # Check for file type and choose corresponding action
            if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                # Process Word document(s)
                run_full_pipeline_for_new_doc(file, temporary=True)
            else:
                st.warning(f"Unsupported file type: {file.type}")


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
    if prompt := st.chat_input(placeholder="Your message: ", accept_file="multiple", file_type="docx", on_submit=_on_submit(uploaded_files=st.session_state["uploaded_files"])):
        # User speichern + anzeigen
        history.add_user_message(prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Bot-Antwort mit Streaming + Markdown
        with st.chat_message("assistant"):
            with st.status("Generiere Antwort...", expanded=True) as status:
                response = st.write_stream(
                    generate_answer_stream(
                        model=GEN_MODEL_CHAT,
                        query=prompt,
                        k_values=K,
                        vector_store=st.session_state["vector_store"],
                        embedding_model_name=EMBEDDING_MODEL_CHAT,
                        history=history,
                        use_full_chapters=True,
                        max_history_messages=10,  # z.B. nur letzte 10 Messages
                    )
                )

                status.update(label="Antwort fertig ✅", state="complete")

        # Antwort ins Memory speichern
        history.add_ai_message(response)


if __name__ == "__main__":
    main()
