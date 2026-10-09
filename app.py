"""
Streamlit chat UI for AskMyThesis.

    streamlit run app.py
    streamlit run app.py -- --config path/to/config.yaml
"""

# IMPORTS ---------------------------------------------------------------------------------

import itertools
import sys
from pathlib import Path

import streamlit as st

# generation.py / retrieval.py use bare imports (`from retrieval import ...`),
# so put src/ on the path before importing the Generator.
sys.path.insert(0, str(Path(__file__).parent / "src"))
from generation import Generator  # noqa: E402

# APP -----------------------------------------------------------------------

st.set_page_config(page_title="AskMyThesis", page_icon="📖")


@st.cache_resource(show_spinner="Loading the retriever and connecting to the model…")
def load_generator():
    """
    Build the Generator once per session and reuse it
    """
    return Generator()


def render_sources(sources):
    """
    Show the cited sections under an answer, with scores when available
    """
    with st.expander(f"Sources ({len(sources)} sections retrieved)"):
        for hit in sources:
            score = hit.get("score")
            score_str = f" · distance {score:.3f}" if score is not None else ""
            st.markdown(f"**[{hit.get('number')}] {hit.get('title')}**{score_str}")


st.title("📖 AskMyThesis")
st.caption("Ask questions about the thesis, in Portuguese or English. "
           "Answers are grounded only in the document.")

generator = load_generator()

# Replay the conversation so far.
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# Handle a new question.
if query := st.chat_input("Ask about the thesis…"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        # Keep the spinner until the first token arrives (retrieval + prompt processing),
        # then stream the rest of the answer as it's generated.
        with st.spinner("Thinking…"):
            result, chunks = generator.stream(query)   # k / search_type from retrieval config
            first = next(chunks, "")
        st.write_stream(itertools.chain([first], chunks))
        if not result.get("error"):
            render_sources(result["sources"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"] if not result.get("error") else None,
    })
