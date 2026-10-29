import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
STATUS_STYLE = {"verified": ":green[VERIFIED]", "partial": ":orange[PARTIAL]", "abstained": ":gray[ABSTAINED]"}
EXAMPLES = [
    "What is multi-head attention?",
    "How does RAG-Token differ from RAG-Sequence?",
    "Which base language model is Self-RAG built on?",
    "What is the capital of Mars?",
]

st.set_page_config(page_title="Trust Layer", layout="wide")
st.title("Trust Layer")
st.caption("Every claim in the answer is checked against its source passages before you see it.")


def api_is_up() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).ok
    except requests.RequestException:
        return False


def ask_api(question: str, mode: str) -> dict | None:
    try:
        response = requests.post(f"{API_URL}/ask", json={"question": question, "mode": mode}, timeout=900)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
        return None


def render(data: dict) -> None:
    st.subheader(f"Answer {STATUS_STYLE.get(data['status'], data['status'])}")
    st.markdown(data["answer"])

    cols = st.columns(5)
    cols[0].metric("Confidence", data["confidence"])
    cols[1].metric("Claims kept", len(data["kept_claims"]))
    cols[2].metric("Claims removed", len(data["removed_claims"]))
    cols[3].metric("Retries", data["loops"])
    cols[4].metric("Latency (s)", data["latency_s"])

    sources = {s["chunk_id"]: s for s in data["sources"]}
    if data["kept_claims"]:
        st.subheader("Verified claims")
    for i, claim in enumerate(data["kept_claims"], start=1):
        st.markdown(f"**{i}.** {claim['claim']}")
        st.caption(f"{claim['method']} · confidence {claim['confidence']} · evidence {', '.join(claim['evidence_ids'])}")
        with st.expander("Show evidence"):
            for chunk_id in claim["evidence_ids"]:
                if chunk_id in sources:
                    source = sources[chunk_id]
                    st.caption(f"{chunk_id} ({source['source']}, p.{source['page']})")
                    st.text(source["text"])

    if data["removed_claims"]:
        with st.expander(f"Removed claims ({len(data['removed_claims'])}): could not be verified"):
            for claim in data["removed_claims"]:
                st.markdown(f"- **{claim['verdict']}**: {claim['claim']}  \n  _{claim['reason']}_")

    with st.expander("Attempts"):
        st.dataframe(pd.DataFrame(data["attempts"]), hide_index=True)


with st.sidebar:
    mode = st.selectbox("Retrieval mode", ["rerank", "hybrid", "dense"])
    st.write("API:", "🟢 online" if api_is_up() else "🔴 offline")
    st.write("Try:")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state["question"] = example

question = st.text_input("Ask a question about your documents", key="question")
if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Retrieving, generating and verifying..."):
        result = ask_api(question.strip(), mode)
    if result:
        render(result)