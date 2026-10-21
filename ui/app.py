"""Streamlit UI: a thin client over the FastAPI /answer endpoint."""

import os

import requests
import streamlit as st

from hotel_rag.models import Dept
from ui.logic import citation_label, citation_markdown, citations_from_payload

API_URL = os.getenv("API_URL", "http://localhost:8000")
DEPARTMENTS = ["todos", *(dept.value for dept in Dept)]
MODE_LABELS = {
    "hybrid_rerank": "Híbrido + rerank (melhor)",
    "hybrid": "Híbrido (RRF)",
    "vector": "Vetorial",
}


def ask(
    question: str,
    top_k: int = 5,
    dept: str | None = None,
    mode: str | None = None,
) -> dict:
    payload: dict = {"question": question, "top_k": top_k}
    if dept:
        payload["dept"] = dept
    if mode:
        payload["mode"] = mode

    response = requests.post(f"{API_URL}/answer", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def main() -> None:
    st.set_page_config(page_title="Hotel RAG Algarve", page_icon="🏨", layout="centered")
    st.title("Hotel RAG — Algarve")
    st.caption(
        "Pergunta sobre os procedimentos internos do hotel. "
        "Cada resposta cita os documentos usados e abstém-se quando não sabe."
    )

    with st.sidebar:
        st.header("Filtros")
        department = st.selectbox("Departamento", DEPARTMENTS)
        top_k = st.slider("Excertos a recuperar", min_value=1, max_value=10, value=5)
        mode = st.selectbox(
            "Modo de retrieval",
            list(MODE_LABELS),
            format_func=lambda key: MODE_LABELS[key],
            help="Compara as três configurações avaliadas no relatório.",
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not (question := st.chat_input("Ex.: a que horas é o check-out?")):
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    dept = None if department == "todos" else department
    with st.chat_message("assistant"):
        with st.spinner("A procurar na documentação…"):
            try:
                answer = ask(question, top_k=top_k, dept=dept, mode=mode)
            except requests.RequestException as error:
                st.error(f"Não foi possível contactar a API: {error}")
                return

        text = answer.get("text", "")
        if not text:
            st.error("A API devolveu uma resposta inesperada.")
            return

        citations = citations_from_payload(answer.get("citations") or [])

        if answer.get("abstained", True):
            st.warning(
                "Não encontrei informação suficiente na documentação para responder com segurança."
            )
        else:
            st.markdown(text)

        for index, citation in enumerate(citations, start=1):
            with st.expander(citation_label(citation, index), expanded=False):
                st.markdown(citation_markdown(citation))

        if citations:
            st.caption(f"Relevância: {answer.get('confidence', 0.0):.2f}")

    st.session_state.messages.append({"role": "assistant", "content": text})


if __name__ == "__main__":
    main()
