"""Streamlit UI: a thin client over the FastAPI /answer endpoint.

Editorial "resort de luxo" presentation layer. All business decisions live in
``ui/logic.py``; this module only wires Streamlit widgets to the API.
"""

import os

import requests
import streamlit as st

from hotel_rag.models import Dept
from ui import theme
from ui.logic import (
    answer_text,
    citation_card_html,
    citations_from_payload,
    example_questions,
)

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


def masthead() -> None:
    st.markdown(
        """
        <div class="rag-eyebrow">Concierge · Algarve</div>
        <div class="rag-title">Hotel RAG</div>
        <div class="rag-subtitle">
        Pergunta sobre os procedimentos internos do hotel. Cada resposta cita os
        documentos usados e abstém-se quando não sabe.
        </div>
        <hr class="rag-rule"><hr class="rag-rule thin">
        """,
        unsafe_allow_html=True,
    )


def example_chips() -> None:
    """Demo questions when the history is empty; the last one abstains."""
    st.caption("Experimenta:")
    for question in example_questions():
        if st.button(question, key=f"chip:{question}", use_container_width=False):
            st.session_state.pending_question = question
            st.rerun()


def sidebar() -> tuple[str, int, str]:
    with st.sidebar:
        st.markdown("Filtros")
        department = st.selectbox("Departamento", DEPARTMENTS)
        top_k = st.slider("Excertos a recuperar", min_value=1, max_value=10, value=5)
        mode = st.selectbox(
            "Modo de retrieval",
            list(MODE_LABELS),
            format_func=lambda key: MODE_LABELS[key],
            help="Compara as três configurações avaliadas no relatório.",
        )
    return department, top_k, mode


def render_answer(answer: dict) -> str:
    text = answer_text(answer)
    if not text:
        st.error("A API devolveu uma resposta inesperada.")
        return ""

    citations = citations_from_payload(answer.get("citations") or [])

    if answer.get("abstained", True):
        st.markdown(
            f'<div class="abstain-card">💛 {text}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(text)

    for index, citation in enumerate(citations, start=1):
        st.markdown(
            citation_card_html(citation, index),
            unsafe_allow_html=True,
        )

    if not answer.get("abstained", True):
        st.markdown(
            theme.confidence_meter(float(answer.get("confidence", 0.0))), unsafe_allow_html=True
        )

    return text


def main() -> None:
    st.set_page_config(page_title="Hotel RAG Algarve", page_icon="🏨", layout="centered")
    theme.inject()
    masthead()
    department, top_k, mode = sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                st.markdown(message["content"], unsafe_allow_html=True)

    if not st.session_state.messages:
        example_chips()

    question = st.chat_input("Ex.: a que horas é o check-out?")

    # If the user clicked an example chip, feed it in automatically.
    pending = st.session_state.pop("pending_question", None)
    if pending and not question:
        question = pending

    if not question:
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

        text = render_answer(answer)

    if text:
        st.session_state.messages.append({"role": "assistant", "content": text})


if __name__ == "__main__":
    main()
