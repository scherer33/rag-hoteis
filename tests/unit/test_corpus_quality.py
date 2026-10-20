from collections import Counter
from pathlib import Path

from hotel_rag.corpus import load_corpus
from hotel_rag.models import Dept, DocType

CORPUS = Path(__file__).parents[2] / "data" / "corpus"


def test_corpus_has_expected_size_and_departments():
    docs = load_corpus(CORPUS)
    assert 30 <= len(docs) <= 35
    assert {d.metadata.dept for d in docs} == {
        Dept.RECECAO,
        Dept.FNB,
        Dept.COPA,
        Dept.ESTACIONAMENTO,
        Dept.MANUTENCAO,
    }


def test_every_document_has_substance():
    for doc in load_corpus(CORPUS):
        assert len(doc.text.splitlines()) >= 30, doc.metadata.doc_id


def test_corpus_has_versioned_pair_table_and_english_doc():
    all_docs = load_corpus(CORPUS, latest_only=False)
    versions = Counter(d.metadata.doc_id for d in all_docs)
    assert max(versions.values()) == 2
    assert any(d.metadata.doc_type is DocType.TABELA for d in all_docs)
    assert any("evacuation" in d.text.lower() for d in all_docs)


def test_doc_ids_are_unique_after_version_filtering():
    docs = load_corpus(CORPUS)
    assert len({d.metadata.doc_id for d in docs}) == len(docs)
