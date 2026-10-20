from datetime import date
from enum import StrEnum

from pydantic import BaseModel


class Dept(StrEnum):
    RECECAO = "rececao"
    FNB = "fnb"
    COPA = "copa"
    ESTACIONAMENTO = "estacionamento"
    MANUTENCAO = "manutencao"


class DocType(StrEnum):
    SOP = "sop"
    MANUAL = "manual"
    FICHA_TECNICA = "ficha_tecnica"
    TABELA = "tabela"
    REGISTO = "registo"
    CONTRATO = "contrato"


class DocMetadata(BaseModel):
    doc_id: str
    title: str
    dept: Dept
    doc_type: DocType
    version: str
    date: date
    property_id: str = "algarve-1"
    source_path: str


class Document(BaseModel):
    metadata: DocMetadata
    text: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int
    page: int | None = None
    metadata: DocMetadata


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    source: str


class Citation(BaseModel):
    doc_id: str
    title: str
    dept: Dept
    page: int | None
    snippet: str
    char_start: int
    char_end: int


class Answer(BaseModel):
    text: str
    citations: list[Citation]
    abstained: bool
    confidence: float
