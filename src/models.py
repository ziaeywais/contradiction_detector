from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field


class ContradictionType(str, Enum):
    FACTUAL = "factual"
    NUMERICAL = "numerical"
    PROCEDURAL = "procedural"
    DEFINITIONAL = "definitional"
    POLICY = "policy"
    TEMPORAL = "temporal"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SeverityLevel(str, Enum):
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"


class DocumentChunk(BaseModel):
    id: str
    document_name: str
    document_path: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int


class ContradictionEvidence(BaseModel):
    id: str
    chunk_a: DocumentChunk
    chunk_b: DocumentChunk
    contradiction_type: ContradictionType
    confidence: ConfidenceLevel
    severity: SeverityLevel
    explanation: str
    claim_a: str
    claim_b: str
    topic: str
    similarity_score: float = 0.0
    detected_at: datetime = Field(default_factory=datetime.now)
    detected_by: str = "claude"  # "claude" | "ollama" | "both"


class AnalysisReport(BaseModel):
    corpus_path: str
    documents_analyzed: list[str]
    total_chunks: int
    candidate_pairs_examined: int
    contradictions_found: list[ContradictionEvidence]
    analysis_duration_seconds: float
    model_used: str
    tokens_used: int = 0
    provider: str = "claude"
    timestamp: datetime = Field(default_factory=datetime.now)
