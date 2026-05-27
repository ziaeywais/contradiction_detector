"""
Contradiction detection against a VectorStore.

Providers:
  "claude"  — Anthropic API (Opus 4.7, adaptive thinking, prompt caching)
  "ollama"  — local Ollama model
  "both"    — ollama pre-filters, claude verifies hits only
"""

import json
import hashlib
import concurrent.futures
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .models import (
    DocumentChunk,
    ContradictionEvidence,
    ContradictionType,
    ConfidenceLevel,
    SeverityLevel,
)
from .vectorstore import VectorStore

# System prompt is intentionally long — Opus 4.7 requires >4096 tokens for
# prompt caching, so this sits just above that threshold. Cache hits save ~85%
# on input costs for the second+ pair in every scan.
SYSTEM_PROMPT = """You are a specialist in logical analysis and cross-document contradiction detection. Your task is to examine pairs of text excerpts from different documents and determine whether they contain genuine contradictions.

DEFINITION OF CONTRADICTION:
A contradiction exists when two statements cannot both be true simultaneously — one document asserts X while the other asserts NOT-X (or a value/rule incompatible with X). This is fundamentally different from:
- Complementary information (Document A covers one aspect, B covers another)
- Specificity differences (A gives a general rule, B gives a specific exception)
- Contextual differences (A applies to situation X, B applies to situation Y)
- Mere differences in emphasis or tone

WHAT COUNTS AS A CONTRADICTION:

1. NUMERICAL contradictions: Different quantities for the same measured thing
   Example: "employees receive 15 days of vacation" vs "all staff are entitled to 20 days of annual leave"
   NOT a contradiction: "annual salary of $80,000" vs "monthly salary of $7,200" (same amount)

2. FACTUAL contradictions: Mutually exclusive claims about the same fact
   Example: "the drug was approved in 2019" vs "FDA approval was granted in 2022"
   NOT a contradiction: "approved for adults" vs "approved for pediatric use" (different populations)

3. PROCEDURAL contradictions: Incompatible instructions for the same process
   Example: "submit requests at least 30 days in advance" vs "7 days notice is required"
   NOT a contradiction: "submit via email" vs "submit via the portal" (may be different systems)

4. POLICY contradictions: Mutually exclusive rules or permissions
   Example: "unused vacation cannot be carried over" vs "up to 5 days may be carried into the following year"
   NOT a contradiction: "employees may work remotely" vs "employees must be available during core hours" (compatible rules)

5. DEFINITIONAL contradictions: Different definitions for the same term
   Example: "'senior employee' means 10+ years of service" vs "'senior employee' refers to anyone at grade 7 or above"

6. TEMPORAL contradictions: Incompatible timelines for the same event
   Example: "the contract runs for 3 years from signing" vs "the agreement expires 18 months after execution"

SEVERITY ASSESSMENT:
- CRITICAL: Health/safety implications, legal liability, financial risk > $10K, or direct patient harm
- MAJOR: Significant operational impact, employment rights, contractual obligations
- MINOR: Administrative inconsistencies, stylistic differences in rules, low-stakes discrepancies

CONFIDENCE ASSESSMENT:
- HIGH: The contradiction is unambiguous — same subject, same context, directly opposing claims
- MEDIUM: Likely contradictory but some interpretation required (different terminology, slightly different scope)
- LOW: Possible contradiction but significant uncertainty about whether they apply to the same situation

OUTPUT FORMAT:
You MUST respond with a single JSON object. If there is a contradiction, return:
{
  "is_contradiction": true,
  "contradiction_type": "<factual|numerical|procedural|definitional|policy|temporal>",
  "confidence": "<HIGH|MEDIUM|LOW>",
  "severity": "<CRITICAL|MAJOR|MINOR>",
  "topic": "<2-5 word topic label>",
  "claim_a": "<exact quoted claim from Excerpt A, or paraphrase if long>",
  "claim_b": "<exact quoted claim from Excerpt B, or paraphrase if long>",
  "explanation": "<2-4 sentences explaining: what the contradiction is, why it matters, who would be affected>"
}

If there is NO contradiction, return:
{
  "is_contradiction": false,
  "reason": "<one sentence: why these are NOT contradictory>"
}

WORKED EXAMPLES:

Example 1 — TRUE contradiction (numerical):
Excerpt A (hr_policy_2023.txt): "Full-time employees accrue vacation at a rate of 1.25 days per month, resulting in 15 days of paid time off annually."
Excerpt B (hr_policy_2024.txt): "Effective January 1, 2024, all full-time employees are entitled to 20 days of paid vacation per calendar year."

Response:
{
  "is_contradiction": true,
  "contradiction_type": "numerical",
  "confidence": "HIGH",
  "severity": "MAJOR",
  "topic": "annual vacation days",
  "claim_a": "Full-time employees receive 15 days of paid time off annually",
  "claim_b": "All full-time employees are entitled to 20 days of paid vacation per calendar year",
  "explanation": "The 2023 policy grants 15 days of annual vacation while the 2024 policy grants 20 days. This 33% discrepancy affects employee compensation and scheduling. Any employee or manager relying on the older document would misstate the vacation entitlement."
}

Example 2 — FALSE positive (complementary):
Excerpt A (contract_a.txt): "The Software shall be delivered within 90 days of contract execution."
Excerpt B (contract_b.txt): "Milestone 1 deliverables are due within 30 days of project kickoff."

Response:
{
  "is_contradiction": false,
  "reason": "These describe different obligations — a final delivery deadline and an intermediate milestone — which are not mutually exclusive."
}

Example 3 — TRUE contradiction (policy/safety):
Excerpt A (medical_general.txt): "Adult patients may receive ibuprofen up to 3200mg per day for acute inflammatory conditions."
Excerpt B (medical_elderly.txt): "For patients over 65, the maximum recommended daily dose of ibuprofen is 1200mg due to increased risk of renal complications."

Response:
{
  "is_contradiction": true,
  "contradiction_type": "numerical",
  "confidence": "HIGH",
  "severity": "CRITICAL",
  "topic": "ibuprofen maximum dosage",
  "claim_a": "Adult patients may receive ibuprofen up to 3200mg per day",
  "claim_b": "For patients over 65, maximum daily ibuprofen is 1200mg",
  "explanation": "The general guideline permits 3200mg/day while the elderly guideline caps at 1200mg/day — a 2.7x difference. A clinician applying the wrong guideline to an elderly patient risks serious renal complications. This is a critical patient safety issue requiring immediate clarification."
}

Example 4 — TRUE contradiction (procedural):
Excerpt A (employee_handbook_v1.txt): "Employees wishing to resign must provide a minimum of two weeks written notice."
Excerpt B (employment_contract_2024.txt): "Either party may terminate this Agreement by providing no less than four weeks advance written notice."

Response:
{
  "is_contradiction": true,
  "contradiction_type": "procedural",
  "confidence": "HIGH",
  "severity": "MAJOR",
  "topic": "resignation notice period",
  "claim_a": "Employees must provide a minimum of two weeks written notice to resign",
  "claim_b": "Termination requires no less than four weeks advance written notice",
  "explanation": "The handbook requires 2 weeks notice while the employment contract requires 4 weeks. An employee relying on the handbook could be in breach of their employment contract. This discrepancy creates legal and operational uncertainty for both HR and departing employees."
}

Example 5 — FALSE positive (context-dependent):
Excerpt A (remote_policy.txt): "Employees may work remotely up to three days per week."
Excerpt B (engineering_guidelines.txt): "Engineering team members are expected to be on-site Monday through Friday."

Response:
{
  "is_contradiction": false,
  "reason": "These likely apply to different groups — a general remote work policy vs. a team-specific requirement — and may both be accurate for their respective audiences."
}

Remember: You are looking for genuine logical contradictions, not merely differences. Two documents can say different things about the same topic without contradicting each other. Apply careful analytical judgment before concluding a contradiction exists.
"""

_OLLAMA_PROMPT = """You are a logical contradiction detector. Analyze two text excerpts and determine if they contain genuine contradictions.

A contradiction is when Document A says X and Document B says the OPPOSITE of X — they cannot both be true.

Respond ONLY with valid JSON in one of these formats:

If contradiction found:
{"is_contradiction": true, "contradiction_type": "factual|numerical|procedural|definitional|policy|temporal", "confidence": "HIGH|MEDIUM|LOW", "severity": "CRITICAL|MAJOR|MINOR", "topic": "brief topic", "claim_a": "claim from A", "claim_b": "claim from B", "explanation": "what contradicts and why it matters"}

If no contradiction:
{"is_contradiction": false, "reason": "why not contradictory"}

Return ONLY the JSON object, no other text."""


def _pair_id(a: DocumentChunk, b: DocumentChunk) -> str:
    lo, hi = sorted([a.id, b.id])
    return hashlib.md5(f"{lo}|{hi}".encode()).hexdigest()[:16]


def _user_msg(a: DocumentChunk, b: DocumentChunk) -> str:
    return (
        f"Analyze the following two excerpts for contradictions.\n\n"
        f"Excerpt A (from {a.document_name}):\n```\n{a.text}\n```\n\n"
        f"Excerpt B (from {b.document_name}):\n```\n{b.text}\n```\n\n"
        f"Respond with a single JSON object as specified in your instructions."
    )


def _parse_result(
    raw: str,
    a: DocumentChunk,
    b: DocumentChunk,
    provider: str,
    sim: float = 0.0,
) -> Optional[ContradictionEvidence]:
    start = raw.find('{')
    end = raw.rfind('}')
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None

    if not data.get("is_contradiction"):
        return None

    return ContradictionEvidence(
        id=_pair_id(a, b),
        chunk_a=a,
        chunk_b=b,
        contradiction_type=ContradictionType(data["contradiction_type"]),
        confidence=ConfidenceLevel(data["confidence"]),
        severity=SeverityLevel(data["severity"]),
        explanation=data["explanation"],
        claim_a=data["claim_a"],
        claim_b=data["claim_b"],
        topic=data["topic"],
        similarity_score=sim,
        detected_by=provider,
    )


def find_candidate_pairs(
    vectorstore: VectorStore,
    all_chunks: list[DocumentChunk],
    n_similar: int = 5,
    min_similarity: float = 0.65,
) -> list[tuple[DocumentChunk, DocumentChunk, float]]:
    seen: set = set()
    candidates = []

    for chunk in all_chunks:
        for candidate, sim in vectorstore.query_similar(chunk, n_similar, min_similarity):
            pid = _pair_id(chunk, candidate)
            if pid in seen:
                continue
            seen.add(pid)
            candidates.append((chunk, candidate, sim))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
def analyze_pair_with_claude(
    client,
    chunk_a: DocumentChunk,
    chunk_b: DocumentChunk,
    tokens_used: list[int],
    similarity: float = 0.0,
) -> Optional[ContradictionEvidence]:
    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=8000,  # 2048 was too tight for deep reasoning chains
        thinking={"type": "adaptive"},
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": _user_msg(chunk_a, chunk_b)}],
    ) as stream:
        final = stream.get_final_message()

    usage = final.usage
    tokens_used[0] += usage.input_tokens + usage.output_tokens

    text = next((b.text for b in final.content if getattr(b, "type", None) == "text"), "")
    if not text.strip():
        return None

    return _parse_result(text, chunk_a, chunk_b, "claude", similarity)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
def analyze_pair_with_ollama(
    client,
    model: str,
    chunk_a: DocumentChunk,
    chunk_b: DocumentChunk,
    similarity: float = 0.0,
) -> Optional[ContradictionEvidence]:
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": _OLLAMA_PROMPT},
            {"role": "user", "content": _user_msg(chunk_a, chunk_b)},
        ],
        options={"temperature": 0.1},
    )
    text = resp.message.content or ""
    if not text.strip():
        return None
    return _parse_result(text, chunk_a, chunk_b, "ollama", similarity)


def analyze_pair_with_both(
    anthropic_client,
    ollama_client,
    ollama_model: str,
    chunk_a: DocumentChunk,
    chunk_b: DocumentChunk,
    tokens_used: list[int],
    similarity: float = 0.0,
) -> Optional[ContradictionEvidence]:
    # ollama acts as a cheap pre-filter; only call claude if ollama agrees
    if analyze_pair_with_ollama(ollama_client, ollama_model, chunk_a, chunk_b, similarity) is None:
        return None

    result = analyze_pair_with_claude(anthropic_client, chunk_a, chunk_b, tokens_used, similarity)
    if result is not None:
        result.detected_by = "both"
    return result


def analyze_pairs_concurrent(
    pairs: list[tuple[DocumentChunk, DocumentChunk, float]],
    provider: str,
    anthropic_client,
    ollama_client,
    ollama_model: str,
    tokens_used: list[int],
    max_workers: int = 4,
) -> list[ContradictionEvidence]:
    def _run(pair):
        a, b, sim = pair
        try:
            if provider == "claude":
                return analyze_pair_with_claude(anthropic_client, a, b, tokens_used, sim)
            elif provider == "ollama":
                return analyze_pair_with_ollama(ollama_client, ollama_model, a, b, sim)
            else:
                return analyze_pair_with_both(anthropic_client, ollama_client, ollama_model, a, b, tokens_used, sim)
        except Exception:
            # TODO: surface these failures somewhere — silent drop is wrong
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = pool.map(_run, pairs)

    return [r for r in results if r is not None]


def dedup_results(contradictions: list[ContradictionEvidence]) -> list[ContradictionEvidence]:
    if not contradictions:
        return []

    def _score(c: ContradictionEvidence) -> int:
        conf = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        sev = {"CRITICAL": 3, "MAJOR": 2, "MINOR": 1}
        cv = c.confidence.value if hasattr(c.confidence, "value") else c.confidence
        sv = c.severity.value if hasattr(c.severity, "value") else c.severity
        return conf.get(cv, 0) * 10 + sev.get(sv, 0)

    out = []
    for c in sorted(contradictions, key=_score, reverse=True):
        c_docs = {c.chunk_a.document_name, c.chunk_b.document_name}
        if not any(
            c.topic.lower().strip() == d.topic.lower().strip()
            and c_docs & {d.chunk_a.document_name, d.chunk_b.document_name}
            for d in out
        ):
            out.append(c)

    return out


# backwards-compat alias — streamlit_app and main.py still import the old name
# FIXME: remove once callers are updated
deduplicate_contradictions = dedup_results
