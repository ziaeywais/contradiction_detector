"""
Document loading and chunking.
Supported formats: .txt  .md  .pdf  .docx
"""

import re
import hashlib
from pathlib import Path

from .models import DocumentChunk


def load_document(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")

    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pip install pypdf")
        reader = PdfReader(str(path))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)

        if not text.strip():
            try:
                from pdfminer.high_level import extract_text as _pdfminer
                text = _pdfminer(str(path)) or ""
            except Exception:
                pass

        return text

    elif suffix == ".docx":
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise ImportError("pip install python-docx")
        doc = DocxDocument(str(path))
        from docx.oxml.ns import qn

        def _walk(el):
            for child in el.iterchildren():
                if child.tag == qn("w:p"):
                    runs = "".join(r.text for r in child.iter(qn("w:t")) if r.text)
                    if runs.strip():
                        yield runs.strip()
                elif child.tag == qn("w:tbl"):
                    for cell in child.iter(qn("w:tc")):
                        yield from _walk(cell)
                else:
                    yield from _walk(child)

        parts = list(_walk(doc.element.body))
        for section in doc.sections:
            for hf in (
                section.header, section.first_page_header, section.even_page_header,
                section.footer, section.first_page_footer, section.even_page_footer,
            ):
                try:
                    parts.extend(p.text.strip() for p in hf.paragraphs if p.text.strip())
                except Exception:
                    pass

        return "\n\n".join(parts)

    raise ValueError(f"Unsupported format: {suffix}")


def _chunk_id(doc_name: str, idx: int, text: str) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", doc_name)
    return f"{safe}__{idx}__{h}"


def chunk_document(
    text: str,
    document_name: str,
    document_path: str,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[DocumentChunk]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[DocumentChunk] = []
    current_parts: list[str] = []
    current_len = 0
    char_cursor = 0
    chunk_start = 0

    def flush(parts, start, end, idx):
        chunk_text = "\n\n".join(parts).strip()
        return DocumentChunk(
            id=_chunk_id(document_name, idx, chunk_text),
            document_name=document_name,
            document_path=document_path,
            chunk_index=idx,
            text=chunk_text,
            char_start=start,
            char_end=end,
        )

    for para in paragraphs:
        para_len = len(para)

        if current_len + para_len > chunk_size and current_parts:
            chunks.append(flush(current_parts, chunk_start, char_cursor, len(chunks)))

            overlap_parts: list[str] = []
            overlap_len = 0
            for part in reversed(current_parts):
                if overlap_len + len(part) > overlap:
                    break
                overlap_parts.insert(0, part)
                overlap_len += len(part) + 2
            current_parts = overlap_parts
            current_len = overlap_len
            chunk_start = char_cursor - overlap_len

        if para_len > chunk_size:
            for sentence in re.split(r"(?<=[.!?])\s+", para):
                s_len = len(sentence)
                if current_len + s_len > chunk_size and current_parts:
                    chunks.append(flush(current_parts, chunk_start, char_cursor, len(chunks)))
                    current_parts = []
                    current_len = 0
                    chunk_start = char_cursor
                current_parts.append(sentence)
                current_len += s_len + 1
                char_cursor += s_len + 1
        else:
            current_parts.append(para)
            current_len += para_len + 2
            char_cursor += para_len + 2

    if current_parts:
        chunk_text = "\n\n".join(current_parts).strip()
        if len(chunk_text) >= 100 or not chunks:
            chunks.append(flush(current_parts, chunk_start, char_cursor, len(chunks)))

    return chunks


def load_corpus(corpus_dir: str) -> list[DocumentChunk]:
    corpus_path = Path(corpus_dir)
    if not corpus_path.is_dir():
        raise ValueError(f"Not a directory: {corpus_dir}")

    supported = {".txt", ".md", ".pdf", ".docx"}
    doc_files = sorted(f for f in corpus_path.iterdir() if f.is_file() and f.suffix.lower() in supported)

    if not doc_files:
        raise ValueError(f"No supported documents in {corpus_dir} ({', '.join(supported)})")

    all_chunks: list[DocumentChunk] = []
    for doc_file in doc_files:
        try:
            text = load_document(str(doc_file))
            if not text.strip():
                import warnings
                warnings.warn(f"Skipping {doc_file.name}: no extractable text")
                continue
            all_chunks.extend(chunk_document(text, doc_file.name, str(doc_file)))
        except Exception as e:
            import warnings
            warnings.warn(f"Skipping {doc_file.name}: {e}")

    return all_chunks
