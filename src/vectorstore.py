import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .models import DocumentChunk


class VectorStore:
    def __init__(self, persist_directory: str = ".contradiction_detector_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.ef = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )
        self.collection = self.client.get_or_create_collection(
            name="document_chunks",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        # upsert is safe to call repeatedly — idempotent on ID
        self.collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "document_name": c.document_name,
                    "document_path": c.document_path,
                    "chunk_index": c.chunk_index,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "chunk_id": c.id,
                }
                for c in chunks
            ],
        )

    def query_similar(
        self,
        chunk: DocumentChunk,
        n_results: int = 5,
        min_similarity: float = 0.60,
    ) -> list[tuple[DocumentChunk, float]]:
        total = self.collection.count()
        if total < 2:
            return []

        # fetch extra candidates to account for same-doc filtering
        fetch_n = min(n_results + 15, total)

        results = self.collection.query(
            query_texts=[chunk.text],
            n_results=fetch_n,
            where={"document_name": {"$ne": chunk.document_name}},
            include=["documents", "metadatas", "distances"],
        )

        out: list[tuple[DocumentChunk, float]] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            sim = 1.0 - dist  # chroma returns cosine distance, not similarity
            if sim < min_similarity:
                continue
            out.append((
                DocumentChunk(
                    id=meta.get("chunk_id", f"{meta['document_name']}__{meta['chunk_index']}"),
                    document_name=meta["document_name"],
                    document_path=meta["document_path"],
                    chunk_index=int(meta["chunk_index"]),
                    text=doc,
                    char_start=int(meta["char_start"]),
                    char_end=int(meta["char_end"]),
                ),
                round(sim, 4),
            ))

        return out[:n_results]

    def get_all_chunks(self) -> list[DocumentChunk]:
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.get(include=["documents", "metadatas"], limit=count)
        return [
            DocumentChunk(
                id=chunk_id,
                document_name=meta["document_name"],
                document_path=meta["document_path"],
                chunk_index=int(meta["chunk_index"]),
                text=doc,
                char_start=int(meta["char_start"]),
                char_end=int(meta["char_end"]),
            )
            for chunk_id, doc, meta in zip(
                results["ids"], results["documents"], results["metadatas"]
            )
        ]

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        self.client.delete_collection("document_chunks")
        self.collection = self.client.get_or_create_collection(
            name="document_chunks",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )
