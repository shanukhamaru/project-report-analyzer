from typing import Dict, List, TypedDict

from langgraph.graph import StateGraph
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.core.logging import get_logger
from app.core.exceptions import RetrievalError, RAGGenerationError
from app.core.config import settings
from app.vectorstore.faiss_store import FAISSStore

logger = get_logger(__name__)


# ============================================================
# Graph State Definition
# ============================================================

class RAGState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    answer: str
    citations: List[Dict]
    retrieval_mode: str


# ============================================================
# Helper: Query Intent Detection
# ============================================================

def is_comparative_query(query: str) -> bool:
    keywords = [
        "compare",
        "comparison",
        "difference",
        "timeline",
        "timelines",
        "multiple",
        "projects",
        "across",
    ]
    q = query.lower()
    return any(k in q for k in keywords)


# ============================================================
# RAG Pipeline
# ============================================================

class RAGPipeline:
    def __init__(self, vectorstore: FAISSStore) -> None:
        self.vectorstore = vectorstore

        # LM Studio (OpenAI-compatible)
        self.llm = ChatOpenAI(
            model=settings.LMSTUDIO_MODEL,
            temperature=0,
            base_url=settings.LMSTUDIO_API_BASE.strip(),
            api_key=settings.LMSTUDIO_API_KEY,
        )

        self.graph = self._build_graph()

    # --------------------------------------------------------
    # Retrieval Node
    # --------------------------------------------------------

    def _retrieve_node(self, state: RAGState) -> Dict:
        query = state["query"]

        try:
            if is_comparative_query(query):
                docs = self.vectorstore.mmr_search(query)
                mode = "MMR"
            else:
                docs = self.vectorstore.similarity_search(query)
                mode = "SIMILARITY"

            logger.info(
                "Retrieval completed | mode=%s | docs=%d",
                mode,
                len(docs),
            )

            return {
                "retrieved_docs": docs,
                "retrieval_mode": mode,
            }

        except Exception as exc:
            logger.error("Retrieval failed", exc_info=True)
            raise RetrievalError("Document retrieval failed") from exc

    # --------------------------------------------------------
    # Generation Node
    # --------------------------------------------------------

    def _generate_node(self, state: RAGState) -> Dict:
        docs = state["retrieved_docs"][:4]  # context limit for llama-2-7b
        query = state["query"]

        if not docs:
            raise RAGGenerationError("No documents available for answer generation")

        context = "\n\n".join(
            f"(Source: {d.metadata.get('source')}, Page: {d.metadata.get('page')})\n"
            f"{d.page_content}"
            for d in docs
        )

        prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

If the answer is not in the context, say "Not found in documents."
"""

        try:
            response = self.llm.invoke(prompt)
            return {"answer": response.content}

        except Exception as exc:
            logger.error("LLM generation failed", exc_info=True)
            raise RAGGenerationError("Answer generation failed") from exc

    # --------------------------------------------------------
    # Citation Node
    # --------------------------------------------------------

    def _citation_node(self, state: RAGState) -> Dict:
        citations = []

        for d in state["retrieved_docs"]:
            citations.append(
                {
                    "source": d.metadata.get("source"),
                    "page": d.metadata.get("page"),
                }
            )

        # Deduplicate citations
        unique = [dict(t) for t in {tuple(c.items()) for c in citations}]

        return {"citations": unique}

    # --------------------------------------------------------
    # Graph Builder
    # --------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(RAGState)

        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("cite", self._citation_node)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "cite")

        return graph.compile()

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def run(self, query: str) -> Dict:
        logger.info("RAG pipeline invoked")

        initial_state: RAGState = {
            "query": query,
            "retrieved_docs": [],
            "answer": "",
            "citations": [],
            "retrieval_mode": "",
        }

        result = self.graph.invoke(initial_state)

        logger.info(
            "RAG pipeline completed | retrieval_mode=%s",
            result.get("retrieval_mode"),
        )

        return result
