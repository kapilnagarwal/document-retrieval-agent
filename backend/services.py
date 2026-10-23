from backend.ingestion.embedder import Embedder
from backend.ingestion.service import IngestionService
from backend.rag.command_finder import CommandFinder
from backend.rag.document_finder import DocumentFinder
from backend.rag.generator import GeminiGenerator
from backend.rag.link_finder import LinkFinder
from backend.rag.pipeline import RAGPipeline
from backend.rag.retriever import Retriever
from backend.rag.source_catalog import SourceCatalog
from backend.rag.vector_store import VectorStore


class Services:
    def __init__(self, settings, embedder=None, generator=None):
        self.settings = settings
        self.store = VectorStore(
            settings.chroma_persist_directory,
            settings.embedding_model,
            settings.chunk_size,
            settings.chunk_overlap,
        )
        self.embedder = embedder or Embedder(settings.embedding_model)
        self.ingestion = IngestionService(settings, self.store, self.embedder)
        self.document_finder = DocumentFinder(self.ingestion)
        self.source_catalog = SourceCatalog(self.ingestion)
        self.link_finder = LinkFinder(self.source_catalog)
        self.command_finder = CommandFinder(self.source_catalog)
        self.retriever = Retriever(self.embedder, self.store, settings.retrieval_distance_threshold)
        self.pipeline = RAGPipeline(
            self.retriever,
            generator or GeminiGenerator(settings),
            settings.retrieval_top_k,
            self.ingestion,
            settings.chat_min_keyword_coverage,
        )
