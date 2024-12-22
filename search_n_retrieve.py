from typing import List, Tuple, Dict
from dataclasses import dataclass
import heapq
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class ScoredDocument:
    doc: 'Document'
    score: float
    collection: str

class MultiCollectionRetriever:
    def __init__(self, collections: Dict[str, 'VectorStore'], 
                 search_params: dict = None,
                 weights: Dict[str, float] = None):
        self.collections = collections
        self.search_params = search_params or {
            "metric_type": "COSINE",
            "params": {"nprobe": 12},
        }
        self.weights = weights or {name: 1.0 for name in collections.keys()}
    
    def retrieve_merged(self, query: str, k: int = 4) -> List['Document']:
        """Merged retrieval strategy - searches all collections and merges results"""
        all_scored_docs = []
        
        for col_name, vectorstore in self.collections.items():
            docs, scores = zip(*vectorstore.similarity_search_with_score(
                query=query,
                param=self.search_params,
                k=k
            ))
            
            # Apply collection weights
            weighted_scores = [s * self.weights[col_name] for s in scores]
            
            for doc, score in zip(docs, weighted_scores):
                doc.metadata.update({
                    "score": score,
                    "collection": col_name
                })
                all_scored_docs.append(ScoredDocument(doc, score, col_name))
        
        # Get top k across all collections
        top_k_docs = heapq.nlargest(k, all_scored_docs, key=lambda x: x.score)
        return [item.doc for item in top_k_docs]
    
    async def retrieve_parallel(self, query: str, k: int = 4) -> List['Document']:
        """Parallel retrieval strategy - searches collections concurrently"""
        async def search_collection(name: str, vectorstore: 'VectorStore'):
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                docs, scores = await loop.run_in_executor(
                    pool,
                    lambda: vectorstore.similarity_search_with_score(
                        query=query,
                        param=self.search_params,
                        k=k
                    )
                )
                return [(doc, score * self.weights[name], name) for doc, score in zip(docs, scores)]
        
        # Run searches in parallel
        tasks = [
            search_collection(name, vectorstore) 
            for name, vectorstore in self.collections.items()
        ]
        results = await asyncio.gather(*tasks)
        
        # Merge and sort results
        all_docs = []
        for collection_results in results:
            for doc, score, col_name in collection_results:
                doc.metadata.update({
                    "score": score,
                    "collection": col_name
                })
                all_docs.append(ScoredDocument(doc, score, col_name))
        
        top_k_docs = heapq.nlargest(k, all_docs, key=lambda x: x.score)
        return [item.doc for item in top_k_docs]
    
    def retrieve_filtered(self, query: str, k: int = 4, 
                         collection_filter: List[str] = None) -> List['Document']:
        """Filtered retrieval strategy - searches only specific collections"""
        if collection_filter is None:
            return self.retrieve_merged(query, k)
            
        filtered_collections = {
            name: store for name, store in self.collections.items()
            if name in collection_filter
        }
        
        retriever = MultiCollectionRetriever(
            filtered_collections,
            self.search_params,
            {name: self.weights[name] for name in filtered_collections}
        )
        return retriever.retrieve_merged(query, k)