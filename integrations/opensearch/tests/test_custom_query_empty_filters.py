# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.dataclasses import Document
from haystack.errors import FilterError

from haystack_integrations.components.retrievers.opensearch import OpenSearchEmbeddingRetriever, OpenSearchBM25Retriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore


class TestCustomQueryEmptyFilters:
    """Test that custom queries work with empty filters."""

    @pytest.fixture
    def document_store_embedding_dim_4(self, request):
        """Create a document store for testing."""
        hosts = ["https://localhost:9200"]
        index = f"{request.node.name}"

        store = OpenSearchDocumentStore(
            hosts=hosts,
            index=index,
            http_auth=("admin", "admin"),
            verify_certs=False,
            embedding_dim=4,
            method={"space_type": "cosinesimil", "engine": "nmslib", "name": "hnsw"},
        )
        yield store
        store._client.indices.delete(index=index, params={"ignore": [400, 404]})

    def test_embedding_retrieval_with_custom_query_empty_filters_dict(self, document_store_embedding_dim_4):
        """Test embedding retrieval with custom query and empty filters dict."""
        docs = [
            Document(content="Test document 1", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="Test document 2", embedding=[0.8, 0.8, 0.8, 1.0]),
        ]
        document_store_embedding_dim_4.write_documents(docs)

        custom_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": "$query_embedding",
                                    "k": 10000,
                                }
                            }
                        }
                    ]
                }
            },
            "collapse": {
                "field": "content.keyword"
            }
        }

        # This should work without raising FilterError
        retriever = OpenSearchEmbeddingRetriever(document_store=document_store_embedding_dim_4)
        result = retriever.run(
            query_embedding=[0.1, 0.1, 0.1, 0.1],
            filters={},  # Empty dict should be handled gracefully
            top_k=10,
            custom_query=custom_query
        )
        
        assert len(result["documents"]) > 0

    def test_embedding_retrieval_with_custom_query_none_filters(self, document_store_embedding_dim_4):
        """Test embedding retrieval with custom query and None filters."""
        docs = [
            Document(content="Test document 1", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="Test document 2", embedding=[0.8, 0.8, 0.8, 1.0]),
        ]
        document_store_embedding_dim_4.write_documents(docs)

        custom_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": "$query_embedding",
                                    "k": 10000,
                                }
                            }
                        }
                    ]
                }
            }
        }

        # This should work without raising FilterError
        retriever = OpenSearchEmbeddingRetriever(document_store=document_store_embedding_dim_4)
        result = retriever.run(
            query_embedding=[0.1, 0.1, 0.1, 0.1],
            filters=None,
            top_k=10,
            custom_query=custom_query
        )
        
        assert len(result["documents"]) > 0

    def test_bm25_retrieval_with_custom_query_empty_filters_dict(self, document_store_embedding_dim_4):
        """Test BM25 retrieval with custom query and empty filters dict."""
        docs = [
            Document(content="functional programming document"),
            Document(content="another functional document"),
        ]
        document_store_embedding_dim_4.write_documents(docs)

        custom_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": "$query",
                                "type": "most_fields",
                            }
                        }
                    ]
                }
            },
            "collapse": {
                "field": "content.keyword"
            }
        }

        # This should work without raising FilterError
        retriever = OpenSearchBM25Retriever(document_store=document_store_embedding_dim_4)
        result = retriever.run(
            query="functional",
            filters={},  # Empty dict should be handled gracefully
            top_k=10,
            custom_query=custom_query
        )
        
        assert len(result["documents"]) > 0

    def test_bm25_retrieval_with_custom_query_none_filters(self, document_store_embedding_dim_4):
        """Test BM25 retrieval with custom query and None filters."""
        docs = [
            Document(content="functional programming document"),
            Document(content="another functional document"),
        ]
        document_store_embedding_dim_4.write_documents(docs)

        custom_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": "$query",
                                "type": "most_fields",
                            }
                        }
                    ]
                }
            }
        }

        # This should work without raising FilterError
        retriever = OpenSearchBM25Retriever(document_store=document_store_embedding_dim_4)
        result = retriever.run(
            query="functional",
            filters=None,
            top_k=10,
            custom_query=custom_query
        )
        
        assert len(result["documents"]) > 0