#!/usr/bin/env python3
"""Test script to verify the empty filters fix."""

from unittest.mock import Mock, patch
import numpy as np
from haystack.dataclasses import Document
from haystack_integrations.components.retrievers.opensearch import OpenSearchEmbeddingRetriever, OpenSearchBM25Retriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack_integrations.document_stores.opensearch.filters import normalize_filters


def test_embedding_retriever_with_empty_filters():
    """Test that embedding retriever works with empty filters dict."""
    print("Testing OpenSearchEmbeddingRetriever with empty filters...")
    
    # Mock the document store
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._embedding_retrieval.return_value = [Document(content="Test doc", embedding=[0.1, 0.2])]
    
    retriever = OpenSearchEmbeddingRetriever(document_store=mock_store)
    
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
            "field": "name"
        }
    }
    
    # This should not raise an error
    embedding = np.random.random(1536).tolist()
    result = retriever.run(
        query_embedding=embedding,
        filters={},  # Empty dict
        top_k=10,
        custom_query=custom_query
    )
    
    print("✓ Empty filters dict test passed")
    
    # Test with None filters as well
    result = retriever.run(
        query_embedding=embedding,
        filters=None,
        top_k=10,
        custom_query=custom_query
    )
    
    print("✓ None filters test passed")
    
    return True


def test_bm25_retriever_with_empty_filters():
    """Test that BM25 retriever works with empty filters dict."""
    print("Testing OpenSearchBM25Retriever with empty filters...")
    
    # Mock the document store
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._bm25_retrieval.return_value = [Document(content="Test doc")]
    
    retriever = OpenSearchBM25Retriever(document_store=mock_store)
    
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
            "field": "name"
        }
    }
    
    # This should not raise an error
    result = retriever.run(
        query="test",
        filters={},  # Empty dict
        top_k=10,
        custom_query=custom_query
    )
    
    print("✓ Empty filters dict test passed")
    
    # Test with None filters as well
    result = retriever.run(
        query="test",
        filters=None,
        top_k=10,
        custom_query=custom_query
    )
    
    print("✓ None filters test passed")
    
    return True


def test_normalize_filters_edge_cases():
    """Test the normalize_filters function with edge cases."""
    print("Testing normalize_filters edge cases...")
    
    try:
        # This should raise FilterError
        normalize_filters({})
        print("✗ Empty dict should have raised FilterError")
        return False
    except Exception as e:
        print(f"✓ Empty dict correctly raises: {type(e).__name__}: {e}")
    
    try:
        # This should work fine
        result = normalize_filters({"field": "test", "operator": "==", "value": "value"})
        print(f"✓ Valid filter works: {result}")
    except Exception as e:
        print(f"✗ Valid filter failed: {type(e).__name__}: {e}")
        return False
    
    return True


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_document_store_methods(_mock_opensearch_client):
    """Test document store methods with empty filters."""
    print("Testing document store methods...")
    
    document_store = OpenSearchDocumentStore(hosts="fake_host")
    
    # Test that the prepare methods handle empty filters correctly
    try:
        # Test BM25 with empty filters
        result = document_store._prepare_bm25_search_request(
            query="test",
            filters={},  # Empty dict
            fuzziness="AUTO",
            top_k=10,
            all_terms_must_match=False,
            custom_query={
                "query": {
                    "bool": {
                        "must": [{"match": {"content": "$query"}}],
                        "filter": "$filters"
                    }
                }
            }
        )
        print("✓ BM25 with empty filters works")
    except Exception as e:
        print(f"✗ BM25 with empty filters failed: {type(e).__name__}: {e}")
        return False
    
    try:
        # Test embedding with empty filters
        result = document_store._prepare_embedding_search_request(
            query_embedding=[0.1, 0.2, 0.3],
            filters={},  # Empty dict
            top_k=10,
            custom_query={
                "query": {
                    "bool": {
                        "must": [{"knn": {"embedding": {"vector": "$query_embedding", "k": 10}}}],
                        "filter": "$filters"
                    }
                }
            }
        )
        print("✓ Embedding with empty filters works")
    except Exception as e:
        print(f"✗ Embedding with empty filters failed: {type(e).__name__}: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("Running fix verification tests...\n")
    
    all_passed = True
    
    all_passed &= test_normalize_filters_edge_cases()
    print()
    
    all_passed &= test_document_store_methods()
    print()
    
    all_passed &= test_embedding_retriever_with_empty_filters()
    print()
    
    all_passed &= test_bm25_retriever_with_empty_filters()
    print()
    
    if all_passed:
        print("🎉 All tests passed! The fix appears to work correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    exit(0 if all_passed else 1)