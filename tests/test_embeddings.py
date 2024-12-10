import pytest
import requests
import json
import logging
from time import sleep
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from secrets_manager import get_service_secrets

secrets = get_service_secrets('gnosis-embedder')
API_KEY = secrets.get('API_KEY')

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
BASE_URL = 'http://localhost:5000'
TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "Python is a versatile programming language",
    "Machine learning is transforming technology",
    "OpenAI's GPT models have revolutionized NLP"
]

@pytest.fixture
def created_embedding_ids():
    """Fixture to create embeddings and return their IDs"""
    created_ids = []
    for text in TEST_TEXTS:
        response = requests.post(
            f'{BASE_URL}/api/embedding',
            json={'text': text},
            headers={'X-API-KEY': API_KEY}
        )
        
        if response.status_code == 202:
            embedding_id = response.json()['id']
            created_ids.append(embedding_id)
            logging.info(f"Created embedding for text: '{text[:30]}...' with ID: {embedding_id}")
    
    # Wait for processing
    sleep(5)
    return created_ids

def test_create_embedding():
    """Test creating embeddings"""
    logging.info("Testing embedding creation...")
    
    for text in TEST_TEXTS:
        response = requests.post(
            f'{BASE_URL}/api/embedding',
            json={'text': text},
            headers={'X-API-KEY': API_KEY}
        )
        
        assert response.status_code == 202
        assert 'id' in response.json()
        logging.info(f"Successfully created embedding for text: '{text[:30]}...'")

@pytest.mark.parametrize('embedding_id', [pytest.lazy_fixture('created_embedding_ids')])
def test_get_embedding(embedding_id):
    """Test retrieving embedding by ID"""
    if isinstance(embedding_id, list):
        embedding_id = embedding_id[0]  # Take first ID if we got a list
        
    logging.info(f"Testing embedding retrieval for ID: {embedding_id}")
    
    response = requests.get(
        f'{BASE_URL}/api/embedding/{embedding_id}',
        headers={'X-API-KEY': API_KEY}
    )
    
    assert response.status_code == 200
    embedding = response.json()['embedding']
    try:
        embedding = json.loads(embedding)
    except json.JSONDecodeError:
        pytest.fail(f"Failed to decode embedding as JSON: {embedding}")

    assert len(embedding) == 1536, f"Expected embedding dimension 1536, got {len(embedding)}"
    logging.info(f"Successfully retrieved embedding for ID: {embedding_id}")

@pytest.mark.parametrize('test_text', [
    "Give me embeddings about ludwig von mises and international trade and interventionism"
])
def test_find_similar_embeddings(test_text):
    """Test finding similar embeddings"""
    logging.info(f"Testing similarity search for text: '{test_text[:30]}...'")
    
    response = requests.post(
        f'{BASE_URL}/api/embedding/similar',
        json={'text': test_text},
        headers={'X-API-KEY': API_KEY}
    )
    
    assert response.status_code == 200
    similar_embeddings = response.json()['similar_embeddings']
    assert isinstance(similar_embeddings, list)
    logging.info(f"Found {len(similar_embeddings)} similar embeddings")

def test_error_cases():
    """Test error cases"""
    logging.info("Testing error cases...")
    
    # Test missing text
    response = requests.post(
        f'{BASE_URL}/api/embedding',
        json={},
        headers={'X-API-KEY': API_KEY}
    )
    assert response.status_code == 400
    
    # Test invalid embedding ID
    response = requests.get(
        f'{BASE_URL}/api/embedding/99999999',
        headers={'X-API-KEY': API_KEY}
    )
    assert response.status_code == 500