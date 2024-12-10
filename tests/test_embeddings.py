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
# IP 52.23.248.97
BASE_URL = 'http://localhost:5000'  # Adjust if your service runs on a different port
TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "Python is a versatile programming language",
    "Machine learning is transforming technology",
    "OpenAI's GPT models have revolutionized NLP"
]

def test_create_embedding():
    """Test creating embeddings"""
    logging.info("Testing embedding creation...")
    
    created_ids = []
    for text in TEST_TEXTS:
        response = requests.post(
            f'{BASE_URL}/api/embedding',
            json={'text': text},
            headers={'X-API-KEY': API_KEY}  # Include API key in the request headers
        )
        
        if response.status_code == 202:
            embedding_id = response.json()['id']
            created_ids.append(embedding_id)
            logging.info(f"Successfully created embedding for text: '{text[:30]}...' with ID: {embedding_id}")
        else:
            logging.error(f"Failed to create embedding for text: '{text[:30]}...'\nResponse: {response.json()}")
    
    return created_ids

def test_get_embedding(embedding_id):
    """Test retrieving embedding by ID"""
    logging.info(f"Testing embedding retrieval for ID: {embedding_id}")
    
    response = requests.get(
        f'{BASE_URL}/api/embedding/{embedding_id}',
        headers={'X-API-KEY': API_KEY}  # Include API key in the request headers
    )
    
    if response.status_code == 200:
        logging.info(f"Successfully retrieved embedding for ID: {embedding_id}")
        # Verify that the embedding is a list of floats with the correct dimension (1536 for ada-002)
        embedding = response.json()['embedding']
        # try to convert to list
        try:
            embedding = json.loads(embedding)
        except json.JSONDecodeError:
            logging.error(f"Failed to decode embedding as JSON: {embedding}")
            raise

        assert len(embedding) == 1536, f"Expected embedding dimension 1536, got {len(embedding)}"
    else:
        logging.error(f"Failed to retrieve embedding for ID: {embedding_id}\nResponse: {response.json()}")

def test_find_similar_embeddings(text):
    """Test finding similar embeddings"""
    logging.info(f"Testing similarity search for text: '{text[:30]}...'")
    
    response = requests.post(
        f'{BASE_URL}/api/embedding/similar',
        json={'text': text},
        headers={'X-API-KEY': API_KEY}  # Include API key in the request headers
    )
    
    if response.status_code == 200:
        similar_embeddings = response.json()['similar_embeddings']
        logging.info(f"Found {len(similar_embeddings)} similar embeddings")
        for idx, emb in enumerate(similar_embeddings):
            logging.info(f"Similarity {idx + 1}: ID={emb['id']}, Score={emb['similarity_score']:.4f}")
    else:
        logging.error(f"Failed to find similar embeddings\nResponse: {response.json()}")

def test_error_cases():
    """Test error cases"""
    logging.info("Testing error cases...")
    
    # Test missing text
    response = requests.post(f'{BASE_URL}/api/embedding', json={}, headers={'X-API-KEY': API_KEY})  # Include API key
    assert response.status_code == 400, "Expected 400 for missing text"
    logging.info("Successfully tested missing text error case")
    
    # Test invalid embedding ID
    response = requests.get(f'{BASE_URL}/api/embedding/99999999', headers={'X-API-KEY': API_KEY})  # Include API key
    print(response.json())
    assert response.status_code == 500, "Expected 500 for invalid embedding ID"
    logging.info("Successfully tested invalid embedding ID error case")

def run_tests():
    """Run all tests"""
    try:
        logging.info("Starting embedding service tests...")
        
        # Test creating embeddings
        created_ids = test_create_embedding()
        
        # Give the database a moment to process
        sleep(5)
        
        # Test retrieving embeddings
        for embedding_id in created_ids:
            test_get_embedding(embedding_id)
        
        # Test similarity search
        test_text = "Give me embeddings about ludwig von mises and international trade and interventionism"
        test_find_similar_embeddings(test_text)
        
        # Test error cases
        test_error_cases()
        
        logging.info("All tests completed successfully!")
        
    except Exception as e:
        logging.error(f"Test failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    run_tests()