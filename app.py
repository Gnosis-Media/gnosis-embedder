from flask import Flask, request, jsonify
from openai import OpenAI
from flask_cors import CORS
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import logging
import os
from threading import Thread
from datetime import datetime
from secrets_manager import get_service_secrets

app = Flask(__name__)
CORS(app)

secrets = get_service_secrets('gnosis-embedder')

C_PORT = int(secrets.get('PORT', 5000))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

OPENAI_API_KEY = secrets.get('OPENAI_API_KEY')

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Database configuration
DB_CONFIG = {
    'dbname': secrets['POSTGRES_DB'],
    'user': secrets['POSTGRES_USER'],
    'password': secrets['POSTGRES_PASSWORD'],
    'host': secrets['POSTGRES_HOST']
}

def get_db_connection():
    """Create and return a database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logging.error(f"Database connection error: {str(e)}")
        raise

def get_embedding(text, model="text-embedding-ada-002"):
    """Get embedding from OpenAI API"""
    try:
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        logging.error(f"Error getting embedding: {str(e)}")
        raise

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )

@app.route('/api/embedding/<int:embedding_id>', methods=['GET'])
def get_embedding_by_id(embedding_id):
    """Get embedding by ID from database"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT embedding FROM embeddings WHERE id = %s", (embedding_id,))
        result = cur.fetchone()
        
        if result is None:
            return jsonify({'error': 'Embedding not found'}), 404
            
        embedding = result[0]
        
        cur.close()
        conn.close()
        
        return jsonify({
            'id': embedding_id,
            'embedding': embedding
        })
        
    except Exception as e:
        logging.error(f"Error retrieving embedding: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/embedding/similar', methods=['POST'])
def find_similar_embedding():
    """
    Find most similar embedding for given text using pgvector
    Request body should contain:
    - text: the query text
    - embedding_ids: (optional) list of embedding IDs to search within
    - limit: (optional) number of results to return (default: 5)
    """
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'No text provided'}), 400
        
    try:
        # Get embedding for input text
        input_embedding = get_embedding(request.json['text'])
        
        # Get optional parameters
        embedding_ids = request.json.get('embedding_ids', [])
        limit = request.json.get('limit', 5)
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Modify query based on whether embedding_ids are provided
        if embedding_ids:
            query = """
                SELECT id, embedding, 1 - (embedding <-> %s::vector(1536)) as similarity
                FROM embeddings
                WHERE id = ANY(%s)
                ORDER BY embedding <-> %s::vector(1536)
                LIMIT %s;
            """
            cur.execute(query, (input_embedding, embedding_ids, input_embedding, limit))
        else:
            query = """
                SELECT id, embedding, 1 - (embedding <-> %s::vector(1536)) as similarity
                FROM embeddings
                ORDER BY embedding <-> %s::vector(1536)
                LIMIT %s;
            """
            cur.execute(query, (input_embedding, input_embedding, limit))
        
        results = cur.fetchall()
        
        if not results:
            return jsonify({
                'error': 'No embeddings found in database',
                'search_space': len(embedding_ids) if embedding_ids else 'all'
            }), 404
            
        # Return top matches with their similarity scores
        similar_embeddings = [{
            'id': result[0],
            'similarity_score': float(result[2])
        } for result in results]
        
        cur.close()
        conn.close()
        
        return jsonify({
            'similar_embeddings': similar_embeddings,
            'most_similar': similar_embeddings[0],
            'search_space': len(embedding_ids) if embedding_ids else 'all'
        })
        
    except Exception as e:
        logging.error(f"Error finding similar embedding: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def update_embedding_async(embedding_id, text):
    """Background task to get and update embedding"""
    try:
        # Get embedding from OpenAI
        embedding = get_embedding(text)
        
        # Update database with the embedding
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute(
            """
            UPDATE embeddings 
            SET embedding = %s, 
                status = 'completed',
                updated_at = CURRENT_TIMESTAMP 
            WHERE id = %s
            """,
            (embedding, embedding_id)
        )
        
        conn.commit()
        cur.close()
        conn.close()
        
        logging.info(f"Successfully updated embedding {embedding_id}")
        
    except Exception as e:
        logging.error(f"Error updating embedding {embedding_id}: {str(e)}")
        # Update status to failed
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "UPDATE embeddings SET status = 'failed', updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                (embedding_id,)
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as update_error:
            logging.error(f"Error updating failed status: {str(update_error)}")


@app.route('/api/embedding', methods=['POST'])
def create_embedding():
    """Create pending embedding entry and process asynchronously"""
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'No text provided'}), 400
        
    try:
        text = request.json['text']
        
        # Create pending embedding entry
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute(
            """
            INSERT INTO embeddings (status, created_at, updated_at) 
            VALUES ('pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) 
            RETURNING id
            """
        )
        new_id = cur.fetchone()[0]
        
        conn.commit()
        cur.close()
        conn.close()
        
        # Start background thread to process embedding
        Thread(
            target=update_embedding_async,
            args=(new_id, text)
        ).start()
        
        return jsonify({
            'id': new_id,
            'status': 'pending',
            'message': 'Embedding creation started'
        }), 202
        
    except Exception as e:
        logging.error(f"Error creating embedding entry: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/embedding/<int:embedding_id>/status', methods=['GET'])
def get_embedding_status(embedding_id):
    """Get the status of an embedding"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute(
            "SELECT status, created_at, updated_at FROM embeddings WHERE id = %s",
            (embedding_id,)
        )
        result = cur.fetchone()
        
        if result is None:
            return jsonify({'error': 'Embedding not found'}), 404
            
        status, created_at, updated_at = result
        
        cur.close()
        conn.close()
        
        return jsonify({
            'id': embedding_id,
            'status': status,
            'created_at': created_at.isoformat(),
            'updated_at': updated_at.isoformat()
        })
        
    except Exception as e:
        logging.error(f"Error getting embedding status: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=C_PORT)