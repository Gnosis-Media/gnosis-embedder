from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from openai import OpenAI
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import logging
import os
from threading import Thread
from datetime import datetime
from secrets_manager import get_service_secrets

# Create Flask app and configure
app = Flask(__name__)
CORS(app)
api = Api(app, 
    version='1.0', 
    title='Gnosis Embedder API',
    description='API for managing and querying text embeddings',
    doc='/docs'
)

# Configure namespaces
ns = api.namespace('api', description='Embedding operations')

secrets = get_service_secrets('gnosis-embedder')

C_PORT = int(secrets.get('PORT', 5000))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

OPENAI_API_KEY = secrets.get('OPENAI_API_KEY')
API_KEY = secrets.get('API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

# Database configuration
DB_CONFIG = {
    'dbname': secrets['POSTGRES_DB'],
    'user': secrets['POSTGRES_USER'],
    'password': secrets['POSTGRES_PASSWORD'],
    'host': secrets['POSTGRES_HOST']
}

# Models for request/response documentation
embedding_model = api.model('Embedding', {
    'id': fields.Integer(description='Embedding ID'),
    'embedding': fields.Raw(description='Vector embedding')  # Change this line
})

similar_request = api.model('SimilarRequest', {
    'text': fields.String(required=True, description='Text to find similar embeddings for'),
    'embedding_ids': fields.List(fields.Integer, description='Optional list of embedding IDs to search within'),
    'limit': fields.Integer(description='Number of results to return')
})

similar_response = api.model('SimilarResponse', {
    'similar_embeddings': fields.List(fields.Nested(api.model('SimilarEmbedding', {
        'id': fields.Integer,
        'similarity_score': fields.Float
    }))),
    'most_similar': fields.Nested(api.model('MostSimilar', {
        'id': fields.Integer,
        'similarity_score': fields.Float
    })),
    'search_space': fields.String
})

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

@ns.route('/embedding/<int:embedding_id>')
class EmbeddingResource(Resource):
    @api.doc('get_embedding')
    @api.marshal_with(embedding_model)
    def get(self, embedding_id):
        """Get embedding by ID from database"""
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            cur.execute("SELECT embedding FROM embeddings WHERE id = %s", (embedding_id,))
            result = cur.fetchone()
            
            if result is None:
                api.abort(404, "Embedding not found")
                
            embedding = result[0]
            
            cur.close()
            conn.close()
            
            return {
                'id': embedding_id,
                'embedding': embedding
            }
            
        except Exception as e:
            logging.error(f"Error retrieving embedding: {str(e)}")
            api.abort(500, "Internal server error")

@ns.route('/embedding/similar')
class SimilarEmbeddingResource(Resource):
    @api.doc('find_similar')
    @api.expect(similar_request)
    @api.marshal_with(similar_response)
    def post(self):
        """Find most similar embeddings for given text"""
        if not request.json or 'text' not in request.json:
            api.abort(400, "No text provided")
            
        try:
            input_embedding = get_embedding(request.json['text'])
            embedding_ids = request.json.get('embedding_ids', [])
            limit = request.json.get('limit', 5)
            
            conn = get_db_connection()
            cur = conn.cursor()
            
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
                api.abort(404, "No embeddings found in database")
                
            similar_embeddings = [{
                'id': result[0],
                'similarity_score': float(result[2])
            } for result in results]
            
            cur.close()
            conn.close()
            
            return {
                'similar_embeddings': similar_embeddings,
                'most_similar': similar_embeddings[0],
                'search_space': len(embedding_ids) if embedding_ids else 'all'
            }
            
        except Exception as e:
            logging.error(f"Error finding similar embedding: {str(e)}")
            api.abort(500, "Internal server error")

def update_embedding_async(embedding_id, text):
    """Background task to get and update embedding"""
    try:
        embedding = get_embedding(text)
        
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

@ns.route('/embedding')
class CreateEmbeddingResource(Resource):
    @api.doc('create_embedding')
    def post(self):
        """Create pending embedding entry and process asynchronously"""
        if not request.json or 'text' not in request.json:
            api.abort(400, "No text provided")
            
        try:
            text = request.json['text']
            
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
            
            Thread(
                target=update_embedding_async,
                args=(new_id, text)
            ).start()
            
            return {
                'id': new_id,
                'status': 'pending',
                'message': 'Embedding creation started'
            }, 202
            
        except Exception as e:
            logging.error(f"Error creating embedding entry: {str(e)}")
            api.abort(500, "Internal server error")

@ns.route('/embedding/<int:embedding_id>/status')
class EmbeddingStatusResource(Resource):
    @api.doc('get_status')
    def get(self, embedding_id):
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
                api.abort(404, "Embedding not found")
                
            status, created_at, updated_at = result
            
            cur.close()
            conn.close()
            
            return {
                'id': embedding_id,
                'status': status,
                'created_at': created_at.isoformat(),
                'updated_at': updated_at.isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error getting embedding status: {str(e)}")
            api.abort(500, "Internal server error")

@app.before_request
def log_request_info():
    # Exempt the /docs endpoint from logging and API key checks
    if request.path.startswith('/docs') or request.path.startswith('/swagger'):
        return

    logging.info(f"Headers: {request.headers}")
    logging.info(f"Body: {request.get_data()}")

    if 'X-API-KEY' not in request.headers:
        logging.warning("No X-API-KEY header")
        return jsonify({'error': 'No X-API-KEY'}), 401
    
    x_api_key = request.headers.get('X-API-KEY')
    if x_api_key != API_KEY:
        logging.warning("Invalid X-API-KEY")
        return jsonify({'error': 'Invalid X-API-KEY'}), 401
    else:
        return

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=C_PORT)