# Intelligence Service

A FastAPI application with MongoDB and Qdrant integration for document storage and vector similarity search.

## Features

- **FastAPI**: Modern, fast web framework for building APIs
- **MongoDB**: Document database for storing application data
- **Qdrant**: Vector database for similarity search and embeddings
- **Async/Await**: Full async support for better performance
- **Pydantic**: Data validation and serialization
- **Docker**: Containerized deployment with docker-compose

## Project Structure

```
intelligence-service/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── documents.py
│   │       │   ├── health.py
│   │       │   └── vectors.py
│   │       └── router.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   ├── db/
│   │   ├── mongodb.py
│   │   └── qdrant_client.py
│   ├── models/
│   │   ├── base.py
│   │   └── document.py
│   ├── schemas/
│   │   └── document.py
│   └── services/
│       ├── document_service.py
│       └── embedding_service.py
├── tests/
├── main.py
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── .env.example
```

## Quick Start

### Using Docker Compose (Recommended)

1. Clone the repository
2. Copy environment file:
   ```bash
   cp .env.example .env
   ```
3. Start services:
   ```bash
   docker-compose up -d
   ```

### Manual Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Start MongoDB and Qdrant (using Docker):
   ```bash
   docker run -d -p 27017:27017 --name mongo mongo:7.0
   docker run -d -p 6333:6333 --name qdrant qdrant/qdrant:v1.7.0
   ```

4. Run the application:
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints

### Health Check
- `GET /` - Root endpoint
- `GET /api/v1/health/` - Health check
- `GET /api/v1/health/db` - Database connectivity check

### Documents
- `POST /api/v1/documents/` - Create document
- `GET /api/v1/documents/{id}` - Get document by ID
- `GET /api/v1/documents/` - List documents
- `PUT /api/v1/documents/{id}` - Update document
- `DELETE /api/v1/documents/{id}` - Delete document
- `POST /api/v1/documents/search` - Search similar documents

### Vectors
- `POST /api/v1/vectors/search` - Search by vector
- `POST /api/v1/vectors/search/text` - Search by text
- `POST /api/v1/vectors/embed` - Generate embedding

## Configuration

Environment variables (see `.env.example`):

- `MONGODB_URL`: MongoDB connection string
- `QDRANT_HOST`: Qdrant host
- `QDRANT_PORT`: Qdrant port
- `OPENAI_API_KEY`: OpenAI API key for embeddings
- `VECTOR_DIMENSION`: Embedding dimension (default: 192, 384)

## Development

1. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run tests:
   ```bash
   pytest
   ```

3. Format code:
   ```bash
   black .
   isort .
   ```

4. Lint code:
   ```bash
   flake8 .
   ```

## API Documentation

Once running, visit:
- Interactive API docs: http://localhost:8000/docs
- ReDoc documentation: http://localhost:8000/redoc