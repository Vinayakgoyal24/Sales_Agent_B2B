# AI Sales Agent Backend

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Monitoring Dashboard](#monitoring-dashboard)
4. [RAG Architecture](#rag-architecture)
5. [RAGAS Experiments and Results](#ragas-experiments-and-results)
6. [Dataset Details](#dataset-details)
7. [Installation](#installation)
8. [Configuration](#configuration)
9. [Usage](#usage)
10. [API Documentation](#api-documentation)
11. [Contributing](#contributing)
12. [License](#license)

## Overview

This AI Sales Agent backend is a comprehensive system designed to streamline sales processes through intelligent automation. The system leverages Retrieval-Augmented Generation (RAG) to provide contextually relevant responses and supports multiple output formats including presentations, PDFs, and email communications.

## Features

### Core Capabilities
- **RAG-Enabled Chatbot**: Intelligent conversational AI with contextual understanding powered by retrieval-augmented generation
- **Dynamic PPT Generation**: Automated quotation presentation creation using Python's `python-pptx` module
- **PDF Generation**: Professional document creation using ReportLab for quotes and reports
- **Email Integration**: Automated email delivery to users via `smtplib` for seamless communication
- **Voice Support**: Windows-based voice interaction capabilities for hands-free operation
- **Bilingual Support**: Full support for English and Japanese queries and responses
- **User Authentication**: Secure user management with JWT tokens and bcrypt password hashing

### Technical Features
- RESTful API architecture
- Secure authentication and authorization
- Multi-format document generation
- Real-time chat capabilities
- Cross-platform compatibility
- Scalable database integration
- **Comprehensive Monitoring Dashboard**: Real-time system monitoring and analytics using Grafana and Prometheus

## RAG Architecture

### System Components

```
User Query → Authentication → RAG Pipeline → Response Generation → Output Format
```

### Core Technologies

#### **Vector Database**
- **Chroma DB**: Primary vector database for storing and retrieving document embeddings
- Optimized for similarity search and retrieval operations
- Supports metadata filtering and hybrid search capabilities

#### **Embedding Model**
- **Azure OpenAI Embedding v3**: High-quality text embeddings for semantic similarity
- Deployed via Azure OpenAI Service for enterprise-grade reliability
- Supports multilingual embedding generation

#### **Generation Model**
- **Azure OpenAI GPT-4.1**: Advanced language model for response generation
- Integrated with Azure OpenAI Service for scalability
- Fine-tuned for sales and customer service contexts

#### **Reranking System**
- **T5 Reranker**: Semantic reranking for improved retrieval relevance
- Post-retrieval ranking to optimize context quality
- Significantly improves response accuracy

#### **Text Processing**
- **Recursive Character Text Splitter**: Intelligent document chunking
- Maintains semantic coherence across text segments
- Optimized chunk sizes for embedding and retrieval

### Architecture Flow

1. **Document Ingestion**: Raw documents are processed and split into semantic chunks
2. **Embedding Generation**: Text chunks are converted to dense vectors using Azure OpenAI Embedding v3
3. **Vector Storage**: Embeddings are stored in Chroma DB with metadata
4. **Query Processing**: User queries are embedded and matched against stored vectors
5. **Retrieval**: Relevant chunks are retrieved using optimized search techniques
6. **Reranking**: T5 reranker improves the relevance order of retrieved chunks
7. **Generation**: GPT-4.1 generates responses using retrieved context
8. **Output Formatting**: Responses are formatted according to requested output type

## RAGAS Experiments and Results

### Experimental Framework

Our RAG system underwent comprehensive evaluation using RAGAS (Retrieval-Augmented Generation Assessment) metrics to optimize performance across multiple dimensions.

### Experiment 1: Search Technique Optimization

**Objective**: Identify the most effective retrieval strategy for our domain-specific data.

**Methods Tested**:
- **BM25**: Traditional keyword-based sparse retrieval
- **Similarity Search**: Dense vector similarity using cosine similarity
- **Hybrid Search**: Combination of BM25 and dense retrieval

**Results**:
| Method | Context Precision | Context Recall | Overall Score |
|--------|------------------|----------------|---------------|
| BM25 | 0.65 | 0.72 | 0.68 |
| Similarity Search | **0.84** | **0.89** | **0.87** |
| Hybrid Search | 0.78 | 0.81 | 0.80 |

**Conclusion**: Similarity search (dense retrieval) provided the best results with highest context precision (0.84) and context recall (0.89), making it our chosen approach.

### Experiment 2: Reranker Comparison

**Objective**: Evaluate different reranking models to improve retrieval relevance.

**Models Tested**:
- **T5 Reranker**: Transformer-based reranking model
- **ColBERT**: Efficient neural information retrieval
- **MiniLM**: Lightweight sentence transformer
- **BGE**: Beijing Academy of AI's embedding model

**Results**:
| Reranker | Relevance Score | Latency (ms) | Memory Usage |
|----------|----------------|--------------|--------------|
| **T5** | **0.92** | 145 | Medium |
| ColBERT | 0.88 | 120 | High |
| MiniLM | 0.79 | 85 | Low |
| BGE | 0.85 | 110 | Medium |

**Conclusion**: T5 reranker achieved the highest relevance score (0.92) with acceptable latency, making it optimal for our quality-focused application.

### Experiment 3: Hyperparameter Optimization

**Objective**: Optimize retrieval parameters for maximum performance.

**Parameter Tested**: Top-K retrieval count

**Configurations**:
- Top-K = 5
- Top-K = 10
- Top-K = 15
- Top-K = 20

**Results**:
| Top-K | Context Precision | Context Recall | Response Quality | Latency (ms) |
|-------|------------------|----------------|------------------|--------------|
| 5 | 0.82 | 0.79 | 0.80 | 95 |
| 10 | 0.85 | 0.84 | 0.85 | 125 |
| **15** | **0.89** | **0.87** | **0.88** | 180 |
| 20 | 0.88 | 0.88 | 0.87 | 245 |

**Conclusion**: Top-K = 15 provided the optimal balance between retrieval quality and system performance, with the highest context precision (0.89) and excellent response quality (0.88).

### Final Optimized Configuration

Based on experimental results, our production system uses:
- **Search Method**: Similarity Search (Dense Retrieval)
- **Reranker**: T5 Reranker
- **Top-K Retrieval**: 15
- **Embedding Model**: Azure OpenAI Embedding v3
- **Generation Model**: GPT-4.1

## Dataset Details

### Data Sources
- **Product Catalogs**: Comprehensive product information and specifications
- **Sales Documentation**: Historical sales data, pricing, and quotations
- **Customer Interactions**: Past customer queries and support tickets
- **Technical Specifications**: Detailed product technical documentation
- **Company Policies**: Sales policies, terms, and conditions

### Data Preprocessing
- **Language Processing**: Bilingual data (English/Japanese) normalization
- **Document Structuring**: Hierarchical organization of product information
- **Metadata Extraction**: Category, price, availability, and specification tagging
- **Quality Assurance**: Data validation and consistency checks

### Dataset Statistics
- **Total Documents**: 50,000+ processed documents
- **Chunk Count**: 500,000+ semantic chunks
- **Language Distribution**: 60% English, 40% Japanese
- **Update Frequency**: Daily incremental updates
- **Data Quality Score**: 94% accuracy after preprocessing

## Installation

### Prerequisites
- Python 3.9+
- Node.js 16+ (for frontend integration)
- Azure OpenAI Service account
- Chroma DB setup

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/your-org/ai-sales-agent-backend.git
cd ai-sales-agent-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install monitoring dependencies
pip install prometheus-client grafana-api

# Install additional packages
pip install chromadb python-pptx reportlab smtplib bcrypt PyJWT
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_VERSION=2023-12-01-preview

# Database Configuration
CHROMA_DB_PATH=./chroma_db
CHROMA_DB_HOST=localhost
CHROMA_DB_PORT=8000

# JWT Configuration
JWT_SECRET_KEY=your_jwt_secret_key
JWT_ALGORITHM=HS256
JWT_EXPIRATION_TIME=24h

# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Monitoring Configuration
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000
GRAFANA_API_KEY=your_grafana_api_key

# Metrics Collection
METRICS_ENABLED=True
METRICS_INTERVAL=30s
CUSTOM_METRICS_ENDPOINT=/metrics
```

## Usage

### Starting the Backend Server

```bash
# Start the development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or use the startup script
python run_server.py
```

### API Endpoints

#### Authentication
- `POST /signup` - User registration
- `POST /login` - User login
- `GET /me` - Get current user profile

#### Core Functionality
- `POST /query` - Smart Query Handler - Main RAG-enabled chat endpoint
- `GET /metrics` - System metrics and performance data

#### Document Generation
- `POST /generate-pdf` - Generate PDF quotations and documents
- `POST /generate-slides` - Generate PowerPoint presentations
- `POST /send-email` - Send email with generated attachments

## API Documentation

### Smart Query Handler Example

```json
POST /query
{
  "message": "I need a quote for 100 units of product X",
  "language": "en",
  "user_id": "user123"
}

Response:
{
  "response": "I can help you with that quote. Let me retrieve the current pricing for product X...",
  "context_sources": ["product_catalog.pdf", "pricing_sheet.xlsx"],
  "confidence_score": 0.92
}
```

### Authentication Examples

#### Signup
```json
POST /signup
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "secure_password123"
}

Response:
{
  "message": "User created successfully",
  "user_id": "user123"
}
```

#### Login
```json
POST /login
{
  "email": "john@example.com",
  "password": "secure_password123"
}

Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user_id": "user123"
}
```

#### Get User Profile
```json
GET /me
Headers: Authorization: Bearer <access_token>

Response:
{
  "user_id": "user123",
  "username": "john_doe",
  "email": "john@example.com",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Document Generation Examples

#### PDF Generation
```json
POST /generate-pdf
{
  "quote_data": {
    "products": [
      {
        "name": "Product X",
        "quantity": 100,
        "unit_price": 500,
        "total": 50000
      }
    ],
    "customer_info": {
      "name": "ABC Corporation",
      "email": "contact@abc.com"
    },
    "total_amount": 50000
  },
  "template": "standard_quote"
}

Response:
{
  "pdf_url": "/downloads/quote_20240115_123456.pdf",
  "file_size": "2.5MB",
  "generated_at": "2024-01-15T10:30:00Z"
}
```

#### Slides Generation
```json
POST /generate-slides
{
  "presentation_data": {
    "title": "Sales Quotation - Product X",
    "products": [...],
    "customer_info": {...},
    "total_amount": 50000
  },
  "template": "professional_quote"
}

Response:
{
  "slides_url": "/downloads/presentation_20240115_123456.pptx",
  "file_size": "8.2MB",
  "slide_count": 12,
  "generated_at": "2024-01-15T10:30:00Z"
}
```

#### Email Sending
```json
POST /send-email
{
  "recipient": "customer@example.com",
  "subject": "Your Product Quote - ABC Corporation",
  "body": "Please find attached your requested quotation.",
  "attachments": [
    "/downloads/quote_20240115_123456.pdf",
    "/downloads/presentation_20240115_123456.pptx"
  ]
}

Response:
{
  "message": "Email sent successfully",
  "email_id": "email_123456",
  "sent_at": "2024-01-15T10:35:00Z"
}
```

### System Metrics Example

```json
GET /metrics
Headers: Authorization: Bearer <access_token>

Response:
{
  "system_health": {
    "status": "healthy",
    "uptime": "5d 12h 34m",
    "last_updated": "2024-01-15T10:30:00Z"
  },
  "rag_performance": {
    "average_response_time": "1.2s",
    "context_precision": 0.89,
    "context_recall": 0.87,
    "total_queries": 15420
  },
  "document_generation": {
    "pdfs_generated": 342,
    "slides_generated": 189,
    "emails_sent": 298
  },
  "database_metrics": {
    "chroma_db_size": "2.3GB",
    "total_embeddings": 500000,
    "active_connections": 12
  },
  "monitoring_metrics": {
    "prometheus_targets": 5,
    "grafana_dashboards": 4,
    "alert_rules": 12,
    "last_scrape": "2024-01-15T10:30:00Z"
  },
  "user_analytics": {
    "new_users_today": 23,
    "active_sessions": 145,
    "avg_session_duration": "12m 34s",
    "total_queries_today": 2847
  },
  "ai_metrics": {
    "tokens_used_today": 125430,
    "avg_tokens_per_query": 234,
    "embedding_cache_hit_rate": 0.78,
    "model_response_time": "0.8s"
  }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
