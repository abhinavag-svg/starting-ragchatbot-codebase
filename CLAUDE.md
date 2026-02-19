# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Starting the Application
```bash
# Quick start using provided script
./run.sh

# Manual start (run from project root directory)
uv run uvicorn backend.app:app --reload --port 8000
```

### Package Management
```bash
# Install dependencies
uv sync

# Add new dependencies (update pyproject.toml)
uv add <package-name>
```

### Environment Setup
- Requires `.env` file with `ANTHROPIC_API_KEY=your_key_here`
- Python 3.13+ required
- Uses uv for package management

## Architecture Overview

This is a RAG (Retrieval-Augmented Generation) system with a modular architecture:

### Core Components
- **RAGSystem** (`backend/rag_system.py`): Main orchestrator that coordinates all components
- **VectorStore** (`backend/vector_store.py`): ChromaDB integration for semantic search
- **DocumentProcessor** (`backend/document_processor.py`): Text chunking and preprocessing
- **AIGenerator** (`backend/ai_generator.py`): Anthropic Claude API integration
- **SessionManager** (`backend/session_manager.py`): Conversation history management
- **ToolManager** (`backend/search_tools.py`): Search tool abstraction layer

### Application Structure
- **Backend**: FastAPI app (`backend/app.py`) serves API endpoints and static files
- **Frontend**: Simple HTML/JS/CSS interface in `frontend/` directory
- **Data**: Course materials stored in `docs/` directory, vector data in `backend/chroma_db/`

### Key Endpoints
- `POST /api/query`: Main query endpoint for RAG functionality
- `GET /api/courses`: Returns course statistics and metadata
- `/`: Serves the frontend web interface

### Data Flow
1. Course documents are processed into chunks and stored in ChromaDB
2. User queries trigger semantic search across stored content
3. Retrieved context is sent to Claude for answer generation
4. Session manager maintains conversation history for context

### Configuration
All settings are centralized in `backend/config.py` using environment variables and defaults. Key configurations include chunk size (800), overlap (150), and max search results (5).

## Vector Database Collections

### Collection Details
- `course_catalog`:
    - stores course titles for name resolution
    - metadata for each course: title, instructor, course_link, lesson_count, lessons_json (list of lessons: lesson_number, lesson_title, lesson_link)
- `course_content`:
    - stores text chunks for semantic search
    - metadata for each chunk: course_title, lesson_number, chunk_index