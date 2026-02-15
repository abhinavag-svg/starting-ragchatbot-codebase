"""Shared fixtures for RAG system tests"""
import pytest
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path for imports
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Now import project modules
from models import Course, Lesson, CourseChunk
from vector_store import SearchResults, VectorStore
from search_tools import ToolManager, CourseSearchTool

try:
    import anthropic
except ImportError:
    # Create mock anthropic module for tests
    anthropic = MagicMock()


@pytest.fixture
def sample_course_data():
    """Sample course, lessons, and chunks for testing"""
    lesson1 = Lesson(
        lesson_number=1,
        title="Introduction to Testing",
        lesson_link="http://example.com/lesson1"
    )
    lesson2 = Lesson(
        lesson_number=2,
        title="Advanced Testing",
        lesson_link="http://example.com/lesson2"
    )

    course = Course(
        title="Test Course",
        course_link="http://example.com/course",
        instructor="Test Instructor",
        lessons=[lesson1, lesson2]
    )

    chunks = [
        CourseChunk(
            content="This is test content for lesson 1",
            course_title="Test Course",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="This is test content for lesson 2",
            course_title="Test Course",
            lesson_number=2,
            chunk_index=1
        )
    ]

    return {
        "course": course,
        "lessons": [lesson1, lesson2],
        "chunks": chunks
    }


@pytest.fixture
def sample_search_results():
    """Sample SearchResults objects for different scenarios"""
    # Successful results with 2 documents
    success_results = SearchResults(
        documents=["Content from lesson 1", "Content from lesson 2"],
        metadata=[
            {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Test Course", "lesson_number": 2, "chunk_index": 1}
        ],
        distances=[0.5, 0.6]
    )

    # Empty results
    empty_results = SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )

    # Error results
    error_results = SearchResults.empty("No course found matching 'Invalid Course'")

    return {
        "success": success_results,
        "empty": empty_results,
        "error": error_results
    }


@pytest.fixture
def mock_vector_store(sample_search_results):
    """Mock VectorStore with configurable behavior"""
    mock = Mock(spec=VectorStore)

    # Default: return successful results
    mock.search.return_value = sample_search_results["success"]

    # Default: return lesson links
    mock.get_lesson_link.return_value = "http://example.com/lesson1"

    # Additional methods
    mock.get_existing_course_titles.return_value = ["Test Course", "Another Course"]
    mock.get_course_count.return_value = 2

    return mock


@pytest.fixture
def mock_anthropic_client_tool_use():
    """Mock Anthropic client that simulates tool use flow"""
    mock_client = Mock(spec=anthropic.Anthropic)

    # First call: tool_use response
    tool_use_response = Mock()
    tool_use_response.stop_reason = "tool_use"

    # Create mock tool use block
    tool_use_block = Mock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "search_course_content"
    tool_use_block.id = "toolu_123"
    tool_use_block.input = {"query": "test query"}

    tool_use_response.content = [tool_use_block]

    # Second call: final answer
    final_response = Mock()
    final_response.stop_reason = "end_turn"

    final_text_block = Mock()
    final_text_block.type = "text"
    final_text_block.text = "This is the final answer based on search results."

    final_response.content = [final_text_block]

    # Configure side_effect for sequential calls
    mock_client.messages.create.side_effect = [tool_use_response, final_response]

    return mock_client


@pytest.fixture
def mock_anthropic_client_no_tool():
    """Mock Anthropic client that returns direct answer without tool use"""
    mock_client = Mock(spec=anthropic.Anthropic)

    # Direct response without tool use
    response = Mock()
    response.stop_reason = "end_turn"

    text_block = Mock()
    text_block.type = "text"
    text_block.text = "This is a direct answer without using tools."

    response.content = [text_block]

    mock_client.messages.create.return_value = response

    return mock_client


@pytest.fixture
def mock_tool_manager():
    """Mock ToolManager for integration tests"""
    mock = Mock(spec=ToolManager)

    # Mock tool definitions
    mock.get_tool_definitions.return_value = [{
        "name": "search_course_content",
        "description": "Search course materials",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "course_name": {"type": "string"},
                "lesson_number": {"type": "integer"}
            },
            "required": ["query"]
        }
    }]

    # Mock tool execution
    mock.execute_tool.return_value = "[Test Course - Lesson 1]\nTest search results"

    # Mock source tracking
    mock.get_last_sources.return_value = [
        {"title": "Test Course - Lesson 1", "link": "http://example.com/lesson1"}
    ]

    mock.reset_sources.return_value = None

    return mock


@pytest.fixture
def mock_session_manager():
    """Mock SessionManager for integration tests"""
    from session_manager import SessionManager

    mock = Mock(spec=SessionManager)

    # Mock conversation history
    mock.get_conversation_history.return_value = "User: Previous question\nAssistant: Previous answer"

    # Mock session operations
    mock.create_session.return_value = "test_session_123"
    mock.add_exchange.return_value = None

    return mock
