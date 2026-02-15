"""Tests for CourseSearchTool.execute() method"""
import pytest
from unittest.mock import Mock, call
import sys
import os

# Add parent directory to path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


# ============================================================================
# 2.1 Parameter Combination Tests (4 tests)
# ============================================================================

def test_execute_query_only(mock_vector_store, sample_search_results):
    """Test execute with query parameter only"""
    tool = CourseSearchTool(mock_vector_store)

    result = tool.execute(query="test query")

    # Verify VectorStore.search called with correct params
    mock_vector_store.search.assert_called_once_with(
        query="test query",
        course_name=None,
        lesson_number=None
    )

    # Verify result contains formatted content
    assert "[Test Course - Lesson 1]" in result
    assert "Content from lesson 1" in result


def test_execute_query_with_course_name(mock_vector_store, sample_search_results):
    """Test execute with query and course_name filter"""
    tool = CourseSearchTool(mock_vector_store)

    result = tool.execute(query="test query", course_name="Test Course")

    # Verify VectorStore.search called with course filter
    mock_vector_store.search.assert_called_once_with(
        query="test query",
        course_name="Test Course",
        lesson_number=None
    )

    assert "[Test Course" in result


def test_execute_query_with_lesson_number(mock_vector_store, sample_search_results):
    """Test execute with query and lesson_number filter"""
    tool = CourseSearchTool(mock_vector_store)

    result = tool.execute(query="test query", lesson_number=1)

    # Verify VectorStore.search called with lesson filter
    mock_vector_store.search.assert_called_once_with(
        query="test query",
        course_name=None,
        lesson_number=1
    )

    assert "Lesson 1" in result


def test_execute_query_with_all_parameters(mock_vector_store, sample_search_results):
    """Test execute with all three parameters"""
    tool = CourseSearchTool(mock_vector_store)

    result = tool.execute(
        query="test query",
        course_name="Test Course",
        lesson_number=2
    )

    # Verify all parameters passed
    mock_vector_store.search.assert_called_once_with(
        query="test query",
        course_name="Test Course",
        lesson_number=2
    )

    assert isinstance(result, str)


# ============================================================================
# 2.2 Error Handling Tests (4 tests)
# ============================================================================

def test_execute_invalid_course_name(mock_vector_store, sample_search_results):
    """Test execute returns error message for invalid course"""
    tool = CourseSearchTool(mock_vector_store)

    # Configure mock to return error results
    mock_vector_store.search.return_value = sample_search_results["error"]

    result = tool.execute(query="test", course_name="Invalid Course")

    # Should return error message
    assert "No course found matching" in result
    assert "Invalid Course" in result


def test_execute_no_results(mock_vector_store, sample_search_results):
    """Test execute handles empty search results"""
    tool = CourseSearchTool(mock_vector_store)

    # Configure mock to return empty results
    mock_vector_store.search.return_value = sample_search_results["empty"]

    result = tool.execute(query="nonexistent content")

    # Should return "no content found" message
    assert "No relevant content found" in result


def test_execute_search_error(mock_vector_store):
    """Test execute handles SearchResults with error field"""
    tool = CourseSearchTool(mock_vector_store)

    # Configure mock to return error SearchResults
    error_results = SearchResults.empty("Search error: Database connection failed")
    mock_vector_store.search.return_value = error_results

    result = tool.execute(query="test")

    # Should return the error message
    assert "Search error" in result
    assert "Database connection failed" in result


def test_execute_chromadb_exception(mock_vector_store):
    """Test execute handles VectorStore.search() raising exception"""
    tool = CourseSearchTool(mock_vector_store)

    # Configure mock to raise exception
    mock_vector_store.search.side_effect = Exception("ChromaDB connection error")

    # Should propagate exception (or handle gracefully depending on implementation)
    with pytest.raises(Exception) as exc_info:
        tool.execute(query="test")

    assert "ChromaDB connection error" in str(exc_info.value)


# ============================================================================
# 2.3 Source Tracking Tests (5 tests)
# ============================================================================

def test_last_sources_populated_single_result(mock_vector_store):
    """Test last_sources list populated with single result"""
    tool = CourseSearchTool(mock_vector_store)

    # Configure mock with single result
    single_result = SearchResults(
        documents=["Single content"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.5]
    )
    mock_vector_store.search.return_value = single_result
    mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"

    tool.execute(query="test")

    # Verify last_sources has exactly 1 entry
    assert len(tool.last_sources) == 1
    assert tool.last_sources[0]["title"] == "Test Course - Lesson 1"


def test_last_sources_populated_multiple_results(mock_vector_store, sample_search_results):
    """Test last_sources list populated with multiple results"""
    tool = CourseSearchTool(mock_vector_store)

    tool.execute(query="test")

    # Verify last_sources has 2 entries (from sample_search_results)
    assert len(tool.last_sources) == 2
    assert all(isinstance(source, dict) for source in tool.last_sources)


def test_last_sources_with_lesson_links(mock_vector_store, sample_search_results):
    """Test lesson links retrieved and included in sources"""
    tool = CourseSearchTool(mock_vector_store)

    mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"

    tool.execute(query="test")

    # Verify get_lesson_link was called for each result with lesson_number
    assert mock_vector_store.get_lesson_link.call_count == 2

    # Verify links are in sources
    assert all(source["link"] == "http://example.com/lesson1" for source in tool.last_sources)


def test_last_sources_without_lesson_links(mock_vector_store):
    """Test sources handle None lesson links gracefully"""
    tool = CourseSearchTool(mock_vector_store)

    # Configure mock to return None for lesson links
    mock_vector_store.get_lesson_link.return_value = None

    single_result = SearchResults(
        documents=["Content"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.5]
    )
    mock_vector_store.search.return_value = single_result

    tool.execute(query="test")

    # Verify link is None in source
    assert tool.last_sources[0]["link"] is None


def test_last_sources_format(mock_vector_store, sample_search_results):
    """Test each source dict has correct structure"""
    tool = CourseSearchTool(mock_vector_store)

    tool.execute(query="test")

    # Verify each source has required keys
    for source in tool.last_sources:
        assert "title" in source
        assert "link" in source
        assert isinstance(source["title"], str)
        # link can be str or None


# ============================================================================
# 2.4 Result Formatting Tests (4 tests)
# ============================================================================

def test_format_results_with_course_only(mock_vector_store):
    """Test result header format with course but no lesson"""
    tool = CourseSearchTool(mock_vector_store)

    # Result without lesson_number
    results_no_lesson = SearchResults(
        documents=["Content"],
        metadata=[{"course_title": "Test Course", "lesson_number": None}],
        distances=[0.5]
    )
    mock_vector_store.search.return_value = results_no_lesson

    result = tool.execute(query="test")

    # Header should be just [Course Title] without lesson
    assert "[Test Course]" in result
    assert "Lesson" not in result or "None" in result


def test_format_results_with_course_and_lesson(mock_vector_store, sample_search_results):
    """Test result header format with course and lesson"""
    tool = CourseSearchTool(mock_vector_store)

    result = tool.execute(query="test")

    # Header should be [Course Title - Lesson N]
    assert "[Test Course - Lesson 1]" in result
    assert "[Test Course - Lesson 2]" in result


def test_format_results_multiple_documents(mock_vector_store, sample_search_results):
    """Test multiple results are joined correctly"""
    tool = CourseSearchTool(mock_vector_store)

    result = tool.execute(query="test")

    # Should contain both documents
    assert "Content from lesson 1" in result
    assert "Content from lesson 2" in result

    # Should be separated by double newline
    assert "\n\n" in result


def test_format_results_calls_get_lesson_link(mock_vector_store, sample_search_results):
    """Test get_lesson_link called for each result with lesson_number"""
    tool = CourseSearchTool(mock_vector_store)

    tool.execute(query="test")

    # Verify get_lesson_link called for each result
    assert mock_vector_store.get_lesson_link.call_count == 2

    # Verify calls with correct parameters
    calls = mock_vector_store.get_lesson_link.call_args_list
    assert calls[0] == call("Test Course", 1)
    assert calls[1] == call("Test Course", 2)


# ============================================================================
# 2.5 Edge Cases (6 tests)
# ============================================================================

def test_execute_empty_query_string(mock_vector_store, sample_search_results):
    """Test empty query string is handled"""
    tool = CourseSearchTool(mock_vector_store)

    result = tool.execute(query="")

    # Should still call search with empty string
    mock_vector_store.search.assert_called_once()
    assert isinstance(result, str)


def test_execute_special_characters(mock_vector_store, sample_search_results):
    """Test query with special characters and unicode"""
    tool = CourseSearchTool(mock_vector_store)

    special_query = "What is MCP? 你好 • Ñoño ™"

    result = tool.execute(query=special_query)

    # Should pass through special characters
    mock_vector_store.search.assert_called_once()
    call_args = mock_vector_store.search.call_args
    assert call_args[1]["query"] == special_query


def test_execute_very_long_query(mock_vector_store, sample_search_results):
    """Test very long query (1000+ characters)"""
    tool = CourseSearchTool(mock_vector_store)

    long_query = "test query " * 100  # ~1100 characters

    result = tool.execute(query=long_query)

    # Should handle long query
    mock_vector_store.search.assert_called_once()
    assert isinstance(result, str)


def test_execute_lesson_number_zero(mock_vector_store, sample_search_results):
    """Test lesson_number=0 edge case"""
    tool = CourseSearchTool(mock_vector_store)

    result = tool.execute(query="test", lesson_number=0)

    # Should pass through lesson_number=0
    call_args = mock_vector_store.search.call_args
    assert call_args[1]["lesson_number"] == 0


def test_execute_negative_lesson_number(mock_vector_store, sample_search_results):
    """Test negative lesson_number"""
    tool = CourseSearchTool(mock_vector_store)

    result = tool.execute(query="test", lesson_number=-1)

    # Should pass through negative value
    call_args = mock_vector_store.search.call_args
    assert call_args[1]["lesson_number"] == -1


def test_execute_none_course_name(mock_vector_store, sample_search_results):
    """Test explicitly passing course_name=None"""
    tool = CourseSearchTool(mock_vector_store)

    result = tool.execute(query="test", course_name=None)

    # Should pass None explicitly
    call_args = mock_vector_store.search.call_args
    assert call_args[1]["course_name"] is None
