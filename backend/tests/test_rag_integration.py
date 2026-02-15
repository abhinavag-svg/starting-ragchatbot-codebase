"""Integration tests for RAG system coordination"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from rag_system import RAGSystem
from config import Config


# ============================================================================
# 4.1 Full Query Flow Tests (3 tests)
# ============================================================================

def test_query_content_specific_end_to_end(mocker, mock_anthropic_client_tool_use, mock_vector_store, mock_session_manager):
    """Test full flow: course question → tool use → sources returned"""
    # Patch all components
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)
    mocker.patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use)

    # Create config and RAG system
    config = Config()
    rag = RAGSystem(config)

    # Configure tool manager mock to return sources
    rag.tool_manager.execute_tool = Mock(return_value="[Test Course - Lesson 1]\nTest content")
    rag.tool_manager.get_last_sources = Mock(return_value=[
        {"title": "Test Course - Lesson 1", "link": "http://example.com/lesson1"}
    ])

    # Execute query
    response, sources = rag.query("What is MCP?", session_id="test_session")

    # Verify response and sources
    assert isinstance(response, str)
    assert len(response) > 0
    assert len(sources) > 0
    assert sources[0]["title"] == "Test Course - Lesson 1"


def test_query_general_knowledge_end_to_end(mocker, mock_anthropic_client_no_tool, mock_vector_store, mock_session_manager):
    """Test general question → no tool → empty sources"""
    # Patch all components
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)
    mocker.patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_no_tool)

    config = Config()
    rag = RAGSystem(config)

    # Configure tool manager to return empty sources
    rag.tool_manager.get_last_sources = Mock(return_value=[])

    # Execute general knowledge query
    response, sources = rag.query("What is Python?")

    # Verify direct response with no sources
    assert isinstance(response, str)
    assert len(sources) == 0


def test_query_with_tool_failure(mocker, mock_anthropic_client_tool_use, mock_vector_store, mock_session_manager):
    """Test tool returns error, AI response handles it"""
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)
    mocker.patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use)

    config = Config()
    rag = RAGSystem(config)

    # Configure tool to return error message
    rag.tool_manager.execute_tool = Mock(return_value="No course found matching 'Invalid'")
    rag.tool_manager.get_last_sources = Mock(return_value=[])

    # Execute query
    response, sources = rag.query("Tell me about Invalid Course")

    # Should still return response
    assert isinstance(response, str)
    assert len(sources) == 0  # No sources when tool returns error


# ============================================================================
# 4.2 Source Retrieval Tests (4 tests)
# ============================================================================

def test_query_returns_sources_after_tool_use(mocker, mock_anthropic_client_tool_use, mock_vector_store, mock_session_manager):
    """Test sources list populated after tool use"""
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)
    mocker.patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use)

    config = Config()
    rag = RAGSystem(config)

    # Configure tool manager with sources
    expected_sources = [
        {"title": "Test Course - Lesson 1", "link": "http://example.com/lesson1"},
        {"title": "Test Course - Lesson 2", "link": "http://example.com/lesson2"}
    ]
    rag.tool_manager.get_last_sources = Mock(return_value=expected_sources)

    response, sources = rag.query("Test query")

    # Verify get_last_sources was called
    rag.tool_manager.get_last_sources.assert_called_once()

    # Verify sources returned
    assert len(sources) == 2
    assert sources == expected_sources


def test_query_returns_empty_sources_no_tool(mocker, mock_anthropic_client_no_tool, mock_vector_store, mock_session_manager):
    """Test sources = [] when tool not used"""
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)
    mocker.patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_no_tool)

    config = Config()
    rag = RAGSystem(config)

    rag.tool_manager.get_last_sources = Mock(return_value=[])

    response, sources = rag.query("General query")

    assert sources == []


def test_query_sources_format_correct(mocker, mock_anthropic_client_tool_use, mock_vector_store, mock_session_manager):
    """Test each source has correct structure"""
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)
    mocker.patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use)

    config = Config()
    rag = RAGSystem(config)

    sources_from_tool = [
        {"title": "Course A - Lesson 1", "link": "http://example.com/a1"},
        {"title": "Course B - Lesson 2", "link": None}  # None link
    ]
    rag.tool_manager.get_last_sources = Mock(return_value=sources_from_tool)

    response, sources = rag.query("Test")

    # Verify format
    for source in sources:
        assert "title" in source
        assert "link" in source
        assert isinstance(source["title"], str)


def test_query_sources_reset_after_retrieval(mocker, mock_anthropic_client_tool_use, mock_vector_store, mock_session_manager):
    """Test reset_sources() called after get_last_sources()"""
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)
    mocker.patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use)

    config = Config()
    rag = RAGSystem(config)

    rag.tool_manager.reset_sources = Mock()

    response, sources = rag.query("Test")

    # Verify reset_sources called
    rag.tool_manager.reset_sources.assert_called_once()


# ============================================================================
# 4.3 Session Management Tests (4 tests)
# ============================================================================

def test_query_with_session_includes_history(mocker, mock_anthropic_client_no_tool, mock_vector_store, mock_session_manager):
    """Test conversation history passed to AIGenerator"""
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)

    # Mock AIGenerator to capture call
    mock_ai_gen = Mock()
    mock_ai_gen.generate_response.return_value = "Test response"
    mocker.patch('rag_system.AIGenerator', return_value=mock_ai_gen)

    config = Config()
    rag = RAGSystem(config)

    # Set up session history
    expected_history = "User: Previous question\nAssistant: Previous answer"
    mock_session_manager.get_conversation_history.return_value = expected_history

    response, sources = rag.query("New question", session_id="test_session")

    # Verify get_conversation_history was called
    mock_session_manager.get_conversation_history.assert_called_once_with("test_session")

    # Verify history passed to generate_response
    call_kwargs = mock_ai_gen.generate_response.call_args[1]
    assert call_kwargs["conversation_history"] == expected_history


def test_query_without_session_no_history(mocker, mock_anthropic_client_no_tool, mock_vector_store, mock_session_manager):
    """Test history=None when no session_id"""
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)

    mock_ai_gen = Mock()
    mock_ai_gen.generate_response.return_value = "Response"
    mocker.patch('rag_system.AIGenerator', return_value=mock_ai_gen)

    config = Config()
    rag = RAGSystem(config)

    response, sources = rag.query("Question")  # No session_id

    # Verify get_conversation_history NOT called
    mock_session_manager.get_conversation_history.assert_not_called()

    # Verify history=None passed
    call_kwargs = mock_ai_gen.generate_response.call_args[1]
    assert call_kwargs["conversation_history"] is None


def test_query_updates_session(mocker, mock_anthropic_client_no_tool, mock_vector_store, mock_session_manager):
    """Test add_exchange() called after response"""
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)
    mocker.patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_no_tool)

    config = Config()
    rag = RAGSystem(config)

    query_text = "Test question"
    response, sources = rag.query(query_text, session_id="test_session")

    # Verify add_exchange called with query and response
    mock_session_manager.add_exchange.assert_called_once()
    call_args = mock_session_manager.add_exchange.call_args[0]
    assert call_args[0] == "test_session"
    assert call_args[1] == query_text
    assert isinstance(call_args[2], str)  # Response


def test_query_session_history_format(mocker, mock_anthropic_client_no_tool, mock_vector_store, mock_session_manager):
    """Test history string format verified"""
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)

    mock_ai_gen = Mock()
    mock_ai_gen.generate_response.return_value = "Response"
    mocker.patch('rag_system.AIGenerator', return_value=mock_ai_gen)

    config = Config()
    rag = RAGSystem(config)

    # History should be formatted string
    formatted_history = "User: Q1\nAssistant: A1\nUser: Q2\nAssistant: A2"
    mock_session_manager.get_conversation_history.return_value = formatted_history

    response, sources = rag.query("New Q", session_id="test")

    # Verify formatted history passed through
    call_kwargs = mock_ai_gen.generate_response.call_args[1]
    history = call_kwargs["conversation_history"]
    assert "User:" in history
    assert "Assistant:" in history


# ============================================================================
# 4.4 Component Coordination Tests (4 tests)
# ============================================================================

def test_query_ai_generator_receives_tools(mocker, mock_anthropic_client_no_tool, mock_vector_store, mock_session_manager):
    """Test tool definitions passed from ToolManager"""
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)

    mock_ai_gen = Mock()
    mock_ai_gen.generate_response.return_value = "Response"
    mocker.patch('rag_system.AIGenerator', return_value=mock_ai_gen)

    config = Config()
    rag = RAGSystem(config)

    response, sources = rag.query("Test")

    # Verify tools passed to generate_response
    call_kwargs = mock_ai_gen.generate_response.call_args[1]
    assert "tools" in call_kwargs
    assert isinstance(call_kwargs["tools"], list)


def test_query_ai_generator_receives_tool_manager(mocker, mock_anthropic_client_no_tool, mock_vector_store, mock_session_manager):
    """Test ToolManager instance passed to AIGenerator"""
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)

    mock_ai_gen = Mock()
    mock_ai_gen.generate_response.return_value = "Response"
    mocker.patch('rag_system.AIGenerator', return_value=mock_ai_gen)

    config = Config()
    rag = RAGSystem(config)

    response, sources = rag.query("Test")

    # Verify tool_manager passed
    call_kwargs = mock_ai_gen.generate_response.call_args[1]
    assert "tool_manager" in call_kwargs
    assert call_kwargs["tool_manager"] == rag.tool_manager


def test_query_tool_manager_get_last_sources_called(mocker, mock_anthropic_client_tool_use, mock_vector_store, mock_session_manager):
    """Test get_last_sources() called after AI response"""
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)
    mocker.patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use)

    config = Config()
    rag = RAGSystem(config)

    rag.tool_manager.get_last_sources = Mock(return_value=[])

    response, sources = rag.query("Test")

    # Verify called after response generation
    rag.tool_manager.get_last_sources.assert_called_once()


def test_query_tool_manager_reset_sources_called(mocker, mock_anthropic_client_tool_use, mock_vector_store, mock_session_manager):
    """Test reset_sources() called after retrieval"""
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)
    mocker.patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use)

    config = Config()
    rag = RAGSystem(config)

    rag.tool_manager.reset_sources = Mock()

    response, sources = rag.query("Test")

    # Verify reset called
    rag.tool_manager.reset_sources.assert_called_once()


# ============================================================================
# 4.5 Error Propagation Tests (3 tests)
# ============================================================================

def test_query_ai_generator_exception_propagates(mocker, mock_vector_store, mock_session_manager):
    """Test exceptions from AIGenerator not swallowed"""
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)

    mock_ai_gen = Mock()
    mock_ai_gen.generate_response.side_effect = Exception("API error")
    mocker.patch('rag_system.AIGenerator', return_value=mock_ai_gen)

    config = Config()
    rag = RAGSystem(config)

    # Exception should propagate
    with pytest.raises(Exception) as exc_info:
        rag.query("Test")

    assert "API error" in str(exc_info.value)


def test_query_vector_store_exception(mocker, mock_anthropic_client_tool_use, mock_vector_store, mock_session_manager):
    """Test VectorStore exception handled by tool"""
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)
    mocker.patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use)

    config = Config()
    rag = RAGSystem(config)

    # Configure tool to return error (VectorStore exception caught by tool)
    rag.tool_manager.execute_tool = Mock(return_value="Search error: Database unavailable")
    rag.tool_manager.get_last_sources = Mock(return_value=[])

    # Should complete without raising exception
    response, sources = rag.query("Test")

    # Error message passed through AI response
    assert isinstance(response, str)


def test_query_session_exception_handling(mocker, mock_anthropic_client_no_tool, mock_vector_store, mock_session_manager):
    """Test session errors don't crash system"""
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)
    mocker.patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_no_tool)

    config = Config()
    rag = RAGSystem(config)

    # Configure session manager to raise exception
    mock_session_manager.get_conversation_history.side_effect = Exception("Session error")

    # Should propagate or handle gracefully
    with pytest.raises(Exception):
        rag.query("Test", session_id="bad_session")


# ============================================================================
# 4.6 Prompt Construction Tests (2 tests)
# ============================================================================

def test_query_prompt_format(mocker, mock_anthropic_client_no_tool, mock_vector_store, mock_session_manager):
    """Test prompt format: 'Answer this question about course materials: {query}'"""
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)

    mock_ai_gen = Mock()
    mock_ai_gen.generate_response.return_value = "Response"
    mocker.patch('rag_system.AIGenerator', return_value=mock_ai_gen)

    config = Config()
    rag = RAGSystem(config)

    user_query = "What is MCP?"
    response, sources = rag.query(user_query)

    # Verify prompt format
    call_kwargs = mock_ai_gen.generate_response.call_args[1]
    prompt = call_kwargs["query"]

    assert "Answer this question about course materials" in prompt
    assert user_query in prompt


def test_query_conversation_history_included(mocker, mock_anthropic_client_no_tool, mock_vector_store, mock_session_manager):
    """Test history appended to system prompt"""
    mocker.patch('rag_system.VectorStore', return_value=mock_vector_store)
    mocker.patch('rag_system.SessionManager', return_value=mock_session_manager)

    mock_ai_gen = Mock()
    mock_ai_gen.generate_response.return_value = "Response"
    mocker.patch('rag_system.AIGenerator', return_value=mock_ai_gen)

    config = Config()
    rag = RAGSystem(config)

    # Set up history
    mock_session_manager.get_conversation_history.return_value = "User: Previous\nAssistant: Answer"

    response, sources = rag.query("New question", session_id="test")

    # Verify history passed in conversation_history parameter
    call_kwargs = mock_ai_gen.generate_response.call_args[1]
    assert "Previous" in call_kwargs["conversation_history"]
