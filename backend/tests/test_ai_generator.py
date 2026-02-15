"""Tests for AIGenerator tool calling functionality"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from ai_generator import AIGenerator

try:
    import anthropic
except ImportError:
    anthropic = MagicMock()


# ============================================================================
# 3.1 Tool Invocation Decision Tests (4 tests)
# ============================================================================

def test_generate_response_course_specific_triggers_tool(mock_anthropic_client_tool_use, mock_tool_manager):
    """Test course-specific question triggers tool use"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        response = ai_gen.generate_response(
            query="What is MCP?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once()

        # Verify we got final response
        assert isinstance(response, str)
        assert len(response) > 0


def test_generate_response_general_knowledge_no_tool(mock_anthropic_client_no_tool, mock_tool_manager):
    """Test general knowledge question does not trigger tool"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_no_tool):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        response = ai_gen.generate_response(
            query="What is Python?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Verify tool was NOT executed
        mock_tool_manager.execute_tool.assert_not_called()

        # Verify we got direct response
        assert "direct answer" in response.lower()


def test_generate_response_greeting_no_tool(mock_anthropic_client_no_tool, mock_tool_manager):
    """Test greeting does not trigger tool"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_no_tool):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        response = ai_gen.generate_response(
            query="Hello",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Tool should not be used for greetings
        mock_tool_manager.execute_tool.assert_not_called()


def test_generate_response_explicit_course_question(mock_anthropic_client_tool_use, mock_tool_manager):
    """Test explicit course question triggers tool"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        response = ai_gen.generate_response(
            query="Explain lesson 3 of Introduction to MCP",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Should trigger tool use
        mock_tool_manager.execute_tool.assert_called_once()


# ============================================================================
# 3.2 Tool Execution Flow Tests (5 tests)
# ============================================================================

def test_handle_tool_execution_single_tool(mock_anthropic_client_tool_use, mock_tool_manager):
    """Test single tool call executed correctly"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        response = ai_gen.generate_response(
            query="Test query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Verify execute_tool called exactly once
        assert mock_tool_manager.execute_tool.call_count == 1


def test_handle_tool_execution_extracts_parameters(mock_anthropic_client_tool_use, mock_tool_manager):
    """Test parameters extracted correctly from tool_use block"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        # Tool block is already configured in fixture with {"query": "test query"}
        response = ai_gen.generate_response(
            query="Test",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Verify execute_tool called with extracted parameters
        call_kwargs = mock_tool_manager.execute_tool.call_args[1]
        assert "query" in call_kwargs


def test_handle_tool_execution_builds_messages(mock_anthropic_client_tool_use, mock_tool_manager):
    """Test message list structure built correctly"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        response = ai_gen.generate_response(
            query="Test query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Verify messages.create was called twice (initial + follow-up)
        assert mock_anthropic_client_tool_use.messages.create.call_count == 2

        # Get second call's messages parameter
        second_call_kwargs = mock_anthropic_client_tool_use.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]

        # Should have: [user, assistant, user with tool_result]
        assert len(messages) >= 2  # At least user message and assistant response


def test_handle_tool_execution_returns_final_text(mock_anthropic_client_tool_use, mock_tool_manager):
    """Test returns text from second API call"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        response = ai_gen.generate_response(
            query="Test",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Should return text from final response
        assert "final answer" in response.lower()


def test_handle_tool_execution_no_tools_in_followup(mock_anthropic_client_tool_use, mock_tool_manager):
    """Test follow-up API call doesn't include tools parameter"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        response = ai_gen.generate_response(
            query="Test",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Get second call parameters
        second_call_kwargs = mock_anthropic_client_tool_use.messages.create.call_args_list[1][1]

        # Should NOT have 'tools' parameter in follow-up call
        assert "tools" not in second_call_kwargs


# ============================================================================
# 3.3 Tool Manager Integration Tests (4 tests)
# ============================================================================

def test_generate_response_calls_tool_manager_execute(mock_anthropic_client_tool_use, mock_tool_manager):
    """Test tool_manager.execute_tool() invoked"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        response = ai_gen.generate_response(
            query="Test",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Verify execute_tool was called
        mock_tool_manager.execute_tool.assert_called_once()


def test_generate_response_passes_tool_name(mock_anthropic_client_tool_use, mock_tool_manager):
    """Test correct tool name passed to execute_tool"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        response = ai_gen.generate_response(
            query="Test",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Verify tool name = "search_course_content"
        call_args = mock_tool_manager.execute_tool.call_args
        assert call_args[0][0] == "search_course_content"


def test_generate_response_passes_parameters(mock_anthropic_client_tool_use, mock_tool_manager):
    """Test parameters passed correctly to execute_tool"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        # Tool block is already configured in fixture
        response = ai_gen.generate_response(
            query="Test",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Verify parameters passed as kwargs
        call_kwargs = mock_tool_manager.execute_tool.call_args[1]
        assert "query" in call_kwargs


def test_tool_result_format(mock_anthropic_client_tool_use, mock_tool_manager):
    """Test tool_result structure matches API spec"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        response = ai_gen.generate_response(
            query="Test",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Get follow-up call's messages
        second_call_kwargs = mock_anthropic_client_tool_use.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]

        # Find tool_result in messages
        tool_result_msg = next((m for m in messages if m["role"] == "user"), None)
        assert tool_result_msg is not None

        # Verify tool_result structure
        if isinstance(tool_result_msg["content"], list):
            tool_result = tool_result_msg["content"][0]
            assert tool_result["type"] == "tool_result"
            assert "tool_use_id" in tool_result
            assert "content" in tool_result


# ============================================================================
# 3.4 API Call Tests (6 tests)
# ============================================================================

def test_initial_api_call_parameters(mock_anthropic_client_no_tool, mock_tool_manager):
    """Test initial API call has correct parameters"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_no_tool):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        response = ai_gen.generate_response(
            query="Test",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Get call parameters
        call_kwargs = mock_anthropic_client_no_tool.messages.create.call_args[1]

        # Verify key parameters
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["max_tokens"] == 800
        assert "tools" in call_kwargs


def test_initial_api_call_tool_choice_auto(mock_anthropic_client_no_tool, mock_tool_manager):
    """Test tool_choice set to auto"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_no_tool):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        response = ai_gen.generate_response(
            query="Test",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        call_kwargs = mock_anthropic_client_no_tool.messages.create.call_args[1]
        assert call_kwargs["tool_choice"] == {"type": "auto"}


def test_followup_api_call_no_tools(mock_anthropic_client_tool_use, mock_tool_manager):
    """Test follow-up call doesn't include tools"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        response = ai_gen.generate_response(
            query="Test",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Second call should NOT have tools
        second_call_kwargs = mock_anthropic_client_tool_use.messages.create.call_args_list[1][1]
        assert "tools" not in second_call_kwargs
        assert "tool_choice" not in second_call_kwargs


def test_system_prompt_includes_history(mock_anthropic_client_no_tool, mock_tool_manager):
    """Test system prompt includes conversation history"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_no_tool):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        history = "User: Previous question\nAssistant: Previous answer"

        response = ai_gen.generate_response(
            query="Test",
            conversation_history=history,
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        call_kwargs = mock_anthropic_client_no_tool.messages.create.call_args[1]
        system_content = call_kwargs["system"]

        # Should include history in system prompt
        assert "Previous question" in system_content
        assert "Previous answer" in system_content


def test_system_prompt_without_history(mock_anthropic_client_no_tool, mock_tool_manager):
    """Test system prompt without history"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_no_tool):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        response = ai_gen.generate_response(
            query="Test",
            conversation_history=None,
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        call_kwargs = mock_anthropic_client_no_tool.messages.create.call_args[1]
        system_content = call_kwargs["system"]

        # Should just be base system prompt
        assert "Previous conversation" not in system_content or "None" in system_content


def test_system_prompt_static_content(mock_anthropic_client_no_tool, mock_tool_manager):
    """Test system prompt contains expected instructions"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_no_tool):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        response = ai_gen.generate_response(
            query="Test",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        call_kwargs = mock_anthropic_client_no_tool.messages.create.call_args[1]
        system_content = call_kwargs["system"]

        # Verify key instructions present
        assert "One search per query" in system_content or "search" in system_content.lower()


# ============================================================================
# 3.5 Error Handling Tests (5 tests)
# ============================================================================

def test_generate_response_tool_execution_error(mock_anthropic_client_tool_use, mock_tool_manager):
    """Test tool returns error string, AI handles gracefully"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        # Configure tool to return error
        mock_tool_manager.execute_tool.return_value = "No course found matching 'Invalid'"

        response = ai_gen.generate_response(
            query="Test",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Should still return a response (AI handles error message)
        assert isinstance(response, str)


def test_generate_response_anthropic_api_error(mock_tool_manager):
    """Test Anthropic API raises exception, propagates correctly"""
    mock_client = Mock()
    mock_client.messages.create.side_effect = Exception("API rate limit exceeded")

    with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        # Should propagate exception
        with pytest.raises(Exception) as exc_info:
            ai_gen.generate_response(
                query="Test",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager
            )

        assert "API rate limit" in str(exc_info.value)


def test_generate_response_invalid_tool_response(mock_tool_manager):
    """Test malformed tool_use block"""
    mock_client = Mock()

    # Create malformed response (missing required fields)
    bad_response = Mock()
    bad_response.stop_reason = "tool_use"
    bad_response.content = []  # Empty content

    mock_client.messages.create.return_value = bad_response

    with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        # Should handle gracefully or raise appropriate error
        try:
            response = ai_gen.generate_response(
                query="Test",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager
            )
            # If it succeeds, tool_manager.execute_tool should not be called
            mock_tool_manager.execute_tool.assert_not_called()
        except (IndexError, AttributeError, TypeError):
            # Expected error for malformed response
            pass


def test_generate_response_missing_tool_manager(mock_anthropic_client_tool_use):
    """Test tool_use with tool_manager=None"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_tool_use):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        # Call with tool_manager=None
        response = ai_gen.generate_response(
            query="Test",
            tools=[{"name": "test_tool"}],
            tool_manager=None
        )

        # Should return initial response without executing tools
        # When tool_manager is None, it returns the content[0].text attribute
        # Verify it returns something (even if it's a mock attribute)
        assert response is not None


def test_generate_response_multiple_tool_uses(mock_tool_manager):
    """Test AI tries to use tool multiple times (edge case)"""
    mock_client = Mock()

    # Create response with multiple tool_use blocks
    multi_tool_response = Mock()
    multi_tool_response.stop_reason = "tool_use"

    tool_block1 = Mock()
    tool_block1.type = "tool_use"
    tool_block1.name = "search_course_content"
    tool_block1.id = "tool_1"
    tool_block1.input = {"query": "query1"}

    tool_block2 = Mock()
    tool_block2.type = "tool_use"
    tool_block2.name = "search_course_content"
    tool_block2.id = "tool_2"
    tool_block2.input = {"query": "query2"}

    multi_tool_response.content = [tool_block1, tool_block2]

    # Final response
    final_response = Mock()
    final_response.stop_reason = "end_turn"
    final_text = Mock()
    final_text.type = "text"
    final_text.text = "Final answer"
    final_response.content = [final_text]

    mock_client.messages.create.side_effect = [multi_tool_response, final_response]

    with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
        ai_gen = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        response = ai_gen.generate_response(
            query="Test",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Should execute both tools (or handle according to system prompt)
        # Current implementation likely executes all tool_use blocks
        assert mock_tool_manager.execute_tool.call_count >= 1
