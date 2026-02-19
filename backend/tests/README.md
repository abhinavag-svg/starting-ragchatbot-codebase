# RAG System Test Suite

## Overview

Comprehensive test suite for the RAG (Retrieval-Augmented Generation) chatbot system with **67 tests** across 3 test suites.

## Test Results

✅ **67/67 tests passing (100%)**

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test suite
uv run pytest tests/test_search_tools.py -v
uv run pytest tests/test_ai_generator.py -v
uv run pytest tests/test_rag_integration.py -v

# Run with coverage
uv run pytest tests/ --cov=. --cov-report=html --cov-report=term

# Run specific test
uv run pytest tests/test_search_tools.py::test_execute_query_only -v
```

## Test Suites

### 1. test_search_tools.py (23 tests)
Tests for `CourseSearchTool.execute()` method:
- **Parameter combinations**: Query only, +course, +lesson, all three
- **Error handling**: Invalid course, no results, search errors
- **Source tracking**: Verifies `last_sources` populated correctly
- **Result formatting**: Course/lesson context headers
- **Edge cases**: Empty queries, special characters, long queries

**Key Validations**:
- ✅ Source tracking works correctly
- ✅ Lesson links retrieved and included
- ✅ Error messages user-friendly

### 2. test_ai_generator.py (24 tests)
Tests for `AIGenerator` tool calling functionality:
- **Tool invocation decisions**: Course-specific → tool, general → no tool
- **Tool execution flow**: Multi-step API calls
- **Tool manager integration**: execute_tool() called correctly
- **API call structure**: Correct parameters, temperature=0, max_tokens=800
- **Error handling**: Tool errors, API errors

**Key Validations**:
- ✅ Tools invoked only for course-specific questions
- ✅ No tool overuse for general knowledge
- ✅ Tool execution flow works end-to-end

### 3. test_rag_integration.py (20 tests)
Integration tests for RAG system coordination:
- **Full query flow**: Query → tool use → response + sources
- **Source retrieval**: get_last_sources(), reset_sources()
- **Session management**: History passed, exchanges recorded
- **Component coordination**: Tools/tool_manager passed correctly
- **Error propagation**: Exceptions handled properly

**Key Validations**:
- ✅ End-to-end RAG flow works
- ✅ Sources retrieved and reset correctly
- ✅ Session management functional

## Coverage

| Component | Coverage | Status |
|-----------|----------|--------|
| ai_generator.py | 100% | ✅ Excellent |
| search_tools.py | 92% | ✅ Excellent |
| models.py | 100% | ✅ Excellent |
| config.py | 100% | ✅ Excellent |
| rag_system.py | 48% | ⚠️ Moderate |

*Note: Lower coverage for rag_system.py is expected - document loading code not tested.*

## Test Infrastructure

### Fixtures (conftest.py)
- `mock_vector_store`: Mock VectorStore with configurable SearchResults
- `mock_anthropic_client_tool_use`: Mock Anthropic with tool_use flow
- `mock_anthropic_client_no_tool`: Mock Anthropic with direct response
- `sample_search_results`: Pre-built SearchResults objects
- `sample_course_data`: Course/Lesson/CourseChunk test data
- `mock_tool_manager`: Mock ToolManager for integration tests
- `mock_session_manager`: Mock SessionManager for session tests

### Mock Strategy
- All tests use mocking to avoid external dependencies (ChromaDB, Anthropic API)
- Tests run quickly (<0.1 seconds total)
- Each test is isolated and can run independently

## Test Results Analysis

See `TEST_RESULTS_ANALYSIS.md` for detailed analysis including:
- Comprehensive test results breakdown
- Code coverage analysis
- Issues found (none!)
- Recommendations for future testing

## Key Findings

**All critical functionality verified**:
1. ✅ CourseSearchTool correctly tracks sources and retrieves lesson links
2. ✅ AIGenerator properly invokes tools for course-specific questions only
3. ✅ RAG system coordinates all components correctly
4. ✅ Session management and conversation history work
5. ✅ Error handling is robust throughout

**No bugs found in the actual RAG system code!**

## Adding New Tests

When adding new tests:

1. Add fixtures to `conftest.py` if needed
2. Follow naming convention: `test_{method}_{scenario}`
3. Use descriptive docstrings
4. Mock external dependencies (ChromaDB, Anthropic API)
5. Verify both success and failure paths

Example:
```python
def test_execute_new_scenario(mock_vector_store, sample_search_results):
    """Test description of what this validates"""
    tool = CourseSearchTool(mock_vector_store)

    # Configure mock behavior
    mock_vector_store.search.return_value = sample_search_results["success"]

    # Execute test
    result = tool.execute(query="test")

    # Verify expectations
    assert "expected" in result
    mock_vector_store.search.assert_called_once()
```

## Continuous Integration

Tests are designed to run in CI/CD pipelines:
- Fast execution (<1 second)
- No external dependencies required
- Deterministic results (no flaky tests)
- Clear failure messages
