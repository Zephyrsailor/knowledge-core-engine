# Integration Tests

This directory contains integration tests that interact with real external services.

## Running Integration Tests

By default, integration tests are skipped to avoid unnecessary API calls and costs.

To run integration tests:

```bash
# Set environment variable
export INTEGRATION_TESTS=true

# Set API keys
export DEEPSEEK_API_KEY=your_deepseek_key
export DASHSCOPE_API_KEY=your_dashscope_key  # For Qwen

# Run all integration tests
pytest tests/integration/

# Run specific provider tests
pytest tests/integration/test_enhancement_with_llm.py::TestDeepSeekIntegration
pytest tests/integration/test_enhancement_with_llm.py::TestQwenIntegration
```

## Test Coverage

- **test_enhancement_with_llm.py**: Tests metadata enhancement with real LLM providers
  - DeepSeek integration
  - Qwen integration
  - Performance comparison
  - Error handling and rate limiting

## Important Notes

1. **Costs**: Running these tests will incur API costs
2. **Rate Limits**: Tests may be rate-limited by providers
3. **API Keys**: Ensure your API keys have sufficient quota
4. **Network**: Requires stable internet connection

## Manual Testing

You can also run specific tests manually:

```bash
# Test DeepSeek enhancement
python tests/integration/test_enhancement_with_llm.py deepseek

# Test Qwen enhancement
python tests/integration/test_enhancement_with_llm.py qwen
```