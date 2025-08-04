# Travel API Automated Test Suite

This directory contains a comprehensive automated test suite for the Travel API, covering all endpoints with various scenarios including happy path, edge cases, and error conditions.

## 📁 Files Overview

- `test_api.py` - Main test suite with all test cases
- `test_requirements.txt` - Test-specific dependencies
- `run_tests.py` - Test runner script with various options
- `pytest.ini` - Pytest configuration file
- `TEST_README.md` - This documentation file

## 🚀 Quick Start

### 1. Install Test Dependencies

```bash
pip install -r test_requirements.txt
```

### 2. Run All Tests

```bash
# Using the test runner script
python run_tests.py

# Or directly with pytest
python -m pytest test_api.py -v
```

### 3. Run Specific Test Categories

```bash
# Run only basic tests (health, root endpoints)
python run_tests.py -m basic

# Run only hotel search tests
python run_tests.py -m hotels

# Run only flight search tests
python run_tests.py -m flights

# Run only airport search tests
python run_tests.py -m airports

# Run only PredictHQ events tests
python run_tests.py -m predicthq

# Run only quick tests (no external API calls)
python run_tests.py --quick
```

## 🧪 Test Categories

The test suite is organized into the following categories:

| Category | Description | Markers |
|----------|-------------|---------|
| **Basic** | Health check and root endpoints | `@pytest.mark.basic` |
| **Categorization** | Bank transaction categorization | `@pytest.mark.categorization` |
| **Hotels** | Hotel search functionality | `@pytest.mark.hotels` |
| **Flights** | Flight search functionality | `@pytest.mark.flights` |
| **Trip Planning** | Complete trip planning | `@pytest.mark.trip_planning` |
| **Airbnb** | Airbnb search and details | `@pytest.mark.airbnb` |
| **Airports** | Airport search functionality | `@pytest.mark.airports` |
| **PredictHQ** | Events and activities search | `@pytest.mark.predicthq` |
| **Itinerary** | Location-based itinerary | `@pytest.mark.itinerary` |
| **Performance** | Performance benchmarks | `@pytest.mark.performance` |
| **Integration** | End-to-end integration tests | `@pytest.mark.integration` |

## 🎯 Test Types

| Type | Description | Markers |
|------|-------------|---------|
| **Unit** | Individual function/endpoint tests | `@pytest.mark.unit` |
| **Integration** | Multi-component tests | `@pytest.mark.integration` |
| **Quick** | Fast tests (no external calls) | `@pytest.mark.quick` |
| **Slow** | Tests with external API calls | `@pytest.mark.slow` |
| **E2E** | End-to-end user scenarios | `@pytest.mark.e2e` |

## 🛠️ Test Runner Options

### Basic Usage

```bash
# Run all tests with verbose output
python run_tests.py -v

# Run with coverage report
python run_tests.py --coverage

# Run with HTML coverage report
python run_tests.py --coverage --html-report

# Run tests in parallel
python run_tests.py --parallel

# Exit on first failure
python run_tests.py -x
```

### Advanced Usage

```bash
# Run specific test categories interactively
python run_tests.py --categories

# Run only quick tests
python run_tests.py --quick

# Run with custom markers
python run_tests.py -m "hotels and unit"

# Show local variables on failures
python run_tests.py -l
```

## 📊 Coverage Reports

The test suite generates coverage reports to help identify untested code:

```bash
# Generate coverage report
python run_tests.py --coverage

# Generate HTML coverage report
python run_tests.py --coverage --html-report
```

HTML reports are saved in the `htmlcov/` directory and can be opened in a web browser.

## 🔧 Test Configuration

### Pytest Configuration (`pytest.ini`)

- **Test Discovery**: Automatically finds test files starting with `test_`
- **Markers**: Defines test categories and types
- **Coverage**: Minimum 70% coverage required
- **Timeouts**: 300-second timeout for slow tests
- **Output**: Colored, verbose output with short tracebacks

### Environment Variables

Some tests may require environment variables:

```bash
# For itinerary tests (Foursquare API)
export FOURSQUARE_API_KEY="your_api_key"

# For PredictHQ events
export PREDICTHQ_ACCESS_TOKEN="your_access_token"
```

## 🧪 Test Structure

### Fixtures

The test suite uses pytest fixtures for reusable test data:

- `sample_transactions` - Bank transaction data
- `sample_hotel_search` - Hotel search parameters
- `sample_flight_search` - Flight search parameters
- `sample_trip_plan` - Trip planning parameters
- `sample_airbnb_search` - Airbnb search parameters

### Mocking

External dependencies are mocked to ensure fast, reliable tests:

- Hotel search APIs
- Flight search APIs
- Airbnb APIs
- Foursquare API
- PredictHQ API

### Test Scenarios

Each endpoint is tested with multiple scenarios:

1. **Happy Path** - Valid requests with expected responses
2. **Edge Cases** - Boundary conditions and unusual inputs
3. **Error Handling** - Invalid requests and error responses
4. **Validation** - Input validation and data type checking
5. **Performance** - Response time and resource usage

## 📈 Test Results

### Success Indicators

- ✅ All tests pass
- 📊 Coverage above 70%
- ⚡ Performance within acceptable limits
- 🔒 No security vulnerabilities

### Common Issues

- **Import Errors**: Missing dependencies
- **Mock Failures**: External API changes
- **Timeout Errors**: Slow network or API responses
- **Validation Errors**: API schema changes

## 🚨 Troubleshooting

### Common Problems

1. **Import Errors**
   ```bash
   pip install -r test_requirements.txt
   ```

2. **Mock Failures**
   - Check if external APIs have changed
   - Update mock responses if needed

3. **Timeout Errors**
   - Increase timeout in `pytest.ini`
   - Check network connectivity

4. **Coverage Issues**
   - Add tests for uncovered code paths
   - Review and update test cases

### Debug Mode

```bash
# Run with detailed output
python run_tests.py -v -l

# Run single test
python -m pytest test_api.py::test_health_check -v

# Run with debugger
python -m pytest test_api.py --pdb
```

## 🔄 Continuous Integration

The test suite is designed to work with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    pip install -r test_requirements.txt
    python run_tests.py --coverage --html-report
```

## 📝 Adding New Tests

### Test Naming Convention

- Test functions: `test_<endpoint>_<scenario>`
- Test files: `test_<module>.py`
- Test classes: `Test<Module>`

### Adding Test Markers

```python
@pytest.mark.hotels
@pytest.mark.unit
def test_hotel_search_new_feature():
    """Test new hotel search feature."""
    # Test implementation
```

### Best Practices

1. **Use descriptive test names**
2. **Test one thing per test function**
3. **Use appropriate markers**
4. **Mock external dependencies**
5. **Include edge cases**
6. **Add performance tests for critical paths**

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing Guide](https://fastapi.tiangolo.com/tutorial/testing/)
- [Python Mocking Guide](https://docs.python.org/3/library/unittest.mock.html)

## 🤝 Contributing

When adding new features to the API:

1. **Write tests first** (TDD approach)
2. **Ensure all tests pass**
3. **Maintain coverage above 70%**
4. **Update this documentation**

---

**Note**: This test suite is designed to be comprehensive and maintainable. Regular updates ensure the API remains reliable and well-tested. 