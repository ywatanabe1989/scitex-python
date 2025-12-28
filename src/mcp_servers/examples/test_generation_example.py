#!/usr/bin/env python3
"""
Example: Using the Developer MCP Server for Test Generation

This example demonstrates how to use the test generation capabilities
to create comprehensive test suites for your SciTeX scripts.
"""

# Example 1: Simple function that needs tests
def calculate_statistics(data):
    """Calculate mean and standard deviation of data."""
    import numpy as np
    
    if len(data) == 0:
        raise ValueError("Empty data array")
    
    mean = np.mean(data)
    std = np.std(data)
    
    return {"mean": mean, "std": std}


# Example 2: Class that needs tests
class DataProcessor:
    """Process and transform data arrays."""
    
    def __init__(self, config=None):
        self.config = config or {"threshold": 0.5}
        self.processed_count = 0
    
    def filter_data(self, data, threshold=None):
        """Filter data above threshold."""
        threshold = threshold or self.config["threshold"]
        filtered = [x for x in data if x > threshold]
        self.processed_count += 1
        return filtered
    
    def normalize_data(self, data):
        """Normalize data to 0-1 range."""
        if not data:
            return []
        
        min_val = min(data)
        max_val = max(data)
        
        if min_val == max_val:
            return [0.5] * len(data)
        
        return [(x - min_val) / (max_val - min_val) for x in data]


# Example 3: Integration workflow that needs tests
def analysis_workflow(input_file, output_file):
    """Complete analysis workflow."""
    import pandas as pd
    import scitex as stx
    
    # Load data
    data = stx.io.load(input_file)
    
    # Process data
    processor = DataProcessor()
    filtered = processor.filter_data(data['values'])
    normalized = processor.normalize_data(filtered)
    
    # Calculate statistics
    stats = calculate_statistics(normalized)
    
    # Save results
    results = {
        "processed_data": normalized,
        "statistics": stats,
        "processing_info": {
            "input_count": len(data),
            "output_count": len(normalized),
            "threshold": processor.config["threshold"]
        }
    }
    
    stx.io.save(results, output_file)
    return results


# What the Developer MCP Server would generate:
print("""
=== Test Generation with Developer MCP Server ===

Request: "Generate comprehensive tests for this data processing module"

The server would create:

1. test_calculate_statistics.py:
   - Test normal operation with various data
   - Test edge cases (empty array, single value)
   - Test error handling
   - Property-based tests for invariants

2. test_data_processor.py:
   - Test initialization with/without config
   - Test filter_data with various thresholds
   - Test normalize_data edge cases
   - Test state management (processed_count)
   - Mock configuration tests

3. test_analysis_workflow.py:
   - Integration tests with real files
   - Mock file I/O for unit tests
   - Test error propagation
   - Test output format validation
   - Performance benchmarks

4. conftest.py (pytest fixtures):
   - Sample data fixtures
   - Temporary file fixtures
   - Configuration fixtures
   - Mock objects

5. Coverage report:
   - Line coverage: 95%
   - Branch coverage: 90%
   - Missing coverage analysis
   - Suggestions for improvement

Example generated test:

```python
import pytest
import numpy as np
from your_module import calculate_statistics

class TestCalculateStatistics:
    def test_normal_operation(self):
        data = [1, 2, 3, 4, 5]
        result = calculate_statistics(data)
        assert result["mean"] == 3.0
        assert abs(result["std"] - 1.414) < 0.001
    
    def test_empty_array_raises_error(self):
        with pytest.raises(ValueError, match="Empty data array"):
            calculate_statistics([])
    
    @pytest.mark.parametrize("data,expected_mean", [
        ([1], 1.0),
        ([1, 1, 1], 1.0),
        ([-1, 0, 1], 0.0),
        (np.random.randn(100), None)  # Statistical test
    ])
    def test_various_inputs(self, data, expected_mean):
        result = calculate_statistics(data)
        if expected_mean is not None:
            assert result["mean"] == expected_mean
        else:
            # Statistical test for random data
            assert -0.5 < result["mean"] < 0.5
```

The tests ensure:
- ✅ All functions work correctly
- ✅ Edge cases are handled
- ✅ Errors are raised appropriately
- ✅ Integration works end-to-end
- ✅ Performance meets requirements
""")

# Example interaction with the server
print("""
=== Interactive Test Development ===

You: "Generate tests for the DataProcessor class"

Server: "I'll generate comprehensive tests for DataProcessor. What testing framework do you prefer?"

You: "Use pytest with fixtures"

Server: "Generated test_data_processor.py with:
- 8 test methods covering all functionality
- 3 fixtures for common test data
- Parametrized tests for edge cases
- Mock tests for configuration
- 98% code coverage

Would you like me to:
1. Add property-based tests using Hypothesis?
2. Include performance benchmarks?
3. Generate integration tests with the workflow?"

You: "Add performance benchmarks"

Server: "Added performance benchmarks:
- test_filter_performance_large_dataset
- test_normalize_performance_scaling
- Baseline timings included
- Memory usage profiling

The tests now ensure both correctness and performance!"
""")

if __name__ == "__main__":
    print("\nDeveloper MCP Server: Making testing easy and comprehensive!")
    print("No more excuses for untested code! 🧪✨")