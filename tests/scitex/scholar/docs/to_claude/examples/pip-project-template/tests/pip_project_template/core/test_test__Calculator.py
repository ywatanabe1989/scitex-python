# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/tests/pip_project_template/core/test__Calculator.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-08-27 05:50:55 (ywatanabe)"
# # File: /home/ywatanabe/proj/pip-project-template/tests/pip_project_template/core/test__Calculator.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# __FILE__ = (
#     "./tests/pip_project_template/core/test__Calculator.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# # Test file for src/core/_Calculator.py
# 
# import sys
# from pathlib import Path
# 
# import pytest
# 
# # Add src to path
# sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
# 
# from pip_project_template.core._Calculator import Calculator
# 
# 
# class TestCalculator:
#     """Test suite for core._Calculator"""
# 
#     def setup_method(self):
#         """Set up test fixtures."""
#         self.calculator = Calculator()
# 
#     def test_import(self):
#         """Test that module imports successfully."""
#         from pip_project_template.core._Calculator import \
#             Calculator  # noqa: F401
# 
#         assert True  # Module imported
# 
#     def test_calculator_instantiation(self):
#         """Test Calculator can be instantiated."""
#         calc = Calculator()
#         assert isinstance(calc, Calculator)
# 
#     def test_add_operation(self):
#         """Test addition operation."""
#         result = self.calculator.calculate(2.5, 3.0, "add")
#         assert result == 5.5
# 
#     def test_add_operation_default(self):
#         """Test addition is the default operation."""
#         result = self.calculator.calculate(10, 5)
#         assert result == 15
# 
#     def test_multiply_operation(self):
#         """Test multiplication operation."""
#         result = self.calculator.calculate(4, 2.5, "multiply")
#         assert result == 10.0
# 
#     def test_negative_numbers(self):
#         """Test operations with negative numbers."""
#         assert self.calculator.calculate(-5, 3, "add") == -2
#         assert self.calculator.calculate(-2, -3, "multiply") == 6
# 
#     def test_zero_operations(self):
#         """Test operations with zero."""
#         assert self.calculator.calculate(0, 5, "add") == 5
#         assert self.calculator.calculate(7, 0, "multiply") == 0
# 
#     def test_invalid_operation(self):
#         """Test that invalid operation raises ValueError."""
#         with pytest.raises(ValueError) as exc_info:
#             self.calculator.calculate(1, 2, "subtract")
#         assert "Unknown operation: subtract" in str(exc_info.value)
# 
#     def test_large_numbers(self):
#         """Test operations with large numbers."""
#         large_num = 1000000.0
#         result = self.calculator.calculate(large_num, large_num, "add")
#         assert result == 2000000.0
# 
#     def test_floating_point_precision(self):
#         """Test floating point operations."""
#         result = self.calculator.calculate(0.1, 0.2, "add")
#         assert (
#             abs(result - 0.3) < 1e-10
#         )  # Account for floating point precision
# 
#     def test_main_function(self):
#         """Test the main function execution."""
#         from unittest.mock import patch
# 
#         from pip_project_template.core._Calculator import main
# 
#         # Test the main function with mocked print
#         with patch("builtins.print") as mock_print:
#             main()
#             # Verify print was called with expected results
#             assert mock_print.call_count == 2
#             # Check the calls contain the expected results
#             calls = [call.args[0] for call in mock_print.call_args_list]
#             assert 15 in calls  # 10 + 5
#             assert 12 in calls  # 3 * 4
# 
#     def test_module_execution(self):
#         """Test module execution as script."""
#         import subprocess
#         import sys
# 
#         # Test the module can be executed as a script
#         result = subprocess.run(
#             [sys.executable, "-m", "pip_project_template.core._Calculator"],
#             capture_output=True,
#             text=True,
#         )
# 
#         # Should exit with code 0 and print the results
#         assert result.returncode == 0
#         assert "15" in result.stdout  # 10 + 5
#         assert "12" in result.stdout  # 3 * 4
# 
# 
# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/tests/pip_project_template/core/test__Calculator.py
# --------------------------------------------------------------------------------
