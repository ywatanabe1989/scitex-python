# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/tests/pip_project_template/utils/test__multiply.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Test file for src/utils/_multiply.py
# 
# import pytest
# import sys
# from pathlib import Path
# 
# # Add src to path
# sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
# 
# from pip_project_template.utils._multiply import multiply
# 
# 
# class TestMultiply:
#     """Test suite for utils._multiply"""
# 
#     def test_import(self):
#         """Test that module imports successfully."""
#         from pip_project_template.utils._multiply import multiply  # noqa: F401
#         assert True  # Module imported
# 
#     def test_positive_numbers(self):
#         """Test multiplication of positive numbers."""
#         assert multiply(2, 3) == 6
#         assert multiply(1.5, 2) == 3.0
#         assert multiply(0.5, 4) == 2.0
# 
#     def test_negative_numbers(self):
#         """Test multiplication with negative numbers."""
#         assert multiply(-2, 3) == -6
#         assert multiply(2, -3) == -6
#         assert multiply(-2, -3) == 6
# 
#     def test_zero(self):
#         """Test multiplication with zero."""
#         assert multiply(0, 5) == 0
#         assert multiply(5, 0) == 0
#         assert multiply(0, 0) == 0
# 
#     def test_one(self):
#         """Test multiplication with one."""
#         assert multiply(1, 5) == 5
#         assert multiply(5, 1) == 5
#         assert multiply(1, 1) == 1
# 
#     def test_floating_point(self):
#         """Test multiplication with floating point numbers."""
#         result = multiply(0.1, 0.3)
#         assert abs(result - 0.03) < 1e-10  # Account for floating point precision
# 
#     def test_large_numbers(self):
#         """Test multiplication with large numbers."""
#         large_num = 1000000.0
#         assert multiply(large_num, 2) == 2000000.0
# 
#     def test_fractions(self):
#         """Test multiplication with fractional numbers."""
#         assert multiply(0.25, 4) == 1.0
#         assert multiply(2.5, 0.4) == 1.0
# 
#     def test_main_function(self):
#         """Test the main function execution."""
#         from unittest.mock import patch
#         from pip_project_template.utils._multiply import main
#         
#         with patch('builtins.print') as mock_print:
#             main()
#             mock_print.assert_called_once_with("4 * 5 = 20")
# 
#     def test_module_execution(self):
#         """Test module execution as script."""
#         import subprocess
#         import sys
#         
#         result = subprocess.run([
#             sys.executable, "-m", "pip_project_template.utils._multiply"
#         ], capture_output=True, text=True)
#         
#         assert result.returncode == 0
#         assert "4 * 5 = 20" in result.stdout
# 
# 
# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/tests/pip_project_template/utils/test__multiply.py
# --------------------------------------------------------------------------------
