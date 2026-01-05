# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/tests/pip_project_template/utils/test__add.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Test file for src/utils/_add.py
# 
# import pytest
# import sys
# from pathlib import Path
# 
# # Add src to path
# sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
# 
# from pip_project_template.utils._add import add
# 
# 
# class TestAdd:
#     """Test suite for utils._add"""
# 
#     def test_import(self):
#         """Test that module imports successfully."""
#         from pip_project_template.utils._add import add  # noqa: F401
#         assert True  # Module imported
# 
#     def test_positive_numbers(self):
#         """Test addition of positive numbers."""
#         assert add(2, 3) == 5
#         assert add(1.5, 2.5) == 4.0
# 
#     def test_negative_numbers(self):
#         """Test addition with negative numbers."""
#         assert add(-2, -3) == -5
#         assert add(-5, 3) == -2
#         assert add(5, -3) == 2
# 
#     def test_zero(self):
#         """Test addition with zero."""
#         assert add(0, 5) == 5
#         assert add(5, 0) == 5
#         assert add(0, 0) == 0
# 
#     def test_floating_point(self):
#         """Test addition with floating point numbers."""
#         result = add(0.1, 0.2)
#         assert abs(result - 0.3) < 1e-10  # Account for floating point precision
# 
#     def test_large_numbers(self):
#         """Test addition with large numbers."""
#         large_num = 1000000000.0
#         assert add(large_num, large_num) == 2000000000.0
# 
#     def test_very_small_numbers(self):
#         """Test addition with very small numbers."""
#         small_num = 1e-10
#         result = add(small_num, small_num)
#         assert abs(result - 2e-10) < 1e-15
# 
#     def test_main_function(self):
#         """Test the main function execution."""
#         from unittest.mock import patch
#         from pip_project_template.utils._add import main
#         
#         with patch('builtins.print') as mock_print:
#             main()
#             mock_print.assert_called_once_with("2 + 3 = 5")
# 
#     def test_module_execution(self):
#         """Test module execution as script."""
#         import subprocess
#         import sys
#         
#         result = subprocess.run([
#             sys.executable, "-m", "pip_project_template.utils._add"
#         ], capture_output=True, text=True)
#         
#         assert result.returncode == 0
#         assert "2 + 3 = 5" in result.stdout
# 
# 
# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/tests/pip_project_template/utils/test__add.py
# --------------------------------------------------------------------------------
