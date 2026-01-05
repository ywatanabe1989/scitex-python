#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-02 16:05:00 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/decorators/test__deprecated.py

"""Tests for deprecated decorator functionality."""

import pytest
# Required for scitex.decorators module
pytest.importorskip("tqdm")
import warnings
import functools
from unittest.mock import patch


class TestDeprecated:
    """Test cases for scitex.decorators._deprecated module."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear any existing warnings filters
        warnings.resetwarnings()

    def test_deprecated_import(self):
        """Test that deprecated decorator can be imported successfully."""
        from scitex.decorators import deprecated
        assert callable(deprecated)

    def test_deprecated_basic_functionality(self):
        """Test basic deprecated decorator functionality."""
        from scitex.decorators import deprecated
        
        @deprecated("This function is old")
        def old_function(x):
            return x * 2
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_function(5)
            
            assert result == 10
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "old_function is deprecated: This function is old" in str(w[0].message)

    def test_deprecated_without_reason(self):
        """Test deprecated decorator without providing a reason."""
        from scitex.decorators import deprecated
        
        @deprecated()
        def no_reason_function():
            return "test"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = no_reason_function()
            
            assert result == "test"
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "no_reason_function is deprecated: None" in str(w[0].message)

    def test_deprecated_with_none_reason(self):
        """Test deprecated decorator with explicit None reason."""
        from scitex.decorators import deprecated
        
        @deprecated(None)
        def none_reason_function():
            return "test"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = none_reason_function()
            
            assert result == "test"
            assert len(w) == 1
            assert "none_reason_function is deprecated: None" in str(w[0].message)

    def test_deprecated_with_complex_reason(self):
        """Test deprecated decorator with complex reason string."""
        from scitex.decorators import deprecated
        
        complex_reason = "Use new_function() instead. This will be removed in v2.0. See documentation at example.com"
        
        @deprecated(complex_reason)
        def complex_function():
            return "deprecated"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            complex_function()
            
            assert len(w) == 1
            assert complex_reason in str(w[0].message)

    def test_deprecated_function_arguments(self):
        """Test deprecated decorator with functions that have arguments."""
        from scitex.decorators import deprecated
        
        @deprecated("Use new_math_function")
        def old_math_function(a, b, c=10):
            return a + b + c
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_math_function(1, 2, c=3)
            
            assert result == 6
            assert len(w) == 1
            assert "old_math_function is deprecated" in str(w[0].message)

    def test_deprecated_function_kwargs(self):
        """Test deprecated decorator with functions using *args and **kwargs."""
        from scitex.decorators import deprecated
        
        @deprecated("Flexible argument function deprecated")
        def flexible_function(*args, **kwargs):
            return sum(args) + sum(kwargs.values())
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = flexible_function(1, 2, 3, x=4, y=5)
            
            assert result == 15  # 1+2+3+4+5
            assert len(w) == 1
            assert "flexible_function is deprecated" in str(w[0].message)

    def test_deprecated_function_with_exceptions(self):
        """Test deprecated decorator when decorated function raises exceptions."""
        from scitex.decorators import deprecated
        
        @deprecated("This error function is deprecated")
        def error_function(should_raise=True):
            if should_raise:
                raise ValueError("Test error")
            return "success"
        
        # Test successful call
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = error_function(False)
            
            assert result == "success"
            assert len(w) == 1
            assert "error_function is deprecated" in str(w[0].message)
        
        # Test exception is still raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="Test error"):
                error_function(True)
            
            # Warning should still be emitted before exception
            assert len(w) == 1
            assert "error_function is deprecated" in str(w[0].message)

    def test_deprecated_multiple_calls(self):
        """Test that each call to deprecated function emits a warning."""
        from scitex.decorators import deprecated
        
        @deprecated("Multi-call test")
        def multi_call_function():
            return "called"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Call multiple times
            for i in range(3):
                result = multi_call_function()
                assert result == "called"
            
            # Should have one warning per call
            assert len(w) == 3
            for warning in w:
                assert "multi_call_function is deprecated" in str(warning.message)

    def test_deprecated_function_attributes_preserved(self):
        """Test that decorated function preserves original attributes."""
        from scitex.decorators import deprecated
        
        @deprecated("Function with docs")
        def documented_function(x, y=5):
            """This function adds two numbers.
            
            Args:
                x (int): First number
                y (int): Second number
                
            Returns:
                int: Sum of x and y
            """
            return x + y
        
        # Check that function attributes are preserved
        assert documented_function.__name__ == "documented_function"
        assert "adds two numbers" in documented_function.__doc__
        assert hasattr(documented_function, '__wrapped__')

    def test_deprecated_with_class_methods(self):
        """Test deprecated decorator with class methods."""
        from scitex.decorators import deprecated
        
        class TestClass:
            @deprecated("Method is deprecated")
            def old_method(self, value):
                return value * 2
            
            @deprecated("Static method deprecated") 
            @staticmethod
            def old_static_method(value):
                return value * 3
            
            @classmethod
            @deprecated("Class method deprecated")
            def old_class_method(cls, value):
                return value * 4
        
        obj = TestClass()
        
        # Test instance method
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = obj.old_method(5)
            assert result == 10
            assert len(w) == 1
            assert "old_method is deprecated" in str(w[0].message)
        
        # Test static method
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = TestClass.old_static_method(5)
            assert result == 15
            assert len(w) == 1
            assert "old_static_method is deprecated" in str(w[0].message)
        
        # Test class method
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = TestClass.old_class_method(5)
            assert result == 20
            assert len(w) == 1
            assert "old_class_method is deprecated" in str(w[0].message)

    def test_deprecated_warning_stacklevel(self):
        """Test that deprecated warnings have correct stack level."""
        from scitex.decorators import deprecated
        
        @deprecated("Stack level test")
        def stacklevel_function():
            return "test"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            def caller_function():
                return stacklevel_function()
            
            caller_function()
            
            assert len(w) == 1
            # The warning should point to the caller, not the decorator
            assert w[0].filename.endswith("test__deprecated.py")

    def test_deprecated_with_return_values(self):
        """Test deprecated decorator preserves all return value types."""
        from scitex.decorators import deprecated
        
        @deprecated("Returns list")
        def return_list():
            return [1, 2, 3]
        
        @deprecated("Returns dict")
        def return_dict():
            return {"key": "value"}
        
        @deprecated("Returns None")
        def return_none():
            return None
        
        @deprecated("Returns tuple")
        def return_tuple():
            return (1, 2, 3)
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            
            assert return_list() == [1, 2, 3]
            assert return_dict() == {"key": "value"}
            assert return_none() is None
            assert return_tuple() == (1, 2, 3)

    def test_deprecated_unicode_reason(self):
        """Test deprecated decorator with unicode characters in reason."""
        from scitex.decorators import deprecated
        
        unicode_reason = "Função obsoleta. Use função_nova() em vez disso. 废弃的函数"
        
        @deprecated(unicode_reason)
        def unicode_function():
            return "unicode test"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = unicode_function()
            
            assert result == "unicode test"
            assert len(w) == 1
            assert unicode_reason in str(w[0].message)

    def test_deprecated_very_long_reason(self):
        """Test deprecated decorator with very long reason string."""
        from scitex.decorators import deprecated
        
        long_reason = "This is a very long deprecation reason that explains in great detail why this function is deprecated and what alternatives should be used instead. " * 10
        
        @deprecated(long_reason)
        def long_reason_function():
            return "long reason test"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = long_reason_function()
            
            assert result == "long reason test"
            assert len(w) == 1
            assert long_reason in str(w[0].message)

    def test_deprecated_special_characters_in_reason(self):
        """Test deprecated decorator with special characters in reason."""
        from scitex.decorators import deprecated
        
        special_reason = "Function deprecated! Use new_func() -> str | None instead. Cost: $0.00"
        
        @deprecated(special_reason)
        def special_chars_function():
            return "special test"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = special_chars_function()
            
            assert result == "special test"
            assert len(w) == 1
            assert special_reason in str(w[0].message)

    def test_deprecated_with_generators(self):
        """Test deprecated decorator with generator functions."""
        from scitex.decorators import deprecated
        
        @deprecated("Generator function deprecated")
        def deprecated_generator(n):
            for i in range(n):
                yield i * 2
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gen = deprecated_generator(3)
            results = list(gen)
            
            assert results == [0, 2, 4]
            assert len(w) == 1
            assert "deprecated_generator is deprecated" in str(w[0].message)

    def test_deprecated_multiple_decorators(self):
        """Test deprecated decorator when combined with other decorators."""
        from scitex.decorators import deprecated
        
        def double_result(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return result * 2
            return wrapper
        
        @deprecated("Multi-decorator test")
        @double_result
        def multi_decorated_function(x):
            return x + 1
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = multi_decorated_function(5)
            
            assert result == 12  # (5 + 1) * 2
            assert len(w) == 1
            assert "multi_decorated_function is deprecated" in str(w[0].message)

    def test_deprecated_warning_category(self):
        """Test that deprecated decorator emits DeprecationWarning specifically."""
        from scitex.decorators import deprecated
        
        @deprecated("Category test")
        def category_test_function():
            return "test"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            category_test_function()
            
            assert len(w) == 1
            assert w[0].category == DeprecationWarning

    def test_deprecated_function_name_with_special_chars(self):
        """Test deprecated decorator with function names containing special characters."""
        from scitex.decorators import deprecated
        
        @deprecated("Special name test")
        def _private_function():
            return "private"
        
        @deprecated("Number name test")
        def func_2():
            return "numbered"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            _private_function()
            func_2()
            
            assert len(w) == 2
            assert "_private_function is deprecated" in str(w[0].message)
            assert "func_2 is deprecated" in str(w[1].message)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/decorators/_deprecated.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-08-21 20:57:29 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/decorators/_deprecated.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# import functools
# import importlib
# import warnings
# 
# 
# def deprecated(reason=None, forward_to=None):
#     """
#     A decorator to mark functions as deprecated. It will result in a warning being emitted
#     when the function is used.
# 
#     Args:
#         reason (str): A human-readable string explaining why this function was deprecated.
#         forward_to (str): Optional module path to forward calls to (e.g., "..session.start").
#                          If provided, calls will be forwarded to the new function instead of
#                          executing the original deprecated function.
#     """
# 
#     def decorator(func):
#         if forward_to:
#             # Create a forwarding wrapper with auto-generated docstring
#             @functools.wraps(func)
#             def new_func(*args, **kwargs):
#                 warnings.warn(
#                     f"{func.__name__} is deprecated: {reason}",
#                     DeprecationWarning,
#                     stacklevel=2,
#                 )
#                 # Dynamic import and call forwarding
#                 module_path, function_name = forward_to.rsplit(".", 1)
# 
#                 # Handle relative imports
#                 if module_path.startswith(".."):
#                     # Get the module where the function was defined (not the calling module)
#                     func_module = func.__module__
# 
#                     if func_module:
#                         # Convert relative import to absolute based on the function's module
#                         package_parts = func_module.split(".")
#                         # Count the number of dots to determine how many levels to go up
#                         level_count = 0
#                         for char in module_path:
#                             if char == ".":
#                                 level_count += 1
#                             else:
#                                 break
# 
#                         # Remove the relative part and create absolute path
#                         if level_count > 0:
#                             base_package_parts = package_parts[:-level_count]
#                             if base_package_parts:
#                                 base_package = ".".join(base_package_parts)
#                                 relative_part = module_path.lstrip(".")
#                                 module_path = (
#                                     base_package + "." + relative_part
#                                     if relative_part
#                                     else base_package
#                                 )
#                             else:
#                                 # Can't go up that many levels, fallback to absolute
#                                 module_path = module_path.lstrip(".")
# 
#                 try:
#                     target_module = importlib.import_module(module_path)
#                     target_function = getattr(target_module, function_name)
#                     return target_function(*args, **kwargs)
#                 except (ImportError, AttributeError) as e:
#                     # Fallback to original function if forwarding fails
#                     warnings.warn(
#                         f"Failed to forward {func.__name__} to {forward_to}: {e}. "
#                         f"Using original deprecated implementation.",
#                         RuntimeWarning,
#                         stacklevel=2,
#                     )
#                     return func(*args, **kwargs)
# 
#             # Auto-generate docstring for forwarding wrapper with target function's docstring
#             original_name = func.__name__
#             new_location = forward_to.replace("..", "scitex.").lstrip(".")
# 
#             # Try to get the target function's docstring
#             target_docstring = ""
#             try:
#                 # Get the same target we'll forward to
#                 target_module_path, target_function_name = forward_to.rsplit(".", 1)
# 
#                 # Handle relative imports for docstring retrieval
#                 if target_module_path.startswith(".."):
#                     func_module = func.__module__
#                     if func_module:
#                         package_parts = func_module.split(".")
#                         level_count = 0
#                         for char in target_module_path:
#                             if char == ".":
#                                 level_count += 1
#                             else:
#                                 break
# 
#                         if level_count > 0:
#                             base_package_parts = package_parts[:-level_count]
#                             if base_package_parts:
#                                 base_package = ".".join(base_package_parts)
#                                 relative_part = target_module_path.lstrip(".")
#                                 target_module_path = (
#                                     base_package + "." + relative_part
#                                     if relative_part
#                                     else base_package
#                                 )
#                             else:
#                                 target_module_path = target_module_path.lstrip(".")
# 
#                 target_module = importlib.import_module(target_module_path)
#                 target_function = getattr(target_module, target_function_name)
#                 if target_function.__doc__:
#                     target_docstring = target_function.__doc__.strip()
#             except (ImportError, AttributeError):
#                 pass  # Fall back to basic docstring if target can't be imported
# 
#             # Create comprehensive docstring combining deprecation notice with target docs
#             if target_docstring:
#                 forwarding_docstring = f"""**DEPRECATED: Use {new_location} instead**
# 
# {target_docstring}
# 
# Deprecation Notice
# ------------------
# This function is deprecated and will be removed in a future version.
# Use `{new_location}` instead. This wrapper forwards all calls to the new function
# while displaying a deprecation warning.
# 
# Parameters
# ----------
# *args : tuple
#     Positional arguments passed to {new_location}
# **kwargs : dict 
#     Keyword arguments passed to {new_location}
#     
# Returns
# -------
# Any
#     Same return value as {new_location}
#     
# Warns
# -----
# DeprecationWarning
#     Always warns that this function is deprecated
# """
#             else:
#                 # Fallback if target docstring unavailable
#                 forwarding_docstring = f"""**DEPRECATED: Use {new_location} instead**
#     
# This function provides backward compatibility for existing code that uses
# {original_name}(). It forwards all calls to the new {new_location}
# function while displaying a deprecation warning.
# 
# Parameters
# ----------
# *args : tuple
#     Positional arguments passed to {new_location}
# **kwargs : dict 
#     Keyword arguments passed to {new_location}
#     
# Returns
# -------
# Any
#     Same return value as {new_location}
#     
# Warns
# -----
# DeprecationWarning
#     Always warns that this function is deprecated
# """
#             new_func.__doc__ = forwarding_docstring
#             return new_func
#         else:
#             # Original behavior for non-forwarding deprecation
#             @functools.wraps(func)
#             def new_func(*args, **kwargs):
#                 warnings.warn(
#                     f"{func.__name__} is deprecated: {reason}",
#                     DeprecationWarning,
#                     stacklevel=2,
#                 )
#                 return func(*args, **kwargs)
# 
#             return new_func
# 
#     return decorator
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/decorators/_deprecated.py
# --------------------------------------------------------------------------------
