#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-02 15:36:00 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/plt/utils/test__close.py

"""Tests for close functionality."""

import pytest
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock, Mock

from scitex.plt.utils import close


class TestClose:
    """Test close function."""

    @patch('scitex.plt.utils._close.plt.close')
    def test_close_matplotlib_figure(self, mock_plt_close):
        """Test closing a standard matplotlib Figure object."""
        # Create a mock matplotlib figure
        mock_figure = Mock(spec=matplotlib.figure.Figure)
        
        # Call close function
        close(mock_figure)
        
        # Verify plt.close was called with the figure
        mock_plt_close.assert_called_once_with(mock_figure)
        
    @patch('scitex.plt.utils._close.plt.close')
    def test_close_scitex_figwrapper(self, mock_plt_close):
        """Test closing an SciTeX FigWrapper object."""
        # Create a mock FigWrapper with _fig_mpl attribute (new implementation)
        mock_figwrapper = MagicMock()
        mock_figwrapper._fig_mpl = Mock(spec=matplotlib.figure.Figure)
        # Make sure it doesn't match matplotlib.figure.Figure isinstance
        mock_figwrapper.__class__ = type("FigWrapper", (), {})

        # Call close function
        close(mock_figwrapper)

        # Verify plt.close was called with the underlying figure
        mock_plt_close.assert_called_once_with(mock_figwrapper._fig_mpl)
            
    def test_close_invalid_object_type_error(self):
        """Test that TypeError is raised for invalid object types."""
        # Create objects that don't have _fig_mpl or figure attributes
        class PlainObject:
            pass

        invalid_objects = [
            "string",
            123,
            [],
            {},
            None,
            PlainObject(),
        ]

        for invalid_obj in invalid_objects:
            with pytest.raises(TypeError) as exc_info:
                close(invalid_obj)

            # Check error message content
            error_msg = str(exc_info.value)
            assert "Cannot close object of type" in error_msg
            assert "Expected FigWrapper or Figure object" in error_msg
            assert type(invalid_obj).__name__ in error_msg
            
    @patch('scitex.plt.utils._close.plt.close')
    def test_close_real_matplotlib_figure(self, mock_plt_close):
        """Test closing with a real matplotlib figure (using mock to avoid GUI)."""
        # Use matplotlib's Figure class directly
        from matplotlib.figure import Figure
        
        # Create actual figure instance (without GUI backend)
        fig = Figure(figsize=(6, 4))
        
        # Call close function
        close(fig)
        
        # Verify plt.close was called
        mock_plt_close.assert_called_once_with(fig)
        
    def test_close_type_checking_specificity(self):
        """Test that type checking is specific and correct."""
        # Test with object that inherits from Figure
        class CustomFigure(matplotlib.figure.Figure):
            def __init__(self):
                # Don't call super().__init__ to avoid matplotlib setup
                pass
                
        with patch('scitex.plt.utils._close.plt.close') as mock_plt_close:
            custom_fig = CustomFigure()
            close(custom_fig)
            mock_plt_close.assert_called_once_with(custom_fig)
            
    def test_close_error_message_formatting(self):
        """Test error message formatting for different object types."""
        # Create plain objects that don't have _fig_mpl or figure attributes
        class StringWrapper:
            pass

        class IntWrapper:
            pass

        class ListWrapper:
            pass

        class DictWrapper:
            pass

        test_objects = [
            (StringWrapper(), "StringWrapper"),
            (IntWrapper(), "IntWrapper"),
            (ListWrapper(), "ListWrapper"),
            (DictWrapper(), "DictWrapper"),
        ]

        for obj, expected_type_name in test_objects:
            with pytest.raises(TypeError) as exc_info:
                close(obj)

            error_msg = str(exc_info.value)
            assert f"object of type {expected_type_name}" in error_msg
            
    @patch('scitex.plt.utils._close.plt.close')
    def test_close_memory_management_pattern(self, mock_plt_close):
        """Test close function in typical memory management pattern."""
        # Simulate creating multiple figures and closing them
        figures = []
        
        for i in range(5):
            mock_fig = Mock(spec=matplotlib.figure.Figure)
            figures.append(mock_fig)
            close(mock_fig)
            
        # Verify all figures were closed
        assert mock_plt_close.call_count == 5
        for i, fig in enumerate(figures):
            assert mock_plt_close.call_args_list[i][0][0] is fig
            
    def test_close_function_signature(self):
        """Test that close function has the expected signature."""
        import inspect
        
        sig = inspect.signature(close)
        params = list(sig.parameters.keys())
        
        # Should have exactly one parameter named 'obj'
        assert len(params) == 1
        assert params[0] == 'obj'
        
        # Parameter should not have a default value
        assert sig.parameters['obj'].default == inspect.Parameter.empty
        
    @patch('scitex.plt.utils._close.plt.close')
    def test_close_exception_propagation(self, mock_plt_close):
        """Test that exceptions from plt.close are propagated."""
        # Make plt.close raise an exception
        mock_plt_close.side_effect = RuntimeError("Mock error from plt.close")
        
        mock_figure = Mock(spec=matplotlib.figure.Figure)
        
        # close should propagate the exception
        with pytest.raises(RuntimeError, match="Mock error from plt.close"):
            close(mock_figure)
            
    def test_close_imports_availability(self):
        """Test that required imports are available."""
        # Test that the function can access required modules
        import scitex.plt.utils._close as close_module

        assert hasattr(close_module, 'matplotlib')
        assert hasattr(close_module, 'plt')
        
    @patch('scitex.plt.utils._close.plt.close')
    def test_close_with_none_figure_attribute(self, mock_plt_close):
        """Test handling when object has None figure attribute."""
        # Create mock object with figure attribute set to None
        # (uses the hasattr(obj, "figure") path)
        mock_figwrapper = MagicMock(spec=[])  # Empty spec to avoid _fig_mpl
        mock_figwrapper.figure = None

        # Should not call plt.close when figure is None
        close(mock_figwrapper)
        mock_plt_close.assert_not_called()
            
    def test_close_docstring_content(self):
        """Test that close function has comprehensive docstring."""
        assert close.__doc__ is not None
        doc = close.__doc__
        
        # Check for key documentation elements
        assert "Close a matplotlib figure" in doc
        assert "Parameters" in doc
        assert "Raises" in doc
        assert "Examples" in doc
        assert "TypeError" in doc
        assert "memory leaks" in doc.lower()
        
    @patch('scitex.plt.utils._close.plt.close')
    def test_close_integration_with_mock_figwrapper_class(self, mock_plt_close):
        """Test close with object having _fig_mpl attribute (FigWrapper pattern)."""
        # Create instance with _fig_mpl attribute (new implementation pattern)
        mock_figwrapper_instance = MagicMock(spec=[])
        mock_fig = Mock(spec=matplotlib.figure.Figure)
        mock_figwrapper_instance._fig_mpl = mock_fig

        close(mock_figwrapper_instance)
        mock_plt_close.assert_called_once_with(mock_fig)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/_close.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-29 20:41:30 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/plt/_close.py
# # ----------------------------------------
# import os
# 
# __FILE__ = "./src/scitex/plt/_close.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import matplotlib
# import matplotlib.pyplot as plt
# 
# 
# def close(obj):
#     """Close a matplotlib figure or SciTeX FigWrapper object.
# 
#     Properly closes matplotlib figures to free memory, handling both
#     standard matplotlib Figure objects and SciTeX FigWrapper objects.
#     This is important for preventing memory leaks when creating many plots.
# 
#     Parameters
#     ----------
#     obj : matplotlib.figure.Figure or scitex.plt.FigWrapper
#         The figure object to close. Can be either a matplotlib Figure
#         or an SciTeX FigWrapper instance.
# 
#     Raises
#     ------
#     TypeError
#         If obj is neither a Figure nor FigWrapper object.
# 
#     Examples
#     --------
#     >>> # Close a matplotlib figure
#     >>> fig, ax = plt.subplots()
#     >>> ax.plot([1, 2, 3], [1, 4, 9])
#     >>> close(fig)
# 
#     >>> # Close an SciTeX FigWrapper
#     >>> fig, axes = scitex.plt.subplots(2, 2)
#     >>> close(fig)
# 
#     >>> # Prevents memory leaks in loops
#     >>> for i in range(100):
#     ...     fig, ax = plt.subplots()
#     ...     ax.plot(data[i])
#     ...     plt.savefig(f'plot_{i}.png')
#     ...     close(fig)  # Important!
# 
#     See Also
#     --------
#     matplotlib.pyplot.close : Standard matplotlib close function
#     scitex.plt.subplots : Creates FigWrapper objects
#     """
#     if isinstance(obj, matplotlib.figure.Figure):
#         plt.close(obj)
#     elif hasattr(obj, "_fig_mpl"):
#         # SciTeX FigWrapper object
#         plt.close(obj._fig_mpl)
#     elif hasattr(obj, "figure"):
#         # Alternative attribute name (backward compatibility)
#         fig = obj.figure
#         if fig is not None:
#             plt.close(fig)
#     else:
#         raise TypeError(
#             f"Cannot close object of type {type(obj).__name__}. Expected FigWrapper or Figure object."
#         )
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/_close.py
# --------------------------------------------------------------------------------
