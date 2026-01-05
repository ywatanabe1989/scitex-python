#!/usr/bin/env python3
# Time-stamp: "2025-06-02 15:15:00 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/dict/test__DotDict.py

"""Tests for DotDict class."""

import json

import pytest

from scitex.dict import DotDict


def test_dotdict_basic_initialization():
    """Test basic DotDict initialization."""
    # Empty initialization
    dd = DotDict()
    assert len(dd) == 0
    assert list(dd.keys()) == []

    # Initialize with dict
    data = {"key1": "value1", "key2": 2}
    dd = DotDict(data)
    assert dd.key1 == "value1"
    assert dd["key2"] == 2
    assert len(dd) == 2


def test_dotdict_invalid_initialization():
    """Test DotDict with invalid initialization."""
    with pytest.raises(TypeError, match="Input must be a dictionary"):
        DotDict("not a dict")

    with pytest.raises(TypeError, match="Input must be a dictionary"):
        DotDict([1, 2, 3])


def test_dotdict_attribute_access():
    """Test attribute-style access."""
    dd = DotDict({"name": "test", "value": 42})

    # Get
    assert dd.name == "test"
    assert dd.value == 42

    # Set
    dd.name = "updated"
    dd.new_attr = "new_value"
    assert dd.name == "updated"
    assert dd.new_attr == "new_value"

    # Delete
    del dd.value
    assert "value" not in dd
    with pytest.raises(AttributeError):
        _ = dd.value


def test_dotdict_item_access():
    """Test dictionary-style item access."""
    dd = DotDict()

    # Set
    dd["key1"] = "value1"
    dd[100] = "integer key"
    dd["invalid-key"] = "hyphenated"

    # Get
    assert dd["key1"] == "value1"
    assert dd[100] == "integer key"
    assert dd["invalid-key"] == "hyphenated"

    # Delete
    del dd[100]
    assert 100 not in dd
    with pytest.raises(KeyError):
        _ = dd[100]


def test_dotdict_nested_conversion():
    """Test nested dictionary conversion."""
    data = {"level1": {"level2": {"level3": "deep_value"}}}
    dd = DotDict(data)

    # Check nested access
    assert dd.level1.level2.level3 == "deep_value"
    assert isinstance(dd.level1, DotDict)
    assert isinstance(dd.level1.level2, DotDict)

    # Update nested
    dd.level1.level2.new_key = "new_value"
    assert dd.level1.level2.new_key == "new_value"


def test_dotdict_integer_keys():
    """Test handling of integer and non-identifier keys."""
    dd = DotDict(
        {
            100: "int_key",
            "valid_key": "string_key",
            "invalid-key": "hyphen",
            "123start": "digit_start",
        }
    )

    # Integer keys can only be accessed via item syntax
    assert dd[100] == "int_key"
    assert dd["invalid-key"] == "hyphen"
    assert dd["123start"] == "digit_start"

    # Valid identifiers can use both
    assert dd.valid_key == "string_key"
    assert dd["valid_key"] == "string_key"


def test_dotdict_standard_methods():
    """Test standard dictionary methods."""
    dd = DotDict({"a": 1, "b": 2, "c": 3})

    # keys, values, items
    assert set(dd.keys()) == {"a", "b", "c"}
    assert set(dd.values()) == {1, 2, 3}
    assert set(dd.items()) == {("a", 1), ("b", 2), ("c", 3)}

    # get
    assert dd.get("a") == 1
    assert dd.get("z", "default") == "default"

    # pop
    assert dd.pop("b") == 2
    assert "b" not in dd
    assert dd.pop("z", "default") == "default"
    with pytest.raises(KeyError):
        dd.pop("nonexistent")


def test_dotdict_update():
    """Test update method."""
    dd = DotDict({"a": 1})

    # Update with dict
    dd.update({"b": 2, "c": 3})
    assert dd.b == 2
    assert dd.c == 3

    # Update with iterable
    dd.update([("d", 4), ("e", 5)])
    assert dd.d == 4
    assert dd.e == 5

    # Update with nested dict
    dd.update({"nested": {"key": "value"}})
    assert isinstance(dd.nested, DotDict)
    assert dd.nested.key == "value"


def test_dotdict_setdefault():
    """Test setdefault method."""
    dd = DotDict({"a": 1})

    # Existing key
    assert dd.setdefault("a", 10) == 1
    assert dd.a == 1

    # New key
    assert dd.setdefault("b", 2) == 2
    assert dd.b == 2

    # New key with dict value
    dd.setdefault("nested", {"key": "value"})
    assert isinstance(dd.nested, DotDict)


def test_dotdict_contains():
    """Test contains operator."""
    dd = DotDict({"a": 1, 100: "int_key"})

    assert "a" in dd
    assert 100 in dd
    assert "nonexistent" not in dd


def test_dotdict_iteration():
    """Test iteration over keys."""
    data = {"a": 1, "b": 2, "c": 3}
    dd = DotDict(data)

    keys = list(dd)
    assert set(keys) == set(data.keys())

    # Iteration in for loop
    collected = []
    for key in dd:
        collected.append((key, dd[key]))
    assert set(collected) == set(data.items())


def test_dotdict_copy():
    """Test shallow copy."""
    dd = DotDict({"a": 1, "nested": {"b": 2}})
    dd_copy = dd.copy()

    # Modify copy
    dd_copy.a = 10
    dd_copy.c = 3

    # Original unchanged
    assert dd.a == 1
    assert "c" not in dd

    # Shallow copy - nested objects are shared
    dd_copy.nested.b = 20
    assert dd.nested.b == 20  # Changed in original too


def test_dotdict_to_dict():
    """Test conversion back to regular dict."""
    dd = DotDict({"a": 1, "nested": {"b": 2, "deep": {"c": 3}}, 100: "int_key"})

    result = dd.to_dict()
    assert isinstance(result, dict)
    assert not isinstance(result, DotDict)
    assert result["a"] == 1
    assert result[100] == "int_key"

    # Check nested conversion
    assert isinstance(result["nested"], dict)
    assert not isinstance(result["nested"], DotDict)
    assert result["nested"]["deep"]["c"] == 3


def test_dotdict_string_representation():
    """Test string and repr methods."""
    dd = DotDict({"name": "test", "value": 42})

    # repr uses pprint.pformat which shows dict-like representation
    repr_str = repr(dd)
    assert "'name'" in repr_str or '"name"' in repr_str
    assert "test" in repr_str

    # str should be JSON format
    str_str = str(dd)
    parsed = json.loads(str_str)
    assert parsed["name"] == "test"
    assert parsed["value"] == 42


def test_dotdict_non_json_serializable():
    """Test handling of non-JSON-serializable values."""

    class CustomObj:
        def __str__(self):
            return "CustomObject"

    dd = DotDict({"obj": CustomObj()})
    str_repr = str(dd)
    assert "CustomObject" in str_repr


def test_dotdict_dir():
    """Test dir() for tab completion."""
    dd = DotDict({"valid_key": 1, "another_key": 2, 100: "int_key", "invalid-key": 3})

    dir_result = dir(dd)

    # Should include valid identifiers
    assert "valid_key" in dir_result
    assert "another_key" in dir_result

    # Should include standard methods
    assert "keys" in dir_result
    assert "items" in dir_result

    # Should not include non-identifier keys
    assert 100 not in dir_result
    assert "invalid-key" not in dir_result


def test_dotdict_protected_attributes():
    """Test handling of protected attributes."""
    dd = DotDict()

    # Should not be able to access _data directly via dot notation
    with pytest.raises(AttributeError):
        _ = dd._nonexistent

    # But can set internal attributes
    dd._custom = "value"
    assert dd._custom == "value"


def test_dotdict_edge_cases():
    """Test edge cases and special scenarios."""
    dd = DotDict()

    # Empty string key
    dd[""] = "empty"
    assert dd[""] == "empty"

    # Keys that conflict with methods
    dd["keys"] = "not_a_method"
    assert dd["keys"] == "not_a_method"
    assert callable(dd.keys)  # Method still accessible

    # None as key
    dd[None] = "none_value"
    assert dd[None] == "none_value"

    # Complex nested update
    dd.update({"level1": {"level2": {"data": [1, 2, {"nested": "value"}]}}})
    assert dd.level1.level2.data[2]["nested"] == "value"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dict/_DotDict.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-11-10 22:40:05 (ywatanabe)"
#
#
# import json
# import pprint as _pprint
#
#
# class DotDict:
#     """
#     A dictionary-like object that allows attribute-like access (for valid identifier keys)
#     and standard item access for all keys (including integers, etc.).
#     """
#
#     def __init__(self, dictionary=None):
#         # Use a private attribute to store the actual data
#         # Avoids conflicts with keys named like standard methods ('keys', 'items', etc.)
#         super().__setattr__("_data", {})
#         if dictionary is not None:
#             if isinstance(dictionary, DotDict):
#                 dictionary = dictionary._data
#             elif not isinstance(dictionary, dict):
#                 raise TypeError("Input must be a dictionary.")
#
#             for key, value in dictionary.items():
#                 if isinstance(value, dict) and not isinstance(value, DotDict):
#                     value = DotDict(value)
#                 self[key] = value
#
#     # --- Attribute Access (for valid identifiers) ---
#
#     def __getattr__(self, key):
#         # Called for obj.key when key is not found by normal attribute lookup
#         # Allow access to internal methods/attributes starting with '_'
#         if key.startswith("_"):
#             return super().__getattribute__(
#                 key
#             )  # Use superclass method for internal attrs
#         try:
#             return self._data[key]
#         except KeyError:
#             # Mimic standard attribute access behavior
#             raise AttributeError(
#                 f"'{type(self).__name__}' object has no attribute '{key}'"
#             )
#
#     def __setattr__(self, key, value):
#         # Called for obj.key = value
#         # Protect internal attributes
#         if key == "_data" or key.startswith("_"):
#             super().__setattr__(key, value)
#         else:
#             if isinstance(value, dict) and not isinstance(value, DotDict):
#                 value = DotDict(value)
#             self._data[key] = value
#
#     def __delattr__(self, key):
#         # Called for del obj.key
#         if key.startswith("_"):
#             super().__delattr__(key)
#         else:
#             try:
#                 del self._data[key]
#             except KeyError:
#                 raise AttributeError(
#                     f"'{type(self).__name__}' object has no attribute '{key}'"
#                 )
#
#     # --- Item Access (for any key type) ---
#
#     def __getitem__(self, key):
#         # Called for obj[key]
#         return self._data[key]  # Raises KeyError if not found
#
#     def __setitem__(self, key, value):
#         # Called for obj[key] = value
#         if isinstance(value, dict) and not isinstance(value, DotDict):
#             value = DotDict(value)
#         self._data[key] = value
#
#     def __delitem__(self, key):
#         # Called for del obj[key]
#         del self._data[key]  # Raises KeyError if not found
#
#     # --- Standard Dictionary Methods (operating on _data) ---
#
#     def get(self, key, default=None):
#         # Use _data's get method
#         return self._data.get(key, default)
#
#     def to_dict(self, include_private=False):
#         """
#         Recursively converts DotDict and nested DotDict objects back to ordinary dictionaries.
#
#         Args:
#             include_private: If False, exclude keys starting with '_' (default: False)
#         """
#         result = {}
#         for key, value in self._data.items():
#             # Skip private keys (starting with _) unless explicitly requested
#             if not include_private and isinstance(key, str) and key.startswith("_"):
#                 continue
#             if isinstance(value, DotDict):
#                 value = value.to_dict(include_private=include_private)
#             result[key] = value
#         return result
#
#     def __str__(self):
#         """
#         Returns a string representation, handling non-JSON-serializable objects.
#         """
#
#         def default_handler(obj):
#             # Handle DotDict specifically during serialization if needed,
#             # otherwise represent non-serializable items as strings.
#             if isinstance(obj, DotDict):
#                 return obj.to_dict()
#             try:
#                 # Attempt standard JSON serialization first
#                 json.dumps(obj)
#                 return obj  # Let json.dumps handle it if possible
#             except (TypeError, OverflowError):
#                 return str(obj)  # Fallback for non-serializable types
#
#         try:
#             # Use the internal _data for representation
#             return json.dumps(self.to_dict(), indent=4, default=default_handler)
#         except TypeError as e:
#             # Fallback if default_handler still fails (e.g., complex recursion)
#             return f"<DotDict object at {hex(id(self))}, contains: {list(self._data.keys())}> Error: {e}"
#
#     def __repr__(self):
#         """
#         Returns a string representation suitable for debugging.
#         Returns a nicely formatted representation that pprint can display properly.
#         Private keys (starting with '_') are hidden by default.
#         """
#         # Use pprint.pformat for nice formatting
#         # Convert to regular dict recursively, hiding private keys
#         return _pprint.pformat(
#             self.to_dict(include_private=False), indent=2, width=80, compact=False
#         )
#
#     def _repr_pretty_(self, p, cycle):
#         """
#         IPython/Jupyter pretty printing support.
#         This method is called by IPython's pprint when displaying DotDict objects.
#         Private keys (starting with '_') are hidden by default.
#
#         Args:
#             p: The pretty printer object
#             cycle: Boolean indicating if we're in a reference cycle
#         """
#         if cycle:
#             p.text("DotDict(...)")
#         else:
#             with p.group(8, "DotDict(", ")"):
#                 if self._data:
#                     # Filter out private keys for display
#                     public_data = self.to_dict(include_private=False)
#                     p.pretty(public_data)
#                 else:
#                     p.text("{}")
#
#     def pformat(
#         self, indent=2, width=80, depth=None, compact=False, include_private=False
#     ):
#         """
#         Return a pretty-formatted string representation of the DotDict.
#
#         Args:
#             indent: Number of spaces per indentation level (default: 2)
#             width: Maximum line width (default: 80)
#             depth: Maximum depth to print (default: None for unlimited)
#             compact: If True, use more compact representation (default: False)
#             include_private: If True, include keys starting with '_' (default: False)
#
#         Returns:
#             Pretty-formatted string
#         """
#         return _pprint.pformat(
#             self.to_dict(include_private=include_private),
#             indent=indent,
#             width=width,
#             depth=depth,
#             compact=compact,
#         )
#
#     def __len__(self):
#         """
#         Returns the number of key-value pairs in the dictionary.
#         """
#         return len(self._data)
#
#     def keys(self):
#         """
#         Returns a view object displaying a list of all the keys.
#         """
#         return self._data.keys()
#
#     def values(self):
#         """
#         Returns a view object displaying a list of all the values.
#         """
#         return self._data.values()
#
#     def items(self):
#         """
#         Returns a view object displaying a list of all the items (key, value pairs).
#         """
#         return self._data.items()
#
#     def update(self, dictionary):
#         """
#         Updates the dictionary with the key-value pairs from another dictionary or iterable.
#         """
#         if isinstance(dictionary, dict):
#             iterator = dictionary.items()
#         # Allow updating from iterables of key-value pairs
#         elif hasattr(dictionary, "__iter__"):
#             iterator = dictionary
#         else:
#             raise TypeError(
#                 "Input must be a dictionary or an iterable of key-value pairs."
#             )
#
#         for key, value in iterator:
#             # Use __setitem__ to handle potential DotDict conversion
#             self[key] = value
#
#     def setdefault(self, key, default=None):
#         """
#         Returns the value of the given key. If the key does not exist, insert the key
#         with the specified default value and return the default value.
#         """
#         if key not in self._data:
#             # Use __setitem__ for potential DotDict conversion of default
#             self[key] = default
#             return default
#         else:
#             return self._data[key]
#
#     def pop(self, key, *args):
#         """
#         Removes the specified key and returns the corresponding value.
#         If key is not found, default is returned if given, otherwise KeyError is raised.
#         Accepts optional default value like dict.pop.
#         """
#         # Mimic dict.pop behavior with optional default
#         if len(args) > 1:
#             raise TypeError(f"pop expected at most 2 arguments, got {1 + len(args)}")
#         if key not in self._data:
#             if args:
#                 return args[0]  # Return default if provided
#             else:
#                 raise KeyError(key)
#         return self._data.pop(key)  # Use internal dict's pop
#
#     def __contains__(self, key):
#         """
#         Checks if the dotdict contains the specified key.
#         """
#         return key in self._data
#
#     def __iter__(self):
#         """
#         Returns an iterator over the keys of the dictionary.
#         """
#         return iter(self._data)
#
#     def copy(self):
#         """
#         Creates a shallow copy of the DotDict object.
#         Nested DotDicts/dicts/lists will be references, not copies.
#         Use deepcopy for a fully independent copy.
#         """
#         # Create a new instance using a copy of the internal data
#         return DotDict(self._data.copy())  # Shallow copy of internal dict
#
#     def __dir__(self):
#         """
#         Provides attribute suggestions for dir() and tab completion.
#         Includes both standard methods/attributes and the keys stored in _data.
#         """
#         # Get standard attributes/methods from the object's structure
#         standard_attrs = set(super().__dir__())
#
#         # Get the keys from the internal data dictionary
#         # Filter for keys that are valid identifiers for attribute access
#         data_keys = set(
#             key
#             for key in self._data.keys()
#             if isinstance(key, str) and key.isidentifier()
#         )
#
#         # Return a sorted list of the combined unique names
#         return sorted(list(standard_attrs.union(data_keys)))
#
#     # --- Comparison Methods ---
#
#     def __eq__(self, other):
#         """Check equality. Supports comparison with dict, DotDict, and scalar values."""
#         if isinstance(other, DotDict):
#             return self._data == other._data
#         elif isinstance(other, dict):
#             return self._data == other
#         else:
#             # For scalar comparison, compare against empty/falsy
#             return False
#
#     def __ne__(self, other):
#         """Check inequality."""
#         return not self.__eq__(other)
#
#     def __lt__(self, other):
#         """Less than comparison - delegate to _data or handle scalars."""
#         if isinstance(other, (int, float)):
#             # If comparing to scalar, treat empty as 0
#             if len(self._data) == 0:
#                 return 0 < other
#             # If single numeric value in total field, use it
#             if "total" in self._data and isinstance(self._data["total"], (int, float)):
#                 return self._data["total"] < other
#             # Otherwise not comparable
#             return NotImplemented
#         return NotImplemented
#
#     def __le__(self, other):
#         """Less than or equal."""
#         return self.__lt__(other) or self.__eq__(other)
#
#     def __gt__(self, other):
#         """Greater than comparison - delegate to _data or handle scalars."""
#         if isinstance(other, (int, float)):
#             # If comparing to scalar, treat empty as 0
#             if len(self._data) == 0:
#                 return 0 > other
#             # If single numeric value in total field, use it
#             if "total" in self._data and isinstance(self._data["total"], (int, float)):
#                 return self._data["total"] > other
#             # Otherwise not comparable
#             return NotImplemented
#         return NotImplemented
#
#     def __ge__(self, other):
#         """Greater than or equal."""
#         return self.__gt__(other) or self.__eq__(other)
#
#     def __bool__(self):
#         """Truth value testing. Empty DotDict is False, non-empty is True."""
#         return len(self._data) > 0
#
#
# # Example Usage:
# if __name__ == "__main__":
#     data = {
#         "name": "example",
#         "version": 1,
#         100: "integer key",
#         "nested": {"value1": True, 200: False},
#         "list_val": [1, {"a": 2}],
#         "invalid-key": "hyphenated",
#     }
#
#     dd = DotDict(data)
#
#     # Access via attribute (for valid identifiers)
#     print(f"dd.name: {dd.name}")
#     print(f"dd.version: {dd.version}")
#     print(f"dd.nested.value1: {dd.nested.value1}")
#     # print(dd.100)  # This would be a SyntaxError, as expected
#
#     # Access via item (for any key)
#     print(f"dd[100]: {dd[100]}")
#     print(f"dd['nested'][200]: {dd['nested'][200]}")
#     print(f"dd['invalid-key']: {dd['invalid-key']}")
#
#     # Modify values
#     dd.name = "updated example"
#     dd[100] = "new integer value"
#     dd.nested[200] = "updated nested int key"
#     dd[300] = "new top-level int key"  # Add new int key
#     dd["new-key"] = "another invalid id key"
#
#     print("\n--- After Modifications ---")
#     print(f"dd.name: {dd.name}")
#     print(f"dd[100]: {dd[100]}")
#     print(f"dd.nested[200]: {dd.nested[200]}")
#     print(f"dd[300]: {dd[300]}")
#     print(f"dd['new-key']: {dd['new-key']}")
#
#     print("\n--- Representation ---")
#     print(f"repr(dd): {repr(dd)}")
#
#     print("\n--- String (JSON) Representation ---")
#     print(f"str(dd):\n{str(dd)}")
#
#     print("\n--- Convert back to dict ---")
#     plain_dict = dd.to_dict()
#     print(f"plain_dict: {plain_dict}")
#     print(f"plain_dict[100]: {plain_dict[100]}")
#     print(f"plain_dict['nested'][200]: {plain_dict['nested'][200]}")
#
#     print("\n--- Iteration ---")
#     for k in dd:
#         print(f"Key: {k}, Value: {dd[k]}")
#
#     print("\n--- Contains ---")
#     print(f"100 in dd: {100 in dd}")
#     print(f"'name' in dd: {'name' in dd}")
#     print(f"999 in dd: {999 in dd}")
#
#     print("\n--- Copy ---")
#     dd_copy = dd.copy()
#     dd_copy[100] = "value in copy"
#     dd_copy.name = "name in copy"
#     print(f"Original dd[100]: {dd[100]}")
#     print(f"Copy dd_copy[100]: {dd_copy[100]}")
#     print(f"Original dd.name: {dd.name}")
#     print(f"Copy dd_copy.name: {dd_copy.name}")
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dict/_DotDict.py
# --------------------------------------------------------------------------------
