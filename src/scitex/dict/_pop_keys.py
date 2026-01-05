#!/usr/bin/env python3
# Timestamp: "2025-11-10 22:40:16 (ywatanabe)"


def pop_keys(keys_list, keys_to_pop):
    """Remove specified keys from a list of keys.

    Parameters
    ----------
    keys_list : list
        The original list of keys.
    keys_to_pop : list
        The list of keys to remove from keys_list.

    Returns
    -------
    list
        A new list with the specified keys removed.

    Example
    -------
    >>> keys_list = ['a', 'b', 'c', 'd', 'e', 'bde']
    >>> keys_to_pop = ['b', 'd']
    >>> pop_keys(keys_list, keys_to_pop)
    ['a', 'c', 'e', 'bde']
    """
    return [k for k in keys_list if k not in keys_to_pop]


# EOF
