#!/usr/bin/env python3
# Timestamp: 2026-01-26
# File: /home/ywatanabe/proj/scitex-python/src/scitex/str/_title_case.py

"""Title case conversion for strings."""


def title_case(text):
    """
    Convert a string to title case.

    Keeps certain prepositions, conjunctions, and articles in lowercase,
    and ensures words detected as potential acronyms (all uppercase) are
    fully capitalized.

    Parameters
    ----------
    text : str
        The text to convert to title case.

    Returns
    -------
    str
        The converted text in title case with certain words in lowercase
        and potential acronyms fully capitalized.

    Examples
    --------
    >>> title_case("welcome to the world of ai and using CPUs for gaming")
    'Welcome to the World of AI and Using CPUs for Gaming'
    """
    # List of words to keep in lowercase
    lowercase_words = [
        "a",
        "an",
        "the",
        "and",
        "but",
        "or",
        "nor",
        "at",
        "by",
        "to",
        "in",
        "with",
        "of",
        "on",
    ]

    words = text.split()
    final_words = []
    for word in words:
        # Check if the word is fully in uppercase and more than one character
        if word.isupper() and len(word) > 1:
            final_words.append(word)
        elif word.lower() in lowercase_words:
            final_words.append(word.lower())
        else:
            final_words.append(word.capitalize())
    return " ".join(final_words)


# EOF
