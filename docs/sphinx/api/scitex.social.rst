scitex.social
=============

Unified social media management. This module is a thin wrapper around
the `socialia <https://github.com/ywatanabe1989/socialia>`_ package.

.. module:: scitex.social

Installation
------------

The social module requires the socialia package:

.. code-block:: bash

   pip install socialia

Quick Start
-----------

.. code-block:: python

   import scitex as stx

   # Twitter/X
   x = stx.social.Twitter()
   x.post("Hello from SciTeX!")

   # LinkedIn
   linkedin = stx.social.LinkedIn()
   linkedin.post("Research update", visibility="public")

   # YouTube analytics
   yt = stx.social.YouTube()
   stats = yt.get_channel_stats()

   # Google Analytics
   ga = stx.social.GoogleAnalytics()
   report = ga.get_report(start_date="7daysAgo")

Environment Variables
---------------------

Credentials use ``SCITEX_SOCIAL_`` prefix (falls back to ``SOCIALIA_``):

- ``SCITEX_SOCIAL_X_CONSUMER_KEY``, ``SCITEX_SOCIAL_X_CONSUMER_KEY_SECRET``
- ``SCITEX_SOCIAL_X_ACCESS_TOKEN``, ``SCITEX_SOCIAL_X_ACCESS_TOKEN_SECRET``
- ``SCITEX_SOCIAL_X_BEARER_TOKEN``
- ``SCITEX_SOCIAL_LINKEDIN_CLIENT_ID``, ``SCITEX_SOCIAL_LINKEDIN_CLIENT_SECRET``
- ``SCITEX_SOCIAL_LINKEDIN_ACCESS_TOKEN``
- ``SCITEX_SOCIAL_REDDIT_CLIENT_ID``, ``SCITEX_SOCIAL_REDDIT_CLIENT_SECRET``
- ``SCITEX_SOCIAL_YOUTUBE_API_KEY``
- ``SCITEX_SOCIAL_GOOGLE_ANALYTICS_PROPERTY_ID``

API Reference
-------------

Platform Clients
~~~~~~~~~~~~~~~~

.. autoclass:: scitex.social.Twitter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scitex.social.LinkedIn
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scitex.social.Reddit
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scitex.social.YouTube
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scitex.social.GoogleAnalytics
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: scitex.social.has_socialia

Module Attributes
~~~~~~~~~~~~~~~~~

.. py:data:: SOCIALIA_AVAILABLE
   :type: bool

   True if socialia is installed and available.

.. py:data:: __socialia_version__
   :type: str | None

   Version string of the installed socialia package, or None if not installed.

.. py:data:: PLATFORM_STRATEGIES
   :type: str

   Content strategies for MCP tools.
