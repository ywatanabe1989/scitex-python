scitex.writer
=============

LaTeX manuscript compilation system. This module is a thin wrapper around
the `scitex-writer <https://github.com/ywatanabe1989/scitex-writer>`_ package.

.. module:: scitex.writer

Installation
------------

The writer module requires the scitex-writer package:

.. code-block:: bash

   pip install scitex-writer

Quick Start
-----------

.. code-block:: python

   from scitex.writer import Writer
   from pathlib import Path

   # Create or attach to a project
   writer = Writer(Path("my_paper"))

   # Compile manuscript
   result = writer.compile_manuscript()
   if result.success:
       print(f"PDF created: {result.output_pdf}")

   # Compile supplementary
   result = writer.compile_supplementary()

   # Compile revision with change tracking
   result = writer.compile_revision(track_changes=True)

API Reference
-------------

Writer Class
~~~~~~~~~~~~

.. autoclass:: scitex.writer.Writer
   :members:
   :undoc-members:
   :show-inheritance:

Dataclasses
~~~~~~~~~~~

.. autoclass:: scitex.writer.CompilationResult
   :members:
   :undoc-members:

.. autoclass:: scitex.writer.ManuscriptTree
   :members:
   :undoc-members:

.. autoclass:: scitex.writer.SupplementaryTree
   :members:
   :undoc-members:

.. autoclass:: scitex.writer.RevisionTree
   :members:
   :undoc-members:

Submodules
~~~~~~~~~~

The following submodules are available:

- :mod:`scitex.writer.bib` - BibTeX management
- :mod:`scitex.writer.compile` - Compilation functions
- :mod:`scitex.writer.figures` - Figure management
- :mod:`scitex.writer.tables` - Table management
- :mod:`scitex.writer.guidelines` - Writing guidelines
- :mod:`scitex.writer.project` - Project management
- :mod:`scitex.writer.prompts` - AI writing prompts

Utility Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: scitex.writer.has_writer

Module Attributes
~~~~~~~~~~~~~~~~~

.. py:data:: WRITER_AVAILABLE
   :type: bool

   True if scitex-writer is installed and available.

.. py:data:: __writer_version__
   :type: str | None

   Version string of the installed scitex-writer package, or None if not installed.
