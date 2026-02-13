Installation
============

Requirements
------------

- Python 3.10 or higher
- pip package manager

Installing SciTeX
-----------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install scitex

With optional extras (plotting, statistics, scholar, etc.):

.. code-block:: bash

    pip install scitex[all]

From Source
~~~~~~~~~~~

Clone the repository and install in development mode:

.. code-block:: bash

    git clone https://github.com/ywatanabe1989/scitex-python.git
    cd scitex-python
    pip install -e ".[all]"

Dependencies
------------

Core Dependencies
~~~~~~~~~~~~~~~~~

These are installed automatically with ``pip install scitex``:

- numpy
- pandas
- PyYAML
- tqdm
- packaging
- natsort

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

SciTeX uses optional extras to keep the base install lightweight.
Install what you need:

.. code-block:: bash

    pip install scitex[plt]       # Matplotlib + figure tools
    pip install scitex[stats]     # Statistical testing
    pip install scitex[scholar]   # Literature management
    pip install scitex[all]       # Everything

All available extras:

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Extra
     - Category
     - Description
   * - ``session``
     - Core
     - Session management dependencies
   * - ``io``
     - Core
     - Extended file I/O (HDF5, MATLAB, Parquet, etc.)
   * - ``config``
     - Core
     - YAML configuration
   * - ``logging``
     - Core
     - Logging infrastructure
   * - ``repro``
     - Core
     - Reproducibility (seed fixing, timestamps)
   * - ``clew``
     - Core
     - Provenance tracking
   * - ``stats``
     - Science
     - Statistical testing (scipy, statsmodels, pingouin)
   * - ``plt``
     - Science
     - Publication figures (matplotlib, figrecipe)
   * - ``dsp``
     - Science
     - Signal processing (scipy, MNE)
   * - ``diagram``
     - Science
     - Mermaid / Graphviz diagram generation
   * - ``scholar``
     - Literature
     - Paper search, PDF download, BibTeX enrichment
   * - ``writer``
     - Literature
     - LaTeX manuscript compilation
   * - ``ai``
     - ML
     - LLM APIs (OpenAI, Anthropic, Google), scikit-learn
   * - ``nn``
     - ML
     - PyTorch neural network layers
   * - ``torch``
     - ML
     - PyTorch core
   * - ``audio``
     - Utilities
     - Text-to-speech (gTTS, pyttsx3, ElevenLabs)
   * - ``browser``
     - Utilities
     - Web automation (Playwright)
   * - ``capture``
     - Utilities
     - Screenshot capture
   * - ``db``
     - Utilities
     - SQLite / PostgreSQL wrappers
   * - ``pd``
     - Utilities
     - Pandas helpers
   * - ``web``
     - Utilities
     - Web crawling and URL extraction
   * - ``cloud``
     - Utilities
     - Cloud service integration
   * - ``cli``
     - Utilities
     - Command-line interface
   * - ``gen``
     - Utilities
     - General utilities (clipboard, shell commands)
   * - ``linalg``
     - Utilities
     - Linear algebra extensions
   * - ``parallel``
     - Utilities
     - Parallel processing
   * - ``resource``
     - Utilities
     - System resource monitoring
   * - ``dev``
     - Meta
     - Development tools (pytest, black, ruff, mypy)
   * - ``heavy``
     - Meta
     - All heavy dependencies (torch, optuna, etc.)
   * - ``all``
     - Meta
     - Everything combined

Multiple extras can be combined:

.. code-block:: bash

    pip install scitex[plt,stats,scholar]

Verifying Installation
----------------------

.. code-block:: python

    import scitex as stx
    print(stx.__version__)

    # Quick smoke test
    fig, ax = stx.plt.subplots()
    ax.plot_line([1, 2, 3], [1, 4, 9])
    stx.io.save(fig, "/tmp/test.png")
    print("SciTeX is working.")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **Import Error**: Ensure core dependencies are installed:

   .. code-block:: bash

       pip install scitex --force-reinstall

2. **Missing optional module**: If a specific module raises ``ImportError``, install its extras:

   .. code-block:: bash

       pip install scitex[plt,stats]

3. **GPU Support**: For PyTorch GPU acceleration:

   .. code-block:: bash

       pip install torch --index-url https://download.pytorch.org/whl/cu121

Getting Help
~~~~~~~~~~~~

- Search existing `GitHub issues <https://github.com/ywatanabe1989/scitex-python/issues>`_
- Create a new issue with a minimal reproducible example
