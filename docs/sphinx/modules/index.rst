Module Overview
===============

SciTeX is organized into focused modules across five categories,
ordered from core infrastructure to higher-level tools.

.. image:: ../_static/architecture.png
   :width: 100%
   :alt: Module architecture: Experiment (session, config, io, logging, repro), Analysis & Visualization (stats, dsp, plt, diagram), Publication (scholar, writer, clew)

Core
----

The essential workflow: session management, file I/O, configuration,
logging, reproducibility, and provenance tracking.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - :doc:`@stx.session <session>`
     - **Entry point** -- decorator providing CLI generation, output directories, config injection, and provenance tracking
   * - :doc:`stx.io <io>`
     - Universal file I/O -- save/load for 30+ formats (CSV, NumPy, pickle, HDF5, images, YAML, JSON, ...)
   * - :doc:`stx.config <config>`
     - YAML configuration with priority resolution (CLI > config files > defaults)
   * - :doc:`stx.logging <logging>`
     - Logging, warnings, exception hierarchy, and stream redirection
   * - :doc:`stx.repro <repro>`
     - Reproducibility: ``fix_seeds()``, ``gen_id()``, ``gen_timestamp()``, ``RandomStateManager``
   * - :doc:`stx.clew <clew>`
     - Hash-based provenance tracking -- verify runs, trace dependency chains

Science & Analysis
------------------

Statistical tests, publication-ready figures, signal processing,
and diagrams.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - :doc:`stx.stats <stats>`
     - 23 statistical tests with effect sizes, confidence intervals, and APA formatting
   * - :doc:`stx.plt <plt>`
     - Publication figures via figrecipe: 47 plot types, mm-based layout, auto CSV export
   * - :doc:`stx.dsp <dsp>`
     - Signal processing: bandpass/lowpass/highpass filters, PSD, Hilbert, wavelet, PAC
   * - :doc:`stx.diagram <diagram>`
     - Paper-optimized diagrams with Mermaid and Graphviz backends

Literature & Writing
--------------------

Research, write, and validate scientific manuscripts.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - :doc:`stx.scholar <scholar>`
     - Literature management: search, download PDFs, enrich BibTeX, organize library
   * - :doc:`stx.writer <writer>`
     - LaTeX manuscript compilation, table/figure management, IMRAD guidelines
   * - :doc:`stx.linter <linter>`
     - AST-based Python linter enforcing SciTeX conventions (45 rules)

Machine Learning
----------------

Training utilities, classification pipelines, and PyTorch layers.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - :doc:`stx.ai <ai>`
     - ML utilities: ``LearningCurveLogger``, ``EarlyStopping``, ``ClassificationReporter``, metrics
   * - :doc:`stx.nn <nn>`
     - PyTorch layers: signal processing, attention, 1D ResNet, PAC

Utilities
---------

Templates, decorators, introspection, and data helpers.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - :doc:`stx.template <template>`
     - Project scaffolding and code snippet templates
   * - :doc:`stx.decorators <decorators>`
     - Type conversion (``@torch_fn``), batching (``@batch_fn``), caching, timeout
   * - :doc:`stx.introspect <introspect>`
     - IPython-like code inspection: ``q()``, ``qq()``, ``dir()``, call graphs
   * - :doc:`stx.pd <pd>`
     - Pandas helpers: ``find_indi``, ``melt_cols``, ``merge_cols``, ``to_xyz``
   * - ``stx.gen``
     - General utilities: ``mk_spath()``, ``run()``, ``copy()/paste()``
   * - ``stx.str``
     - LaTeX formatting, axis labels, ``color_text()``, ``to_latex_style()``
   * - ``stx.dict``
     - ``DotDict``, ``flatten()``, ``safe_merge()``, ``listed_dict()``
   * - ``stx.db``
     - SQLite3 and PostgreSQL wrappers with batch operations
   * - ``stx.social``
     - Social media posting and Google Analytics

.. toctree::
   :maxdepth: 2
   :caption: Core
   :hidden:

   session
   io
   config
   logging
   repro
   clew

.. toctree::
   :maxdepth: 2
   :caption: Science & Analysis
   :hidden:

   stats
   plt
   dsp
   diagram

.. toctree::
   :maxdepth: 2
   :caption: Literature & Writing
   :hidden:

   scholar
   writer
   linter

.. toctree::
   :maxdepth: 2
   :caption: Machine Learning
   :hidden:

   ai
   nn

.. toctree::
   :maxdepth: 2
   :caption: Utilities
   :hidden:

   template
   decorators
   introspect
   pd
   gen
