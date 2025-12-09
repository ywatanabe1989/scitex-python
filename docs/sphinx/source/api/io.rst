scitex.io
=========

File I/O
--------

.. py:function:: scitex.io.save(fig, path, **kwargs)

   Save figure to file. Automatically generates CSV with plot data.

   :param fig: Figure object from ``stx.plt.subplots()``
   :param path: Output file path (.png, .pdf, .svg, etc.)
   :param kwargs: Additional arguments passed to matplotlib savefig

   **Example:**

   .. code-block:: python

      import scitex as stx

      fig, ax = stx.plt.subplots()
      ax.stx_line([1, 2, 3, 4, 5], id="my_data")

      # Saves both PNG and CSV
      stx.io.save(fig, "output.png")

.. py:function:: scitex.io.load(path)

   Load data from file.

   :param path: Input file path
   :returns: Loaded data (DataFrame, array, etc.)
