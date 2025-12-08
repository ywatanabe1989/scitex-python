Quickstart
==========

Basic Plotting
--------------

.. code-block:: python

   import scitex as stx

   # Create figure and axis
   fig, ax = stx.plt.subplots()

   # Plot data
   ax.stx_line([1, 2, 3, 4, 5])

   # Save figure (auto-generates CSV)
   stx.io.save(fig, "output.png")

Statistical Plots
-----------------

.. code-block:: python

   import numpy as np
   import scitex as stx

   # Generate sample data
   data = np.random.randn(100, 50)

   fig, ax = stx.plt.subplots()
   ax.stx_mean_std(data)
   ax.set_xyt(x="Time", y="Value", t="Mean +/- SD")

   stx.io.save(fig, "stats.png")

Categorical Plots
-----------------

.. code-block:: python

   import scitex as stx

   fig, ax = stx.plt.subplots()
   ax.stx_violin([group1, group2, group3])

   stx.io.save(fig, "violin.png")
