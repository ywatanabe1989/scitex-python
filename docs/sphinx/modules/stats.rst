Stats Module (``stx.stats``)
============================

23 statistical tests with effect sizes, confidence intervals, and
publication-ready formatting.

Quick Reference
---------------

.. code-block:: python

    import scitex as stx
    import numpy as np

    g1 = np.random.normal(0, 1, 50)
    g2 = np.random.normal(0.5, 1, 50)

    # Run a test
    result = stx.stats.test_ttest_ind(g1, g2)

    # StatResult with all details
    print(result.statistic)     # t-statistic
    print(result.p_value)       # p-value
    print(result.effect_size)   # Cohen's d
    print(result.ci_lower)      # 95% CI lower
    print(result.ci_upper)      # 95% CI upper

    # Get as DataFrame or LaTeX
    df = stx.stats.test_ttest_ind(g1, g2, return_as="dataframe")
    tex = stx.stats.test_ttest_ind(g1, g2, return_as="latex")

Available Tests
---------------

**Parametric**

- ``test_ttest_1samp(data, popmean)`` -- One-sample t-test
- ``test_ttest_ind(g1, g2)`` -- Independent two-sample t-test
- ``test_ttest_rel(g1, g2)`` -- Paired t-test
- ``test_anova(*groups)`` -- One-way ANOVA
- ``test_anova_2way(data, factor1, factor2)`` -- Two-way ANOVA
- ``test_anova_rm(data, groups)`` -- Repeated measures ANOVA

**Non-parametric**

- ``test_mannwhitneyu(g1, g2)`` -- Mann-Whitney U test
- ``test_wilcoxon(g1, g2)`` -- Wilcoxon signed-rank test
- ``test_kruskal(*groups)`` -- Kruskal-Wallis H test
- ``test_friedman(*groups)`` -- Friedman test
- ``test_brunner_munzel(g1, g2)`` -- Brunner-Munzel test

**Correlation**

- ``test_pearson(x, y)`` -- Pearson correlation
- ``test_spearman(x, y)`` -- Spearman rank correlation
- ``test_kendall(x, y)`` -- Kendall tau correlation

**Categorical**

- ``test_chi2(observed)`` -- Chi-squared test
- ``test_fisher(table)`` -- Fisher's exact test
- ``test_mcnemar(table)`` -- McNemar test
- ``test_cochran_q(*groups)`` -- Cochran's Q test

**Normality**

- ``test_shapiro(data)`` -- Shapiro-Wilk test
- ``test_ks(data)`` -- Kolmogorov-Smirnov test

Test Recommendation
-------------------

Not sure which test to use? Let SciTeX recommend:

.. code-block:: python

    recommendations = stx.stats.recommend_tests(
        n_groups=2,
        sample_sizes=[30, 35],
        outcome_type="continuous",
        paired=False,
    )

Output Formats
--------------

Every test supports ``return_as`` parameter:

- ``"auto"`` (default) -- Returns ``StatResult`` namedtuple
- ``"dataframe"`` -- Returns pandas DataFrame
- ``"latex"`` -- Returns LaTeX-formatted string
- ``"dict"`` -- Returns plain dict

Descriptive Statistics
----------------------

.. code-block:: python

    stx.stats.describe(data)            # mean, std, median, IQR, etc.
    stx.stats.effect_size(g1, g2)       # Cohen's d
    stx.stats.normality_test(data)      # Shapiro-Wilk
    stx.stats.p_to_stars(0.003)         # "**"

Multiple Comparison Correction
------------------------------

.. code-block:: python

    corrected = stx.stats.correct_pvalues(
        [0.01, 0.03, 0.05, 0.001],
        method="fdr_bh",
    )

Post-hoc Tests
--------------

.. code-block:: python

    stx.stats.posthoc_test(
        [g1, g2, g3],
        group_names=["Control", "Treatment A", "Treatment B"],
        method="tukey",
    )

API Reference
-------------

.. automodule:: scitex.stats
   :members:
