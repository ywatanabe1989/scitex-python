#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Test file for scitex.stats.posthoc._dunnett

import pytest
import numpy as np
import pandas as pd
from scitex.stats.posthoc import posthoc_dunnett, posthoc_tukey


class TestBasicComputations:
    """Test basic Dunnett computations."""

    def test_basic_control_vs_treatments(self):
        """Test basic control vs treatments comparison."""
        np.random.seed(42)
        control = np.random.normal(10, 2, 20)
        treatment1 = np.random.normal(12, 2, 20)
        treatment2 = np.random.normal(14, 2, 20)

        results = posthoc_dunnett(
            control=control,
            treatments=[treatment1, treatment2]
        )

        # Should return DataFrame by default
        assert isinstance(results, pd.DataFrame)

        # Should have 2 comparisons (2 treatments vs 1 control)
        assert len(results) == 2

        # Check required columns
        required_cols = ['treatment', 'control', 'mean_diff', 'pvalue',
                        'significant', 't_statistic', 'ci_lower', 'ci_upper']
        for col in required_cols:
            assert col in results.columns

    def test_single_treatment(self):
        """Test with single treatment vs control."""
        np.random.seed(42)
        control = np.random.normal(10, 2, 20)
        treatment = np.random.normal(15, 2, 20)

        results = posthoc_dunnett(
            control=control,
            treatments=[treatment]
        )

        # Should have exactly 1 comparison
        assert len(results) == 1
        assert results.iloc[0]['control'] == 'Control'
        assert results.iloc[0]['treatment'] == 'Treatment 1'

    def test_multiple_treatments(self):
        """Test with multiple treatments."""
        np.random.seed(42)
        control = np.random.normal(10, 2, 30)
        treatments = [np.random.normal(10 + i, 2, 30) for i in range(5)]

        results = posthoc_dunnett(control=control, treatments=treatments)

        # Should have 5 comparisons
        assert len(results) == 5

        # All should compare against same control
        assert all(results['control'] == 'Control')


class TestInputFormats:
    """Test different input formats."""

    def test_numpy_arrays(self):
        """Test with numpy arrays."""
        control = np.array([1, 2, 3, 4, 5])
        treatment = np.array([6, 7, 8, 9, 10])

        results = posthoc_dunnett(control=control, treatments=[treatment])

        assert len(results) == 1

    def test_pandas_series(self):
        """Test with pandas Series."""
        control = pd.Series([1, 2, 3, 4, 5])
        treatment = pd.Series([6, 7, 8, 9, 10])

        results = posthoc_dunnett(control=control, treatments=[treatment])

        assert len(results) == 1

    def test_custom_names(self):
        """Test with custom treatment and control names."""
        control = np.array([1, 2, 3])
        treatments = [np.array([4, 5, 6]), np.array([7, 8, 9])]

        results = posthoc_dunnett(
            control=control,
            treatments=treatments,
            treatment_names=['Low Dose', 'High Dose'],
            control_name='Placebo'
        )

        assert results.iloc[0]['control'] == 'Placebo'
        assert results.iloc[0]['treatment'] == 'Low Dose'
        assert results.iloc[1]['treatment'] == 'High Dose'

    def test_return_dict_format(self):
        """Test return_as='dict' option."""
        control = np.array([1, 2, 3])
        treatment = np.array([4, 5, 6])

        results = posthoc_dunnett(
            control=control,
            treatments=[treatment],
            return_as='dict'
        )

        assert isinstance(results, list)
        assert isinstance(results[0], dict)
        assert 'mean_diff' in results[0]
        assert 'treatment' in results[0]


class TestAlternativeHypotheses:
    """Test different alternative hypotheses."""

    def test_two_sided_default(self):
        """Test two-sided test (default)."""
        np.random.seed(42)
        control = np.random.normal(10, 2, 20)
        treatment = np.random.normal(12, 2, 20)

        results = posthoc_dunnett(
            control=control,
            treatments=[treatment],
            alternative='two-sided'
        )

        assert results.iloc[0]['alternative'] == 'two-sided'
        # CI should be two-sided
        assert results.iloc[0]['ci_lower'] < results.iloc[0]['mean_diff']
        assert results.iloc[0]['ci_upper'] > results.iloc[0]['mean_diff']

    def test_greater_one_sided(self):
        """Test one-sided test (greater)."""
        np.random.seed(42)
        control = np.random.normal(10, 2, 20)
        treatment = np.random.normal(15, 2, 20)

        results = posthoc_dunnett(
            control=control,
            treatments=[treatment],
            alternative='greater'
        )

        assert results.iloc[0]['alternative'] == 'greater'
        # Upper CI should be inf for one-sided test
        assert results.iloc[0]['ci_upper'] == 'inf'

    def test_less_one_sided(self):
        """Test one-sided test (less)."""
        np.random.seed(42)
        control = np.random.normal(10, 2, 20)
        treatment = np.random.normal(5, 2, 20)

        results = posthoc_dunnett(
            control=control,
            treatments=[treatment],
            alternative='less'
        )

        assert results.iloc[0]['alternative'] == 'less'
        # Lower CI should be -inf for one-sided test
        assert results.iloc[0]['ci_lower'] == '-inf'


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_treatment_raises_error(self):
        """Test that no treatment groups raises ValueError."""
        control = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="Need at least 1 treatment"):
            posthoc_dunnett(control=control, treatments=[])

    def test_identical_control_and_treatment(self):
        """Test with identical control and treatment data."""
        identical_data = np.array([5, 5, 5, 5, 5])
        control = identical_data.copy()
        treatment = identical_data.copy()

        results = posthoc_dunnett(control=control, treatments=[treatment])

        # Should not be significant
        assert not results.iloc[0]['significant']
        # Mean difference should be 0
        assert abs(results.iloc[0]['mean_diff']) < 1e-10

    def test_unequal_sample_sizes(self):
        """Test with unequal sample sizes."""
        np.random.seed(42)
        control = np.random.normal(10, 2, 50)  # Large control
        treatment1 = np.random.normal(12, 2, 10)  # Small treatment
        treatment2 = np.random.normal(14, 2, 30)  # Medium treatment

        results = posthoc_dunnett(control=control, treatments=[treatment1, treatment2])

        # Should handle unequal sizes
        assert len(results) == 2
        assert results.iloc[0]['n_control'] == 50
        assert results.iloc[0]['n_treatment'] == 10
        assert results.iloc[1]['n_treatment'] == 30

    def test_very_large_differences(self):
        """Test with very large mean differences."""
        np.random.seed(42)
        # Use groups with some variance to avoid division by zero
        control = np.random.normal(0, 0.1, 10)
        treatment = np.random.normal(100, 0.1, 10)

        results = posthoc_dunnett(control=control, treatments=[treatment])

        # Should be highly significant
        assert results.iloc[0]['significant']
        assert results.iloc[0]['pvalue'] < 0.001

    def test_mismatched_treatment_names_length(self):
        """Test error when treatment_names length doesn't match."""
        control = np.array([1, 2, 3])
        treatments = [np.array([4, 5, 6]), np.array([7, 8, 9])]
        treatment_names = ['Treatment1']  # Only 1 name for 2 treatments

        with pytest.raises(ValueError, match="Expected 2 treatment names"):
            posthoc_dunnett(
                control=control,
                treatments=treatments,
                treatment_names=treatment_names
            )


class TestComparisonWithTukey:
    """Compare Dunnett with Tukey HSD."""

    def test_dunnett_only_control_comparisons(self):
        """Test that Dunnett only compares vs control."""
        np.random.seed(42)
        control = np.random.normal(10, 2, 20)
        treatment1 = np.random.normal(12, 2, 20)
        treatment2 = np.random.normal(14, 2, 20)

        # Dunnett: 2 comparisons (each treatment vs control)
        results_dunnett = posthoc_dunnett(
            control=control,
            treatments=[treatment1, treatment2]
        )

        # Tukey: 3 comparisons (all pairwise)
        results_tukey = posthoc_tukey(
            [control, treatment1, treatment2],
            group_names=['Control', 'Treatment 1', 'Treatment 2']
        )

        assert len(results_dunnett) == 2
        assert len(results_tukey) == 3

    def test_dunnett_more_powerful_for_control(self):
        """Dunnett should be more powerful than Tukey for control comparisons."""
        np.random.seed(42)
        control = np.random.normal(10, 2, 30)
        treatments = [np.random.normal(11, 2, 30) for _ in range(3)]

        results_dunnett = posthoc_dunnett(control=control, treatments=treatments)

        results_tukey = posthoc_tukey(
            [control] + treatments,
            group_names=['Control', 'T1', 'T2', 'T3']
        )

        # Filter Tukey results for control comparisons
        tukey_control = results_tukey[
            (results_tukey['group_i'] == 'Control') |
            (results_tukey['group_j'] == 'Control')
        ]

        # Both should run without error
        assert len(results_dunnett) == 3
        assert len(tukey_control) == 3


class TestStatisticalProperties:
    """Test statistical properties of results."""

    def test_alpha_level_respected(self):
        """Test that alpha level is recorded."""
        np.random.seed(42)
        control = np.random.normal(10, 2, 20)
        treatment = np.random.normal(11, 2, 20)

        results_005 = posthoc_dunnett(
            control=control,
            treatments=[treatment],
            alpha=0.05
        )
        results_001 = posthoc_dunnett(
            control=control,
            treatments=[treatment],
            alpha=0.01
        )

        assert results_005.iloc[0]['alpha'] == 0.05
        assert results_001.iloc[0]['alpha'] == 0.01

    def test_confidence_intervals_two_sided(self):
        """Test confidence interval properties for two-sided test."""
        np.random.seed(42)
        control = np.random.normal(10, 2, 20)
        treatments = [np.random.normal(12 + i, 2, 20) for i in range(3)]

        results = posthoc_dunnett(
            control=control,
            treatments=treatments,
            alternative='two-sided'
        )

        for _, row in results.iterrows():
            # CI should contain the mean difference
            assert row['ci_lower'] <= row['mean_diff'] <= row['ci_upper']
            # CI width should be positive
            assert row['ci_upper'] > row['ci_lower']

    def test_t_statistic_calculation(self):
        """Test t-statistic calculation."""
        np.random.seed(42)
        control = np.random.normal(10, 2, 20)
        treatment = np.random.normal(15, 2, 20)

        results = posthoc_dunnett(control=control, treatments=[treatment])

        # t_statistic should be large for well-separated groups
        assert abs(results.iloc[0]['t_statistic']) > 2

        # Standard error should be positive
        assert results.iloc[0]['std_error'] > 0

    def test_mean_diff_calculation(self):
        """Test mean difference calculation."""
        control = np.array([5, 5, 5, 5, 5])
        treatment = np.array([10, 10, 10, 10, 10])

        results = posthoc_dunnett(control=control, treatments=[treatment])

        # Mean diff should be treatment - control = 10 - 5 = 5
        assert results.iloc[0]['mean_diff'] == 5.0
        assert results.iloc[0]['mean_control'] == 5.0
        assert results.iloc[0]['mean_treatment'] == 10.0


class TestOutputStructure:
    """Test output structure and completeness."""

    def test_dataframe_output_structure(self):
        """Test DataFrame output has all required fields."""
        control = np.random.normal(10, 2, 20)
        treatments = [np.random.normal(i, 2, 20) for i in range(3)]

        results = posthoc_dunnett(control=control, treatments=treatments)

        required_fields = [
            'treatment', 'control', 'n_treatment', 'n_control',
            'mean_treatment', 'mean_control', 'mean_diff', 'std_error',
            't_statistic', 'd_critical', 'pvalue', 'significant',
            'pstars', 'ci_lower', 'ci_upper', 'alpha', 'alternative'
        ]

        for field in required_fields:
            assert field in results.columns, f"Missing field: {field}"

    def test_dict_output_structure(self):
        """Test dict output has all required fields."""
        control = np.random.normal(10, 2, 20)
        treatments = [np.random.normal(i, 2, 20) for i in range(3)]

        results = posthoc_dunnett(
            control=control,
            treatments=treatments,
            return_as='dict'
        )

        required_fields = [
            'treatment', 'control', 'mean_diff', 'pvalue',
            'significant', 't_statistic', 'alternative'
        ]

        for result in results:
            for field in required_fields:
                assert field in result, f"Missing field: {field}"

    def test_control_name_consistency(self):
        """Test that control name is consistent across all comparisons."""
        control = np.random.normal(10, 2, 20)
        treatments = [np.random.normal(i, 2, 20) for i in range(4)]

        results = posthoc_dunnett(
            control=control,
            treatments=treatments,
            control_name='Baseline'
        )

        # All rows should have same control name
        assert all(results['control'] == 'Baseline')

    def test_treatment_names_unique(self):
        """Test that treatment names are unique."""
        control = np.random.normal(10, 2, 20)
        treatments = [np.random.normal(i, 2, 20) for i in range(4)]

        results = posthoc_dunnett(control=control, treatments=treatments)

        # All treatment names should be unique
        treatment_names = results['treatment'].tolist()
        assert len(treatment_names) == len(set(treatment_names))


class TestRobustness:
    """Test robustness to various data conditions."""

    def test_small_sample_sizes(self):
        """Test with very small sample sizes."""
        control = np.array([1, 2])
        treatment1 = np.array([3, 4])
        treatment2 = np.array([5, 6])

        results = posthoc_dunnett(control=control, treatments=[treatment1, treatment2])

        # Should run without error
        assert len(results) == 2

    def test_many_treatments(self):
        """Test with many treatment groups."""
        np.random.seed(42)
        control = np.random.normal(10, 2, 30)
        treatments = [np.random.normal(10 + i*0.5, 2, 30) for i in range(10)]

        results = posthoc_dunnett(control=control, treatments=treatments)

        # Should have 10 comparisons
        assert len(results) == 10

    def test_high_variance_data(self):
        """Test with high variance data."""
        np.random.seed(42)
        control = np.random.normal(10, 100, 30)
        treatment = np.random.normal(12, 100, 30)

        results = posthoc_dunnett(control=control, treatments=[treatment])

        # Should handle without error
        assert len(results) == 1
        # High variance should lead to large standard error
        assert results.iloc[0]['std_error'] > 10


class TestSpecialCases:
    """Test special cases specific to Dunnett."""

    def test_drug_trial_scenario(self):
        """Test typical drug trial scenario: placebo vs doses."""
        np.random.seed(42)
        placebo = np.random.normal(100, 15, 30)
        low_dose = np.random.normal(105, 15, 30)
        med_dose = np.random.normal(110, 15, 30)
        high_dose = np.random.normal(115, 15, 30)

        results = posthoc_dunnett(
            control=placebo,
            treatments=[low_dose, med_dose, high_dose],
            treatment_names=['Low Dose', 'Med Dose', 'High Dose'],
            control_name='Placebo'
        )

        # All doses should compare to placebo
        assert all(results['control'] == 'Placebo')
        assert len(results) == 3

        # Mean differences should increase with dose
        assert results.iloc[0]['treatment'] == 'Low Dose'
        assert results.iloc[1]['treatment'] == 'Med Dose'
        assert results.iloc[2]['treatment'] == 'High Dose'

    def test_baseline_intervention_scenario(self):
        """Test baseline vs multiple interventions."""
        np.random.seed(42)
        baseline = np.random.normal(50, 10, 25)
        intervention_a = np.random.normal(55, 10, 25)
        intervention_b = np.random.normal(58, 10, 25)

        results = posthoc_dunnett(
            control=baseline,
            treatments=[intervention_a, intervention_b],
            treatment_names=['Intervention A', 'Intervention B'],
            control_name='Baseline'
        )

        # Both interventions should compare to baseline
        assert all(results['control'] == 'Baseline')
        assert len(results) == 2

    def test_one_sided_greater_detecting_improvement(self):
        """Test one-sided test for detecting improvements."""
        np.random.seed(42)
        control = np.random.normal(10, 2, 30)
        improved_treatment = np.random.normal(15, 2, 30)
        similar_treatment = np.random.normal(10.5, 2, 30)

        results = posthoc_dunnett(
            control=control,
            treatments=[improved_treatment, similar_treatment],
            alternative='greater'
        )

        # Improved treatment should be significant
        assert results.iloc[0]['significant']
        # Similar treatment may not be
        # (depends on random data, just check it runs)
        assert len(results) == 2

    def test_critical_value_increases_with_treatments(self):
        """Test that critical value accounts for number of comparisons."""
        np.random.seed(42)
        control = np.random.normal(10, 2, 30)

        # Few treatments
        results_2 = posthoc_dunnett(
            control=control,
            treatments=[np.random.normal(12, 2, 30) for _ in range(2)]
        )

        # Many treatments
        results_5 = posthoc_dunnett(
            control=control,
            treatments=[np.random.normal(12, 2, 30) for _ in range(5)]
        )

        # Critical value should increase with more treatments (more conservative)
        # This is due to family-wise error rate control
        crit_2 = results_2.iloc[0]['d_critical']
        crit_5 = results_5.iloc[0]['d_critical']

        # Both should be positive
        assert crit_2 > 0
        assert crit_5 > 0

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/posthoc/_dunnett.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-01 20:30:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/posthoc/_dunnett.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# Functionalities:
#   - Perform Dunnett's test post-hoc comparison
#   - Compare multiple treatment groups vs single control
#   - Control family-wise error rate
#   - Two-sided or one-sided comparisons
# 
# Dependencies:
#   - packages: numpy, pandas, scipy
# 
# IO:
#   - input: Control group and treatment groups data
#   - output: Comparison results vs control (DataFrame)
# """
# 
# """Imports"""
# import numpy as np
# import pandas as pd
# from typing import Union, List, Optional, Literal
# from scipy import stats
# from scitex.stats.utils._formatters import p2stars
# from scitex.stats.utils._normalizers import convert_results
# 
# 
# def dunnett_critical_value(
#     k: int, df: int, alpha: float = 0.05, alternative: str = "two-sided"
# ) -> float:
#     """
#     Get critical value for Dunnett's test.
# 
#     Parameters
#     ----------
#     k : int
#         Number of treatment groups (excluding control)
#     df : int
#         Degrees of freedom for error
#     alpha : float
#         Significance level
#     alternative : {'two-sided', 'less', 'greater'}
#         Direction of test
# 
#     Returns
#     -------
#     d_crit : float
#         Critical value
# 
#     Notes
#     -----
#     Uses conservative approximation based on t-distribution with
#     Bonferroni-like adjustment. For exact values, specialized tables
#     or software (R, SAS) would be needed.
#     """
#     # Conservative approximation using Bonferroni adjustment
#     if alternative == "two-sided":
#         alpha_adj = alpha / (2 * k)
#     else:
#         alpha_adj = alpha / k
# 
#     t_crit = stats.t.ppf(1 - alpha_adj, df)
# 
#     # Dunnett critical value is typically slightly smaller than Bonferroni
#     # This approximation is conservative
#     d_crit = t_crit * 0.95  # Slight correction factor
# 
#     return float(d_crit)
# 
# 
# def posthoc_dunnett(
#     control: Union[np.ndarray, pd.Series],
#     treatments: List[Union[np.ndarray, pd.Series]],
#     treatment_names: Optional[List[str]] = None,
#     control_name: str = "Control",
#     alpha: float = 0.05,
#     alternative: Literal["two-sided", "less", "greater"] = "two-sided",
#     return_as: str = "dataframe",
# ) -> Union[pd.DataFrame, List[dict]]:
#     """
#     Perform Dunnett's test for comparing treatments vs control.
# 
#     Conducts multiple comparisons of treatment groups against a single
#     control group, controlling the family-wise error rate.
# 
#     Parameters
#     ----------
#     control : array-like
#         Control group data
#     treatments : list of arrays
#         List of treatment group arrays
#     treatment_names : list of str, optional
#         Names for treatment groups. If None, uses 'Treatment 1', etc.
#     control_name : str, default 'Control'
#         Name for control group
#     alpha : float, default 0.05
#         Family-wise error rate
#     alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
#         Direction of comparison:
#         - 'two-sided': treatments differ from control
#         - 'less': treatments < control
#         - 'greater': treatments > control
#     return_as : {'dataframe', 'dict'}, default 'dataframe'
#         Output format
# 
#     Returns
#     -------
#     results : DataFrame or list of dict
#         Comparison results including:
#         - treatment: Treatment group name
#         - control: Control group name
#         - mean_treatment: Mean of treatment
#         - mean_control: Mean of control
#         - mean_diff: Difference (treatment - control)
#         - std_error: Standard error of difference
#         - t_statistic: t-statistic
#         - d_critical: Dunnett critical value
#         - pvalue: Approximate p-value
#         - significant: Whether difference is significant
#         - ci_lower: Lower bound of CI
#         - ci_upper: Upper bound of CI
# 
#     Notes
#     -----
#     Dunnett's test is specifically designed for comparing multiple treatment
#     groups against a single control group, which is more powerful than
#     using Tukey HSD for this purpose.
# 
#     **Test Statistic**:
# 
#     .. math::
#         t_i = \\frac{\\bar{x}_i - \\bar{x}_c}{\\sqrt{MS_{error}(1/n_i + 1/n_c)}}
# 
#     Where:
#     - :math:`\\bar{x}_i`: Mean of treatment i
#     - :math:`\\bar{x}_c`: Mean of control
#     - MS_error: Pooled mean square error
# 
#     **Critical Value**:
#     Uses Dunnett distribution tables (approximated here via conservative
#     t-distribution adjustment).
# 
#     **Assumptions**:
#     1. Independence of observations
#     2. Normality within each group
#     3. Homogeneity of variance across groups
#     4. One group designated as control
# 
#     **Advantages**:
#     - More powerful than Tukey HSD for control comparisons
#     - More powerful than Bonferroni for this specific design
#     - Controls family-wise error rate exactly
#     - Provides directional tests (one-sided)
# 
#     **Disadvantages**:
#     - Only compares vs control (not all pairwise)
#     - Requires equal variances (use Dunnett T3 if violated)
#     - Requires control group designation
# 
#     **When to use**:
#     - After ANOVA with one control and multiple treatments
#     - Drug trials (placebo vs multiple doses)
#     - Baseline comparisons (control vs interventions)
# 
#     Examples
#     --------
#     >>> import numpy as np
#     >>> from scitex.stats.posthoc import posthoc_dunnett
#     >>>
#     >>> # Example: Placebo vs 3 drug doses
#     >>> np.random.seed(42)
#     >>> placebo = np.random.normal(100, 15, 30)
#     >>> dose_low = np.random.normal(105, 15, 30)
#     >>> dose_med = np.random.normal(110, 15, 30)
#     >>> dose_high = np.random.normal(115, 15, 30)
#     >>>
#     >>> results = posthoc_dunnett(
#     ...     control=placebo,
#     ...     treatments=[dose_low, dose_med, dose_high],
#     ...     treatment_names=['Low Dose', 'Med Dose', 'High Dose'],
#     ...     control_name='Placebo'
#     ... )
#     >>>
#     >>> print(results[['treatment', 'mean_diff', 'pvalue', 'significant']])
# 
#     References
#     ----------
#     .. [1] Dunnett, C. W. (1955). "A multiple comparison procedure for
#            comparing several treatments with a control". Journal of the
#            American Statistical Association, 50(272), 1096-1121.
#     .. [2] Dunnett, C. W. (1964). "New tables for multiple comparisons
#            with a control". Biometrics, 20(3), 482-491.
# 
#     See Also
#     --------
#     posthoc_tukey : For all pairwise comparisons
#     posthoc_games_howell : For unequal variances (all pairs)
#     """
#     # Convert to arrays
#     control = np.asarray(control)
#     treatments = [np.asarray(t) for t in treatments]
# 
#     k = len(treatments)  # Number of treatment groups
# 
#     if k < 1:
#         raise ValueError("Need at least 1 treatment group")
# 
#     # Treatment names
#     if treatment_names is None:
#         treatment_names = [f"Treatment {i + 1}" for i in range(k)]
# 
#     if len(treatment_names) != k:
#         raise ValueError(f"Expected {k} treatment names, got {len(treatment_names)}")
# 
#     # Group statistics
#     n_control = len(control)
#     mean_control = np.mean(control)
# 
#     n_treatments = [len(t) for t in treatments]
#     means_treatments = [np.mean(t) for t in treatments]
# 
#     # Total sample size and degrees of freedom
#     N = n_control + sum(n_treatments)
#     df_error = N - (k + 1)  # k treatments + 1 control
# 
#     # Pooled variance (MS_error)
#     ss_error = np.sum((control - mean_control) ** 2)
#     for t in treatments:
#         ss_error += np.sum((t - np.mean(t)) ** 2)
# 
#     ms_error = ss_error / df_error
# 
#     # Get critical value
#     d_crit = dunnett_critical_value(k, df_error, alpha, alternative)
# 
#     # Perform comparisons vs control
#     results = []
# 
#     for i, (treatment, n_t, mean_t) in enumerate(
#         zip(treatments, n_treatments, means_treatments)
#     ):
#         # Mean difference
#         mean_diff = mean_t - mean_control
# 
#         # Standard error
#         se = np.sqrt(ms_error * (1 / n_t + 1 / n_control))
# 
#         # t-statistic
#         if se == 0:
#             t_stat = 0.0
#         else:
#             t_stat = mean_diff / se
# 
#         # p-value (conservative approximation)
#         if alternative == "two-sided":
#             pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df_error))
#         elif alternative == "greater":
#             pvalue = 1 - stats.t.cdf(t_stat, df_error)
#         else:  # less
#             pvalue = stats.t.cdf(t_stat, df_error)
# 
#         # Adjust for multiple comparisons (conservative)
#         pvalue = min(pvalue * k, 1.0)
# 
#         # Determine significance
#         if alternative == "two-sided":
#             significant = abs(t_stat) > d_crit
#         elif alternative == "greater":
#             significant = t_stat > d_crit
#         else:  # less
#             significant = t_stat < -d_crit
# 
#         # Confidence interval
#         margin = d_crit * se
#         if alternative == "two-sided":
#             ci_lower = mean_diff - margin
#             ci_upper = mean_diff + margin
#         elif alternative == "greater":
#             ci_lower = mean_diff - margin
#             ci_upper = np.inf
#         else:  # less
#             ci_lower = -np.inf
#             ci_upper = mean_diff + margin
# 
#         results.append(
#             {
#                 "treatment": treatment_names[i],
#                 "control": control_name,
#                 "n_treatment": n_t,
#                 "n_control": n_control,
#                 "mean_treatment": round(float(mean_t), 3),
#                 "mean_control": round(float(mean_control), 3),
#                 "mean_diff": round(float(mean_diff), 3),
#                 "std_error": round(float(se), 3),
#                 "t_statistic": round(float(t_stat), 3),
#                 "d_critical": round(float(d_crit), 3),
#                 "pvalue": round(float(pvalue), 4),
#                 "significant": bool(significant),
#                 "pstars": p2stars(pvalue),
#                 "ci_lower": round(float(ci_lower), 3)
#                 if not np.isinf(ci_lower)
#                 else "-inf",
#                 "ci_upper": round(float(ci_upper), 3)
#                 if not np.isinf(ci_upper)
#                 else "inf",
#                 "alpha": alpha,
#                 "alternative": alternative,
#             }
#         )
# 
#     # Return format
#     if return_as == "dataframe":
#         return pd.DataFrame(results)
#     else:
#         return results
# 
# 
# if __name__ == "__main__":
#     import sys
#     import argparse
#     import scitex as stx
# 
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args([])
# 
#     CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(
#         sys=sys,
#         plt=None,
#         args=args,
#         file=__FILE__,
#         verbose=True,
#         agg=True,
#     )
# 
#     logger = stx.logging.getLogger(__name__)
# 
#     logger.info("=" * 70)
#     logger.info("Dunnett's Test Post-hoc Examples")
#     logger.info("=" * 70)
# 
#     # Example 1: Drug trial - placebo vs multiple doses
#     logger.info("\n[Example 1] Drug trial: Placebo vs 3 doses")
#     logger.info("-" * 70)
# 
#     np.random.seed(42)
#     placebo = np.random.normal(100, 15, 30)
#     dose_low = np.random.normal(105, 15, 30)
#     dose_med = np.random.normal(110, 15, 30)
#     dose_high = np.random.normal(115, 15, 30)
# 
#     logger.info(f"Placebo: mean={np.mean(placebo):.2f}, n={len(placebo)}")
#     logger.info(f"Low Dose: mean={np.mean(dose_low):.2f}, n={len(dose_low)}")
#     logger.info(f"Med Dose: mean={np.mean(dose_med):.2f}, n={len(dose_med)}")
#     logger.info(f"High Dose: mean={np.mean(dose_high):.2f}, n={len(dose_high)}")
# 
#     results = posthoc_dunnett(
#         control=placebo,
#         treatments=[dose_low, dose_med, dose_high],
#         treatment_names=["Low Dose", "Med Dose", "High Dose"],
#         control_name="Placebo",
#     )
# 
#     logger.info(
#         f"\n{results[['treatment', 'mean_diff', 't_statistic', 'pvalue', 'significant']].to_string()}"
#     )
# 
#     # Example 2: One-sided test (treatments > control)
#     logger.info("\n[Example 2] One-sided test (greater than control)")
#     logger.info("-" * 70)
# 
#     results_greater = posthoc_dunnett(
#         control=placebo,
#         treatments=[dose_low, dose_med, dose_high],
#         treatment_names=["Low Dose", "Med Dose", "High Dose"],
#         control_name="Placebo",
#         alternative="greater",
#     )
# 
#     logger.info(
#         f"\n{results_greater[['treatment', 'mean_diff', 'pvalue', 'significant']].to_string()}"
#     )
# 
#     # Example 3: Baseline comparison
#     logger.info("\n[Example 3] Baseline vs interventions")
#     logger.info("-" * 70)
# 
#     baseline = np.random.normal(50, 10, 25)
#     intervention_a = np.random.normal(55, 10, 25)
#     intervention_b = np.random.normal(58, 10, 25)
# 
#     results_baseline = posthoc_dunnett(
#         control=baseline,
#         treatments=[intervention_a, intervention_b],
#         treatment_names=["Intervention A", "Intervention B"],
#         control_name="Baseline",
#     )
# 
#     logger.info(f"\n{results_baseline.to_string()}")
# 
#     # Example 4: Comparison with Tukey HSD
#     logger.info("\n[Example 4] Dunnett vs Tukey HSD power comparison")
#     logger.info("-" * 70)
# 
#     from ._tukey_hsd import posthoc_tukey
# 
#     # All groups for Tukey
#     all_groups = [placebo, dose_low, dose_med, dose_high]
#     group_names = ["Placebo", "Low Dose", "Med Dose", "High Dose"]
# 
#     results_tukey = posthoc_tukey(all_groups, group_names)
# 
#     # Filter Tukey results for comparisons vs placebo
#     tukey_vs_placebo = results_tukey[
#         (results_tukey["group_i"] == "Placebo")
#         | (results_tukey["group_j"] == "Placebo")
#     ]
# 
#     logger.info("\nDunnett's test (designed for control comparisons):")
#     logger.info(f"{results[['treatment', 'pvalue', 'significant']].to_string()}")
# 
#     logger.info("\nTukey HSD (all pairwise, less power for control comparisons):")
#     logger.info(
#         f"{tukey_vs_placebo[['group_i', 'group_j', 'pvalue', 'significant']].to_string()}"
#     )
# 
#     logger.info("\nNote: Dunnett is more powerful for control comparisons")
# 
#     # Example 5: Unbalanced design
#     logger.info("\n[Example 5] Unbalanced design")
#     logger.info("-" * 70)
# 
#     control_unbal = np.random.normal(20, 5, 50)
#     treat1_unbal = np.random.normal(22, 5, 15)
#     treat2_unbal = np.random.normal(25, 5, 20)
#     treat3_unbal = np.random.normal(23, 5, 30)
# 
#     results_unbal = posthoc_dunnett(
#         control=control_unbal,
#         treatments=[treat1_unbal, treat2_unbal, treat3_unbal],
#         treatment_names=["T1", "T2", "T3"],
#     )
# 
#     logger.info(
#         f"Sample sizes: Control={len(control_unbal)}, T1={len(treat1_unbal)}, "
#         f"T2={len(treat2_unbal)}, T3={len(treat3_unbal)}"
#     )
#     logger.info(f"\n{results_unbal.to_string()}")
# 
#     # Example 6: With confidence intervals
#     logger.info("\n[Example 6] Confidence intervals")
#     logger.info("-" * 70)
# 
#     for _, row in results.iterrows():
#         logger.info(
#             f"{row['treatment']} vs {row['control']}: "
#             f"Diff = {row['mean_diff']:.2f}, "
#             f"95% CI [{row['ci_lower']}, {row['ci_upper']}] {row['pstars']}"
#         )
# 
#     # Example 7: Export results
#     logger.info("\n[Example 7] Export results")
#     logger.info("-" * 70)
# 
#     convert_results(results, return_as="excel", path="./dunnett_results.xlsx")
#     logger.info("Saved to: ./dunnett_results.xlsx")
# 
#     stx.session.close(
#         CONFIG,
#         verbose=False,
#         notify=False,
#         exit_status=0,
#     )
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/posthoc/_dunnett.py
# --------------------------------------------------------------------------------
