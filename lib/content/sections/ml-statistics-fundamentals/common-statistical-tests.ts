/**
 * Common Statistical Tests Section
 */

export const commonstatisticaltestsSection = {
  id: 'common-statistical-tests',
  title: 'Common Statistical Tests',
  content: `# Common Statistical Tests

## Introduction

Statistical tests are the tools we use to make decisions about data. In machine learning, we constantly ask questions like:

- Is this feature significantly associated with the target?
- Are these two models statistically different in performance?
- Do different user groups have significantly different outcomes?
- Has model performance significantly degraded?

Each question requires the right statistical test. Choosing the wrong test can lead to incorrect conclusions!

## Decision Tree for Choosing Tests

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

np.random.seed(42)

print("=== Choosing the Right Test ===")
print()
print("Question 1: What are you comparing?")
print("  → One group to a value? → One-sample t-test")
print("  → Two independent groups? → Two-sample t-test or Mann-Whitney")
print("  → Two related groups (paired)? → Paired t-test or Wilcoxon")
print("  → Three or more groups? → ANOVA or Kruskal-Wallis")
print()
print("Question 2: What type of data?")
print("  → Continuous + Normal? → Parametric tests (t-test, ANOVA)")
print("  → Continuous + Not normal? → Non-parametric (Mann-Whitney, Kruskal-Wallis)")
print("  → Categorical? → Chi-square test")
print()
print("Question 3: Are assumptions met?")
print("  → Check normality, equal variances, independence")
\`\`\`

## Z-Test

**Use case**: Compare sample mean to population mean when σ is known (rare in practice)

**Assumptions**:
- Population standard deviation known
- Normal distribution or large sample (n > 30)

\`\`\`python
def z_test_example():
    """Z-test: Is sample mean different from population mean?"""
    
    # Known: Population mean IQ = 100, σ = 15
    pop_mean = 100
    pop_std = 15
    
    # Sample: 50 ML practitioners
    sample_size = 50
    sample = np.random.normal(105, 15, sample_size)  # Slightly higher
    sample_mean = sample.mean()
    
    # Z-statistic
    se = pop_std / np.sqrt(sample_size)
    z_stat = (sample_mean - pop_mean) / se
    
    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    print("=== Z-Test Example ===")
    print(f"H₀: μ = {pop_mean} (ML practitioners have average IQ)")
    print(f"H₁: μ ≠ {pop_mean} (ML practitioners have different IQ)")
    print()
    print(f"Population mean: {pop_mean}")
    print(f"Population σ: {pop_std}")
    print(f"Sample mean: {sample_mean:.2f}")
    print(f"Sample size: {sample_size}")
    print(f"Standard error: {se:.2f}")
    print(f"Z-statistic: {z_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print()
    
    if p_value < 0.05:
        print("✓ Reject H₀: Sample mean is significantly different")
    else:
        print("✗ Fail to reject H₀: No significant difference")

z_test_example()
\`\`\`

## T-Tests

### One-Sample T-Test

**Use case**: Compare sample mean to a known value when σ is unknown

\`\`\`python
def one_sample_ttest():
    """Is model accuracy significantly different from 0.75?"""
    
    # H₀: accuracy = 0.75
    # H₁: accuracy ≠ 0.75
    
    target_accuracy = 0.75
    accuracies = np.random.normal(0.78, 0.05, 30)  # 30 cross-validation runs
    
    # Perform t-test
    t_stat, p_value = stats.ttest_1samp(accuracies, target_accuracy)
    
    mean_acc = accuracies.mean()
    se = stats.sem(accuracies)
    ci = stats.t.interval(0.95, len(accuracies)-1, mean_acc, se)
    
    print("=== One-Sample T-Test ===")
    print(f"Testing if accuracy differs from {target_accuracy}")
    print(f"Sample mean: {mean_acc:.4f}")
    print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print()
    
    if p_value < 0.05:
        print(f"✓ Model accuracy ({mean_acc:.4f}) is significantly different from {target_accuracy}")
    else:
        print(f"✗ No significant difference from {target_accuracy}")

one_sample_ttest()
\`\`\`

### Two-Sample T-Test (Independent)

**Use case**: Compare means of two independent groups

\`\`\`python
def two_sample_ttest():
    """Compare two models' accuracies"""
    
    # Model A and Model B performances on different test sets
    model_a = np.random.normal(0.75, 0.05, 30)
    model_b = np.random.normal(0.78, 0.05, 30)
    
    # Levene's test for equal variances
    _, p_levene = stats.levene(model_a, model_b)
    equal_var = p_levene > 0.05
    
    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(model_a, model_b, equal_var=equal_var)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((model_a.var() + model_b.var()) / 2)
    cohens_d = (model_b.mean() - model_a.mean()) / pooled_std
    
    print("=== Two-Sample T-Test ===")
    print(f"Model A mean: {model_a.mean():.4f} (n={len(model_a)})")
    print(f"Model B mean: {model_b.mean():.4f} (n={len(model_b)})")
    print(f"Difference: {model_b.mean() - model_a.mean():.4f}")
    print()
    print(f"Equal variances? {equal_var} (p_levene={p_levene:.4f})")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Cohen's d: {cohens_d:.4f} ({interpret_cohens_d(cohens_d)})")
    print()
    
    if p_value < 0.05:
        print("✓ Model B is significantly different from Model A")
    else:
        print("✗ No significant difference between models")

def interpret_cohens_d(d):
    """Interpret Cohen's d effect size"""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

two_sample_ttest()
\`\`\`

### Paired T-Test

**Use case**: Compare two related measurements (before/after, same subjects)

\`\`\`python
def paired_ttest():
    """Compare model performance before and after feature engineering"""
    
    n_samples = 25
    before = np.random.normal(0.70, 0.05, n_samples)
    improvement = np.random.normal(0.05, 0.02, n_samples)  # Consistent improvement
    after = before + improvement
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(after, before)
    
    differences = after - before
    mean_diff = differences.mean()
    se_diff = stats.sem(differences)
    ci = stats.t.interval(0.95, len(differences)-1, mean_diff, se_diff)
    
    print("=== Paired T-Test ===")
    print(f"Before feature engineering: {before.mean():.4f}")
    print(f"After feature engineering: {after.mean():.4f}")
    print(f"Mean improvement: {mean_diff:.4f}")
    print(f"95% CI for improvement: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print()
    
    if p_value < 0.05:
        print("✓ Feature engineering significantly improved performance")
    else:
        print("✗ No significant improvement")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before-After plot
    for i in range(n_samples):
        axes[0].plot([0, 1], [before[i], after[i]], 'o-', alpha=0.3, color='gray')
    axes[0].plot([0, 1], [before.mean(), after.mean()], 'ro-', linewidth=3, markersize=10, label='Mean')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Before', 'After'])
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Paired Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Difference distribution
    axes[1].hist(differences, bins=15, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--', linewidth=2, label='No change')
    axes[1].axvline(mean_diff, color='g', linestyle='-', linewidth=2, label=f'Mean: {mean_diff:.4f}')
    axes[1].set_xlabel('Improvement (After - Before)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Improvements')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('paired_ttest.png', dpi=300, bbox_inches='tight')
    plt.show()

paired_ttest()
\`\`\`

## Chi-Square Test

**Use case**: Test independence between categorical variables

\`\`\`python
def chi_square_test():
    """Are user type and model prediction independent?"""
    
    # Contingency table: User Type vs Prediction
    # Rows: Prediction (Correct, Incorrect)
    # Columns: User Type (Premium, Free)
    
    observed = np.array([
        [450, 350],  # Correct predictions
        [50, 150]    # Incorrect predictions
    ])
    
    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    
    print("=== Chi-Square Test for Independence ===")
    print("H₀: User type and prediction accuracy are independent")
    print("H₁: User type and prediction accuracy are related")
    print()
    print("Observed frequencies:")
    print(pd.DataFrame(observed, 
                       index=['Correct', 'Incorrect'],
                       columns=['Premium', 'Free']))
    print()
    print("Expected frequencies (if independent):")
    print(pd.DataFrame(expected,
                       index=['Correct', 'Incorrect'],
                       columns=['Premium', 'Free']))
    print()
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_value:.6f}")
    print()
    
    if p_value < 0.05:
        print("✓ Reject H₀: User type and accuracy are related")
        print("  Model performs differently for different user types!")
    else:
        print("✗ Fail to reject H₀: No significant relationship")
    
    # Calculate and show proportions
    print()
    print("Accuracy by user type:")
    print(f"Premium users: {observed[0,0]/(observed[0,0]+observed[1,0]):.2%}")
    print(f"Free users: {observed[0,1]/(observed[0,1]+observed[1,1]):.2%}")

chi_square_test()
\`\`\`

## ANOVA (Analysis of Variance)

**Use case**: Compare means of three or more groups

\`\`\`python
def anova_test():
    """Compare three models' performances"""
    
    # Three models tested on different datasets
    model_a = np.random.normal(0.75, 0.05, 25)
    model_b = np.random.normal(0.78, 0.05, 25)
    model_c = np.random.normal(0.76, 0.05, 25)
    
    # One-way ANOVA
    f_stat, p_value = stats.f_oneway(model_a, model_b, model_c)
    
    print("=== One-Way ANOVA ===")
    print("H₀: All models have equal mean accuracy")
    print("H₁: At least one model has different mean accuracy")
    print()
    print(f"Model A mean: {model_a.mean():.4f}")
    print(f"Model B mean: {model_b.mean():.4f}")
    print(f"Model C mean: {model_c.mean():.4f}")
    print()
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    print()
    
    if p_value < 0.05:
        print("✓ Reject H₀: At least one model is significantly different")
        print("\\nPost-hoc tests needed to determine which models differ:")
        
        # Pairwise t-tests with Bonferroni correction
        pairs = [('A', 'B', model_a, model_b),
                 ('A', 'C', model_a, model_c),
                 ('B', 'C', model_b, model_c)]
        
        alpha_corrected = 0.05 / len(pairs)  # Bonferroni
        print(f"\\nPairwise comparisons (α = {alpha_corrected:.4f}):")
        
        for name1, name2, data1, data2 in pairs:
            _, p = stats.ttest_ind(data1, data2)
            sig = "✓" if p < alpha_corrected else "✗"
            print(f"  {name1} vs {name2}: p={p:.6f} {sig}")
    else:
        print("✗ Fail to reject H₀: No significant difference among models")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = [model_a, model_b, model_c]
    positions = [1, 2, 3]
    
    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_xticklabels(['Model A', 'Model B', 'Model C'])
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'ANOVA: Comparing Multiple Models\\n(F={f_stat:.2f}, p={p_value:.4f})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('anova_test.png', dpi=300, bbox_inches='tight')
    plt.show()

anova_test()
\`\`\`

## Non-Parametric Tests

When assumptions of parametric tests aren't met (non-normal, small sample, outliers), use non-parametric tests.

### Mann-Whitney U Test

**Non-parametric alternative to two-sample t-test**

\`\`\`python
def mann_whitney_test():
    """Compare two models with non-normal distributions"""
    
    # Skewed performance distributions (not normal)
    model_a = np.random.exponential(0.75, 30)
    model_b = np.random.exponential(0.80, 30)
    
    # Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(model_a, model_b, alternative='two-sided')
    
    print("=== Mann-Whitney U Test ===")
    print("(Non-parametric alternative to two-sample t-test)")
    print()
    print(f"Model A median: {np.median(model_a):.4f}")
    print(f"Model B median: {np.median(model_b):.4f}")
    print(f"U-statistic: {u_stat:.2f}")
    print(f"P-value: {p_value:.6f}")
    print()
    
    if p_value < 0.05:
        print("✓ Models have significantly different distributions")
    else:
        print("✗ No significant difference")

mann_whitney_test()
\`\`\`

### Wilcoxon Signed-Rank Test

**Non-parametric alternative to paired t-test**

\`\`\`python
def wilcoxon_test():
    """Paired comparison with non-normal differences"""
    
    n = 20
    before = np.random.exponential(0.70, n)
    after = before + np.random.exponential(0.05, n)
    
    # Wilcoxon signed-rank test
    w_stat, p_value = stats.wilcoxon(after, before, alternative='greater')
    
    print("=== Wilcoxon Signed-Rank Test ===")
    print("(Non-parametric alternative to paired t-test)")
    print()
    print(f"Median before: {np.median(before):.4f}")
    print(f"Median after: {np.median(after):.4f}")
    print(f"Median improvement: {np.median(after - before):.4f}")
    print(f"W-statistic: {w_stat:.2f}")
    print(f"P-value: {p_value:.6f}")
    print()
    
    if p_value < 0.05:
        print("✓ Significant improvement detected")
    else:
        print("✗ No significant improvement")

wilcoxon_test()
\`\`\`

### Kruskal-Wallis Test

**Non-parametric alternative to ANOVA**

\`\`\`python
def kruskal_wallis_test():
    """Compare three or more groups (non-parametric)"""
    
    model_a = np.random.exponential(0.75, 25)
    model_b = np.random.exponential(0.80, 25)
    model_c = np.random.exponential(0.77, 25)
    
    # Kruskal-Wallis test
    h_stat, p_value = stats.kruskal(model_a, model_b, model_c)
    
    print("=== Kruskal-Wallis Test ===")
    print("(Non-parametric alternative to ANOVA)")
    print()
    print(f"Model A median: {np.median(model_a):.4f}")
    print(f"Model B median: {np.median(model_b):.4f}")
    print(f"Model C median: {np.median(model_c):.4f}")
    print(f"H-statistic: {h_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    print()
    
    if p_value < 0.05:
        print("✓ At least one model has significantly different distribution")
    else:
        print("✗ No significant difference among models")

kruskal_wallis_test()
\`\`\`

## Assumptions Checking

\`\`\`python
def check_assumptions(data1, data2=None):
    """Check assumptions for parametric tests"""
    
    print("=== Checking Test Assumptions ===")
    print()
    
    # 1. Normality (Shapiro-Wilk test)
    _, p_norm1 = stats.shapiro(data1)
    print(f"Group 1 normality: p={p_norm1:.4f} {'✓' if p_norm1 > 0.05 else '✗ Use non-parametric'}")
    
    if data2 is not None:
        _, p_norm2 = stats.shapiro(data2)
        print(f"Group 2 normality: p={p_norm2:.4f} {'✓' if p_norm2 > 0.05 else '✗ Use non-parametric'}")
        
        # 2. Equal variances (Levene's test)
        _, p_var = stats.levene(data1, data2)
        print(f"Equal variances: p={p_var:.4f} {'✓' if p_var > 0.05 else '✗ Use Welch t-test'}")
    
    print()
    print("Recommendation:")
    if p_norm1 > 0.05 and (data2 is None or p_norm2 > 0.05):
        print("  ✓ Use parametric test (t-test, ANOVA)")
    else:
        print("  → Use non-parametric test (Mann-Whitney, Kruskal-Wallis)")

# Example
data_normal = np.random.normal(0.75, 0.05, 30)
data_skewed = np.random.exponential(0.75, 30)

check_assumptions(data_normal, data_skewed)
\`\`\`

## ML-Specific Applications

\`\`\`python
def ml_feature_significance():
    """Test if a feature significantly improves model"""
    
    # Model performance with and without feature
    without_feature = np.random.normal(0.75, 0.03, 20)
    with_feature = np.random.normal(0.77, 0.03, 20)
    
    # Paired t-test (same CV folds)
    t_stat, p_value = stats.ttest_rel(with_feature, without_feature)
    
    print("=== Feature Significance Test ===")
    print(f"Without feature: {without_feature.mean():.4f}")
    print(f"With feature: {with_feature.mean():.4f}")
    print(f"Improvement: {(with_feature - without_feature).mean():.4f}")
    print(f"P-value: {p_value:.6f}")
    print()
    
    if p_value < 0.05:
        print("✓ Feature significantly improves model → Include it")
    else:
        print("✗ Feature doesn't significantly help → Consider excluding")

ml_feature_significance()
\`\`\`

## Key Takeaways

1. **Choose test based on**: Data type, number of groups, independence, normality
2. **Parametric tests**: Assume normal distribution, more powerful if assumptions met
3. **Non-parametric tests**: No distribution assumptions, robust to outliers
4. **Check assumptions**: Normality (Shapiro-Wilk), equal variance (Levene's)
5. **Effect size matters**: Complement p-values with Cohen's d or similar
6. **Multiple comparisons**: Apply corrections (Bonferroni, FDR)
7. **Paired tests**: More powerful when measurements are related
8. **ANOVA**: For 3+ groups; follow up with post-hoc tests

## Test Selection Guide

| Situation | Parametric | Non-Parametric |
|-----------|------------|----------------|
| 1 sample vs value | One-sample t-test | Wilcoxon signed-rank |
| 2 independent groups | Two-sample t-test | Mann-Whitney U |
| 2 paired groups | Paired t-test | Wilcoxon signed-rank |
| 3+ independent groups | ANOVA | Kruskal-Wallis |
| Categorical variables | Chi-square | Fisher's exact |

## Connection to Machine Learning

- **Model comparison**: Which model is significantly better?
- **Feature selection**: Does feature significantly improve performance?
- **A/B testing**: Is variant B significantly different from A?
- **Hyperparameter significance**: Does parameter change significantly affect results?
- **Performance monitoring**: Has performance significantly degraded?
- **Group differences**: Do predictions differ significantly across demographics?
- **Intervention effects**: Did model update significantly change outcomes?

Master these tests and you can rigorously evaluate every decision in your ML pipeline!
`,
};
