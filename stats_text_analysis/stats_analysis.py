from scipy import stats
from statsmodels.stats.diagnostic import kstest_normal

def perform_statistical_analysis(sentiment1, sentiment2, alpha=0.05):
    print(
        "Performing statistical analysis of sentiment scores by LLM and traditional methods..."
    )

    # Single-Sample K-S Test
    print(
        "Examining Whether the Sentiment Scores are Normally Distributed By Single-Sample K-S Test..."
    )
    stats_ks_test1, p_value1 = kstest_normal(sentiment1)
    stats_ks_test2, p_value2 = kstest_normal(sentiment2)
    print(f"K-S Statistics of LLM: {stats_ks_test1}, p-value: {p_value1}")
    print(
        f"K-S Statistics of Traditional Methods: {stats_ks_test2}, p-value: {p_value2}"
    )
    if p_value1 > alpha:
        print("The first kind of sentiment scores are normally distributed.")
    else:
        print("The first kind of sentiment scores are not normally distributed.")
    if p_value1 > alpha:
        print("The second kind of sentiment scores are normally distributed.")
    else:
        print("The second kind of sentiment scores are not normally distributed.")

    # Two-Sample K-S Test
    print("Performing Two-Sample K-S Test...")
    ks_statistic_both, p_value_both = stats.ks_2samp(sentiment1, sentiment2)
    print(f"K-S Statistics: {ks_statistic_both}, p-value: {p_value_both}")
    if p_value_both > alpha:
        print("The two kinds of sentiment scores are from the same distribution.")
    else:
        print("The two kinds of sentiment scores are not from the same distribution.")
