import pandas as pd

def most_correlation(df_01,num=20):

    # Convert all columns that can be to numeric for correlation analysis
    df_numeric = df_01.apply(pd.to_numeric, errors='coerce')

    # Drop columns with all NaNs (non-numeric or unparseable)
    df_numeric = df_numeric.dropna(axis=1, how='all')

    # Compute the correlation matrix
    correlation_matrix = df_numeric.corr()

    # Find the pairs with the highest correlations (absolute value)
    corr_unstacked = correlation_matrix.abs().unstack().sort_values(ascending=False)

    # Remove self-correlations (correlation of 1)
    corr_filtered = corr_unstacked[corr_unstacked < 1]

    # 1e4 precesion to avoid floating point issues
    corr_filtered = corr_filtered.round(5)
    # Drop duplicate pairs (since correlation is symmetric)
    corr_filtered = corr_filtered[~corr_filtered.index.duplicated()]

    # Show the top 10 most correlated variable pairs
    top_correlations = corr_filtered.head(2*num)
    top_correlations

    # se repiten cada dos 
    top_correlations = top_correlations.iloc[::2]

    return top_correlations,df_numeric