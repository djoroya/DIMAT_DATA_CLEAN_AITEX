def summary(df):
    print("Data Summary:")
    print(f"Number of rows: \t \t{len(df)}")
    print(f"Number of columns: \t \t{len(df.columns)}")
    print(f"Total missing values: \t\t{df.isnull().sum().sum()} from {df.size} cells")
    # percentage of missing values
    total_cells = df.size
    total_missing = df.isnull().sum().sum()
    missing_percentage = (total_missing / total_cells) * 100
    # contamos las rows que tiene algun valor nulo en porcentage
    
    try:
        df_numeric = df.select_dtypes(include=['number'])
        rows_with_missing = df_numeric.sum(axis=1,skipna=False).isna().sum()

        print(f"Rows with missing values: \t{rows_with_missing}")
        print(f"% of rows with missing values: \t{(rows_with_missing / len(df)) * 100:.2f}%")
    except Exception as e:
        print("Error calculating rows with missing values:", e)
        rows_with_missing = 0

    
    print(f"Percentage of missing values: \t{missing_percentage:.2f}%")
    