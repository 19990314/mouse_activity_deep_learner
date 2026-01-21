import pandas as pd


def calculate_column_alignment(csv_path, col1, col2):
    merged_labels = {
        0: "turn",
        1: "forward",
        2: "still",
        3: "explore",
        4: "rear",
        5: "groom",
    }

    try:
        # 1. Load the dataset
        df = pd.read_csv(csv_path)

        # Check if columns exist
        if col1 not in df.columns or col2 not in df.columns:
            print(f"Error: One or both columns ('{col1}', '{col2}') not found in CSV.")
            return

        # 2. Handle missing values
        # (Comparing two NaNs usually returns False in Python, so we fill them
        # or drop them depending on your needs. Here we drop for a clean comparison.)
        df_clean = df[[col1, col2]].dropna()
        total_rows = len(df_clean)

        if total_rows == 0:
            print("Error: No data to compare after removing empty rows.")
            return

        # 3. Calculate alignment (Exact matches)
        # This creates a boolean Series where True = Match
        matches = (df_clean[col1] == df_clean[col2]).sum()
        percent_alignment = (matches / total_rows) * 100

        # 4. Output results
        print(f"--- Alignment Results ---")
        print(f"Total valid rows: {total_rows}")
        print(f"Matching rows:    {matches}")
        print(f"Alignment:        {percent_alignment:.2f}%")

        #5
        print(f"--- Detailed Results ---")
        for i in merged_labels.keys():
            mask = df_clean[col1] == i
            matches = (df_clean[col1] == df_clean[col2] & mask).sum()
            if mask.sum() != 0:
                percent_alignment = (matches / total_rows) * 100
            print(f"{merged_labels[i]}  Acc:        {percent_alignment:.2f}%")
        return percent_alignment

    except Exception as e:
        print(f"An error occurred: {e}")


# --- Example Usage ---
# Replace 'data.csv' with your filename and the column names with your actual headers
calculate_column_alignment(r"D:\My Drive\moseq_proj\data\videos\test\sc04_d3_10mintest_annotations.csv", 'human_labeled_state', 'syllable_merged')