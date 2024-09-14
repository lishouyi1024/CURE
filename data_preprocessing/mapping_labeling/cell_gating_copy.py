import pandas as pd
import argparse

def immune_phenotyping(input_file, output_file, thresholds):
    df = pd.read_table(input_file, sep='\t')

    # Apply CD45 gating
    df['CD45_Pos'] = df['CD45'] > thresholds['CD45']

    total = len(df)

    # Instead of discarding CD45- cells, initialize all columns to 'none' or False as appropriate
    # Initialize gating columns
    for column in ['T_cells', 'CD4_Pos', 'CD4_Neg', 'B_cells', 'NK_cells', 'Monocytes_and_Granulocytes']:
        df[column] = False

    # Initialize classification columns
    df['Binary_classification'] = 'none'
    df['Multiclass_5'] = 'none'
    df['Multiclass_6'] = 'none'

    # Apply gating only to CD45+ cells
    mask = df['CD45_Pos']

    # Now apply the gating to only CD45+ cells, updating only those rows
    df.loc[mask, 'T_cells'] = (df['CD3'] > thresholds['CD3']) & mask
    df.loc[mask, 'CD4_Pos'] = (df['T_cells'] & (df['CD4'] > thresholds['CD4']))
    df.loc[mask, 'CD4_Neg'] = (df['T_cells'] & ~(df['CD4'] > thresholds['CD4']))
    df.loc[mask, 'B_cells'] = ~(df['CD3'] > thresholds['CD3']) & (df['CD20'] > thresholds['CD20']) & mask
    df.loc[mask, 'NK_cells'] = ~(df['CD3'] > thresholds['CD3']) & ~(df['CD20'] > thresholds['CD20']) & (df['CD56'] > thresholds['CD56']) & mask
    df.loc[mask, 'Monocytes_and_Granulocytes'] = ~(df['CD3'] > thresholds['CD3']) & ~(df['CD20'] > thresholds['CD20']) & ~(df['CD56'] > thresholds['CD56']) & mask
    # df.loc[mask, 'Granulocytes'] = ~(df['CD3'] > thresholds['CD3']) & ~(df['CD20'] > thresholds['CD20']) & ~(df['CD56'] > thresholds['CD56']) & ~((df['CD14'] > thresholds['CD14']) | (df['CD68'] > thresholds['CD68'])) & mask

    # Update binary classification for CD45+ cells
    def binary_classification(row):
        if not row['CD45_Pos']:
            return 'none'
        elif row['T_cells'] or row['B_cells'] or row['NK_cells']:
            return 'Lymphocytes'
        else:
            return 'Myelocytes'
    df['Binary_classification'] = df.apply(binary_classification, axis=1)

    # Update multiclass classification for CD45+ cells
    def classify_5_classes(row):
        if not row['CD45_Pos']:
            return 'none'
        elif row['T_cells']:
            return 'T-cells'
        elif row['B_cells']:
            return 'B-cells'
        elif row['NK_cells']:
            return 'NK-cells'
        else:
            return 'Monocytes_and_Granulocytes'
    df['Multiclass_5'] = df.apply(classify_5_classes, axis=1)

    def classify_6_classes(row):
        if not row['CD45_Pos']:
            return 'none'
        elif row['CD4_Pos']:
            return 'T-cells_CD4+'
        elif row['CD4_Neg']:
            return 'T-cells_CD4-'
        elif row['B_cells']:
            return 'B-cells'
        elif row['NK_cells']:
            return 'NK-cells'
        else:
            return 'Monocytes_and_Granulocytes'
    df['Multiclass_6'] = df.apply(classify_6_classes, axis=1)

    # Print the counts
    print(f"Total cells: {total}")
    print(f"CD45+ cells: {df['CD45_Pos'].sum()}")
    print(f"T cells (CD3+): {df['T_cells'].sum()}")
    print(f"CD4+ T cells: {df['CD4_Pos'].sum()}")
    print(f"CD4- T cells: {df['CD4_Neg'].sum()}")
    print(f"B cells (CD20+): {df['B_cells'].sum()}")
    print(f"NK cells (CD56+): {df['NK_cells'].sum()}")
    print(f"Monocytes_and_Granulocytes: {df['Monocytes_and_Granulocytes'].sum()}")

    print()

    print(f"Binary_classification - Lymphocytes: {(df['Binary_classification'] == 'Lymphocytes').sum()}")

    # Save the gated cells to a new text file
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Successfully gated all cells and added classification to {output_file}")


thresholds = {
    'CD45': 1.0, 
    'CD3': 0.5,
    'CD4': 0.5,
    'CD20': 0.5,
    'CD56': 0.5,
    'CD14': 0.5,
}

def main():
    parser = argparse.ArgumentParser(description="Concatenate matched pairs text files.")

    parser.add_argument(
        "--input_file", required=True, 
        help="The input file containing the matched pairs text files.")
    
    parser.add_argument(
        "--output_file", required=True, 
        help="The output text file path.")
    
    args = parser.parse_args()
    
    immune_phenotyping(args.input_file, args.output_file, thresholds)

if __name__ == "__main__":
    main()