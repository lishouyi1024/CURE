import os
import argparse
import pandas as pd

def concatenate_matched_pairs(input_folder, output_file):
    dfs = []

    # Loop through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)
            # Read the matched pairs file
            df = pd.read_csv(file_path, sep='\t')  # tab-separated txt files
            dfs.append(df)

    # Concatenate all dataframes
    concatenated_df = pd.concat(dfs, ignore_index=True)

    concatenated_df.rename(columns={'imc_CD45': 'CD45', 'imc_CD3' : 'CD3',
                                    'imc_CD4' : 'CD4', 'imc_CD8a' : 'CD8a',
                                    'imc_CD20' : 'CD20', 'imc_CD56' : 'CD56',
                                    'imc_CD14' : 'CD14', 'imc_CD68' : 'CD68'}, 
                                    inplace=True)

    # Save the concatenated dataframe to a new text file
    concatenated_df.to_csv(output_file, sep='\t', index=False)
    print(f"All matched pairs have been concatenated into {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Concatenate matched pairs text files.")
    parser.add_argument(
        "--input_folder", required=True, 
        help="The folder containing the matched pairs text files.")
    
    parser.add_argument(
        "--output_file", required=True, 
        help="The output text file path.")
    
    args = parser.parse_args()
    
    concatenate_matched_pairs(args.input_folder, args.output_file)

if __name__ == "__main__":
    main()
