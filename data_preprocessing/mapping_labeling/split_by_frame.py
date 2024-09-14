import pandas as pd
import os
import argparse

def split_by_frame(input_file, output_dir):
    # Read the input file
    data = pd.read_csv(input_file, sep='\t')

    # Check if output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Group by frame_id and write separate files
    for frame_id, group in data.groupby('frame_id'):
        output_file = os.path.join(output_dir, f'frame_{frame_id}.txt')
        group.to_csv(output_file, sep='\t', index=False)

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Split a text file by frame_id.')
    parser.add_argument('input_file', type=str, help='Input text file path')
    parser.add_argument('output_dir', type=str, help='Output directory path')

    # Parse arguments
    args = parser.parse_args()

    # Run the function
    split_by_frame(args.input_file, args.output_dir)
