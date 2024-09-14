import argparse
import subprocess
import os
import pandas as pd

# def get_center_coordinates(reference_file, frame_id):
#     df = pd.read_csv(reference_file, sep='\t')
#     # print("Column names:", df.columns)
#     frame_data = df[df['frame_id'] == frame_id]
    
#     if not frame_data.empty:
#         return frame_data.iloc[0]['x'], frame_data.iloc[0]['y']
#     else:
#         raise ValueError(f"No center coordinates found for frame ID {frame_id}")

def transform_whole_slide(imc_folder, if_folder, reference_file, scaling, flipping, slide_id, base_dir):
    # load reference file
    df = pd.read_csv(reference_file, sep='\t')
    
    # loop through IMC files in the folder
    for filename in os.listdir(imc_folder):
        if filename.endswith("full.csv"):
            # Split the filename and check if it has at least 7 parts (index 6)
            parts = filename.split('_')
            if len(parts) > 6 and parts[6].isdigit():
                frame_id = int(parts[6])  
            else:
                continue  # Skip this file if index 6 does not exist or is not a digit
            
            # cell_id = int(filename.split('_')[7]) 
            # print(frame_id, cell_id)
            frame_data = df[df['frame_id'] == frame_id]
            if not frame_data.empty:
                x_center, y_center = frame_data.iloc[0]['x'], frame_data.iloc[0]['y']
            else:
                continue
            imc_data_path = os.path.join(imc_folder, filename)
            if_data_path = os.path.join(if_folder, f'frame_{frame_id}.txt')

            # Run transformation_frame.py
            cmd_transformation = [
                'python', 'transformation_frame.py',
                '--frame_id', str(frame_id),
                '--x_center', str(x_center),
                '--y_center', str(y_center),
                '--if_data', if_data_path,
                '--imc_data', imc_data_path,
                '--scaling', str(scaling),
                '--flipping', str(flipping),
                '--mode', 'process',
                '--base_dir', base_dir
            ]
            print(cmd_transformation)
            subprocess.run(cmd_transformation)

            # Run mapPointSets.py
            cmd_mapping = [
                'python', 'mapPointSets.py',
                '--IF', f'{slide_id}/transformed_if/if_data_transformed_{frame_id}.txt',
                '--IMC', f'{slide_id}/transformed_imc/imc_data_transformed_{frame_id}.txt',
                '--frame_id', str(frame_id),
                '--slide_id', slide_id,
                '-o', f'{slide_id}/matched/{slide_id}_matched_frame_{frame_id}.txt',
                '--mode', 'process',
                '--output_img_dir', f'{slide_id}/matched_plots/'
            ]
            print(cmd_mapping)
            subprocess.run(cmd_mapping)

def main():
    parser = argparse.ArgumentParser(
        description="Transform and map IF and IMC data for an entire slide.")
    
    parser.add_argument(
        "--imc_folder", required=True, 
        help="Path to the folder containing IMC data files")
    
    parser.add_argument(
        "--if_folder", required=True, 
        help="Path to the folder containing IF data files")
    
    parser.add_argument(
        "--reference_file", required=True, 
        help="Path to the reference file containing center coordinates")
    
    parser.add_argument(
        "--scaling", type=float, default=0.59, 
        help="Scaling factor for IF data")
    
    parser.add_argument(
        "--flipping", type=bool, default=True, 
        help="Whether to flip IMC data")
    
    parser.add_argument(
        "--slide_id", required=True, 
        help="Slide ID")
    
    parser.add_argument(
        "--base_dir", required=True, 
        help="Base Directory")

    args = parser.parse_args()

    transform_whole_slide(args.imc_folder, args.if_folder, args.reference_file, 
                          args.scaling, args.flipping, args.slide_id, args.base_dir)

if __name__ == "__main__":
    main()
