import argparse
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(frame_id, if_data_path, imc_data_path, x_center, y_center, 
              scaling, flipping, x_center_imc, y_center_imc, dir, visual):
    df_if = pd.read_table(if_data_path)
    df_imc = pd.read_table(imc_data_path, sep=',')
    df_imc.rename(columns={'CellID': 'cell_id'}, inplace=True)
    df_imc.rename(columns={'X_centroid': 'x'}, inplace=True)
    df_imc.rename(columns={'Y_centroid': 'y'}, inplace=True)

    # Store original coordinates in new columns
    df_if['original_x'] = df_if['x']
    df_if['original_y'] = df_if['y']

    if_origin = [x_center, y_center]
    imc_origin = [x_center_imc, y_center_imc]
    # center the origin and IF scaling
    df_if.loc[:, ['x', 'y']] = scaling*(df_if.loc[:, ['x', 'y']] - if_origin) 
    df_imc.loc[:, ['x', 'y']] = df_imc.loc[:, ['x', 'y']] - imc_origin

    # Flipping
    if flipping:
        df_imc[['x', 'y']] = df_imc[['y', 'x']]

    # Cropping IF image to match with the dimension of imc 
    df_if = df_if[(df_if['x'] >= -x_center_imc) & (df_if['y'] >= -y_center_imc) 
                  & (df_if['x'] <= x_center_imc) & (df_if['y'] <= y_center_imc)]
    
    # Save transformed data to txt files
    base_dir = dir

    # Subdirectories for transformed data
    if_dir = os.path.join(base_dir, 'transformed_if')
    imc_dir = os.path.join(base_dir, 'transformed_imc')

    # Create directories if they don't exist
    os.makedirs(if_dir, exist_ok=True)
    os.makedirs(imc_dir, exist_ok=True)

    # File paths for the transformed data
    if_output_path = os.path.join(if_dir, f'if_data_transformed_{frame_id}.txt')
    imc_output_path = os.path.join(imc_dir, f'imc_data_transformed_{frame_id}.txt')

    # Save transformed data to txt files
    df_if.to_csv(if_output_path, sep='\t', index=False)
    df_imc.to_csv(imc_output_path, sep='\t', index=False)

    # Visualization in test mode
    if visual == True:
        df_if['mode'] = 'IF'
        df_imc['mode'] = 'IMC'
        df = pd.concat([df_if, df_imc])
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df, x='x', y='y', hue='mode')
        plt.title(f"Transformed Data Visualization for Frame ID {frame_id}")
        plt.show()

    return df_if, df_imc



def main():
    parser = argparse.ArgumentParser(
        description="Transform and visualize IF and IMC data by frame.")
    
    parser.add_argument(
        "--frame_id", required=True, type=int, 
        help="Frame ID")
    
    parser.add_argument(
        "--x_center", required=True, type=int, 
        help="x parameter of the center coordinate of the IF data")

    parser.add_argument(
        "--y_center", required=True, type=int, 
        help="y parameter of the center coordinateof the IF data")

    parser.add_argument(
        "--base_dir", required=True, type=str, 
        help="base directory folder name (slide ID)")
    
    parser.add_argument(
        "--if_data", type=str, required=True, 
        help="File path to IF data")
    
    parser.add_argument(
        "--imc_data", type=str, required=True, 
        help="Path to IMC data")

    parser.add_argument(
        "--scaling", type=float, default=1.0, 
        help="Scaling factor")
    
    parser.add_argument(
        "--flipping", type=bool, default=False,
        help="Flip IMC data")
    
    parser.add_argument(
        "--x_center_imc", type=int, default=200,
        help="x parameter of the center coordinate of the IMC data")
    
    parser.add_argument(
        "--y_center_imc", type=int, default=200,
        help="x parameter of the center coordinate of the IMC data")
    
    parser.add_argument(
        "--mode", type=str, default='test',
        help="test mode or process mode. test mode will plot the graph")
    
    args = parser.parse_args()

    visual = False
    if args.mode == 'test':
        visual = True

    load_data(args.frame_id, args.if_data, args.imc_data, args.x_center, 
              args.y_center, args.scaling, args.flipping, args.x_center_imc,
              args.y_center_imc, args.base_dir, visual)

if __name__ == "__main__":
    main()
