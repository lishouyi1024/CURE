#!/home/amin/miniconda3/bin/python
# /Users/lsy/gurobi.lic

from gurobipy import *
import numpy as np
import pandas as pd
import pickle
import sys
import argparse
import matplotlib.pyplot as plt
import os

class Matching:
    def __init__(self, w, sigma2, IF, IMC, dist, r):
        self.w = w
        self.IF = IF
        self.IMC = IMC
        self.sigma2 = sigma2
        self.dist = dist
        self.r = r
        self.M = IMC.shape[0]
        self.N = IF.shape[0]
        self.x = [[0 for j in range(self.N)] for i in range(self.M)]

        self.model = Model("IMC-IF-Assignment-Problem")

    def generate_p(self):
        P = np.sum((self.IF[None, :, :] - self.IMC[:, None, :]) ** 2, axis=2)

        c = (2 * np.pi * self.sigma2) ** (2 / 2)
        c = c * self.w / (1 - self.w)
        c = c * self.M / self.N

        P = np.exp(-P / (2 * self.sigma2))
        den = np.sum(P, axis=0)
        den = np.tile(den, (self.M, 1))
        den[den == 0] = np.finfo(float).eps
        den += c
        
        self.P = np.divide(P, den)

    def formulate(self):
        
        # 1) Define variables
        for i in range(self.M):
            for j in range(self.N):
                self.x[i][j] = self.model.addVar(vtype=GRB.BINARY,
                                                 name=f"x_{i}_{j}")

        # 2) Define constraints
        # -- Unique mapping
        for i in range(self.M):
            self.model.addConstr(quicksum(self.x[i]), GRB.LESS_EQUAL, 1,
                                 name=f"unique_matching_M_{i}")
            
        for j in range(self.N):
            self.model.addConstr(
                quicksum([self.x[i][j] for i in range(self.M)]),
                GRB.LESS_EQUAL, 1, name=f"unique_matching_N_{j}"
            )
        
        # -- Distance constraint
        for i in range(self.M):
            for j in range(self.N):
                if self.dist[i, j] > self.r:
                    self.model.addConstr(self.x[i][j], GRB.EQUAL, 0,
                                         name=f"distance_{i}_{j}")

        # 3) Define objective
        self.model.setObjective(
            quicksum([self.x[i][j] * self.P[i, j]
                      for i in range(self.M) for j in range(self.N)]),
            GRB.MAXIMIZE
        )

    def match(self):
        self.generate_p()
        self.formulate()
        self.model.update()
        try:
            self.model.optimize()
            if self.model.Status == GRB.OPTIMAL:
                print("Optimization was successful.")
            else:
                print("Optimization was unsuccessful.")
                return False
        except GurobiError as e:
            print("Optimize failed due to:", e.message)
            return False
        return True

    def getResults(self):
        if not self.match():  # Ensure optimization was successful
            return []
        matchings = []
        for i in range(self.M):
            for j in range(self.N):
                if self.x[i][j].X == 1:
                    matchings.append([i, j, self.dist[i, j]])
                   # print(self.x[i][j].VarName)
        
        matchings = pd.DataFrame(matchings, columns=['imc_i', 'if_i', 'dist'])
        IF_coords = pd.DataFrame(self.IF, columns=['if_x', 'if_y']).iloc[matchings['if_i'], :]
        IMC_coords = pd.DataFrame(self.IMC, columns=['imc_x', 'imc_y']).iloc[matchings['imc_i'], :]
        matchings.reset_index(inplace=True, drop=True)
        IF_coords.reset_index(inplace=True, drop=True)
        IMC_coords.reset_index(inplace=True, drop=True)
        result = pd.concat([matchings, IF_coords, IMC_coords], axis=1)
        return(result)


def loadData(path):
    """Load any data in .pickle format."""
    f = open(path, "rb")
    data = pickle.load(f)
    print(f"Data loaded from {path}!")
    return(data)


def pointsetDistance(IF, IMC):
    IMC = np.repeat(IMC[:, np.newaxis, :], len(IF), axis=1)
    IF = np.repeat(IF[np.newaxis, :, :], len(IMC), axis=0)
    dist = IMC - IF
    dist = np.sqrt(np.sum(dist ** 2, axis=2))
    return(dist)


# def plot_matched_pairs(matched_df, slide_id, frame_id, output_dir):
#     plt.figure(figsize=(12, 8))
#     plt.scatter(df_if['if_x'], df_if['if_y'], 
#                 color='blue', label='IF Points')
#     plt.scatter(df_imc['imc_x'], df_imc['imc_y'], 
#                 color='red', label='IMC Points')

#     # Set the background color
#     ax = plt.gca()
#     ax.set_facecolor('white')

#     # Draw lines connecting matched pairs
#     for index, row in matched_df.iterrows():
#         plt.plot([df_if.loc[index, 'if_x'], 
#                   df_imc.loc[index, 'imc_x']],
#                  [df_if.loc[index, 'if_y'], 
#                   df_imc.loc[index, 'imc_y']],
#                  color='black', alpha=1)
#         # label the imc cell id for each pair
#         # imc_x, imc_y = df_imc.loc[row['imc_i'], ['imc_x', 'imc_y']]
#         # plt.text(df_imc.imc_x, df_imc.imc_y, f'{round(df_imc.imc_i)}', fontsize=8, color='black')

#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.title(f'Visualization of Matched Pairs for Slide {slide_id} Frame {frame_id}')
#     plt.legend()
#     qc_dir = output_dir
#     os.makedirs(qc_dir, exist_ok=True) 
#     plot_path = os.path.join(qc_dir, f'matched_pairs_{slide_id}_frame_{frame_id}.png')
#     plt.savefig(plot_path)
#     plt.close()  
#     print(f"Plot saved to {plot_path}")
#     if args.mode == 'test':
#         plt.show()


def plot_matched_pairs(df_if, df_imc, matched_df, slide_id, frame_id, output_dir):
    plt.figure(figsize=(12, 8))
    plt.scatter(df_if['if_x'], df_if['if_y'], color='blue', label='IF Points')
    plt.scatter(df_imc['imc_x'], df_imc['imc_y'], color='red', label='IMC Points')

    # Draw lines connecting matched pairs
    for _, row in matched_df.iterrows():
        plt.plot([row['if_x'], row['imc_x']], [row['if_y'], row['imc_y']], 
                 color='black', alpha=0.5)
        # label the imc cell id for each pair
        # imc_x, imc_y = df_imc.loc[row['imc_i'], ['imc_x', 'imc_y']]
        # plt.text(df_imc.imc_x, df_imc.imc_y, f'{round(df_imc.imc_i)}', fontsize=8, color='black')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Visualization of Matched Pairs for Slide {slide_id} Frame {frame_id}')
    plt.legend(loc='best')

    # Prepare the output directory and save the plot
    qc_dir = output_dir
    os.makedirs(qc_dir, exist_ok=True) 
    plot_path = os.path.join(qc_dir, f'matched_pairs_{slide_id}_frame_{frame_id}.png')
    plt.savefig(plot_path)
    plt.close()  
    print(f"Plot saved to {plot_path}")
    if args.mode == 'test':
        plt.show()



def main(args):
    df_if = pd.read_table(args.IF, sep='\t')
    df_imc = pd.read_table(args.IMC, sep='\t')
    frame_id = args.frame_id
    slide_id = args.slide_id
    IF = df_if.loc[:, ['x', 'y']].to_numpy()
    IMC = df_imc.loc[:, ['x', 'y']].to_numpy()
    dist = pointsetDistance(IF=IF, IMC=IMC)

    radius = args.r
    w = args.w
    sigma2 = args.s

    print("Matching started...")
    matching = Matching(w=w, sigma2=sigma2, IF=IF, IMC=IMC, dist=dist, r=radius)
    match_succeed = matching.match()
    if match_succeed:
        print("Matching succeeded!")
    else:
        print("Matching failed.")
        return
    matched_df = matching.getResults()

    # print('matched')
    # print(matched_df)

    # Convert results to DataFrame and merge with original data
    # matched_df = pd.DataFrame(result, columns=['imc_cell_id', 'if_cell_id', 
    #                                            'distance'])
    
    # Prepend prefix in original datasets
    df_if = df_if.add_prefix('if_')
    df_imc = df_imc.add_prefix('imc_')

    # Merge datasets based on matched IDs
    merged_df = pd.merge(matched_df, df_imc, on=['imc_x', 'imc_y'], how='left')
    merged_df = pd.merge(merged_df, df_if, on=['if_x', 'if_y'], how='left')

    # Add frame id to the output file
    merged_df['frame_id'] = frame_id

    # Rename the original_x and original_y to x and y to fit the extract_event.py 
    merged_df.rename(columns={'if_original_x': 'x'}, inplace=True)
    merged_df.rename(columns={'if_original_y': 'y'}, inplace=True)

    # Output to text file
    output_file = args.output
    merged_df.to_csv(output_file, sep="\t", index=False)
    print(f"Output saved to {output_file}")

    # print(merged_df)

    # Visualization of matched pairs
    output_img_dir = args.output_img_dir
    # print(matched_df.head())
    plot_matched_pairs(df_if, df_imc, matched_df, slide_id, frame_id, output_img_dir)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='finds the transformation between 2 point sets',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '--IF', type=str, required=True,
        help="""path to tab-delimited (x,y) coordinate file of target (fixed) 
                (IF) points""")

    parser.add_argument(
        '--IMC', type=str, required=True,
        help="""path to tab-delimited (x,y) coordinate file of source (moving) 
                (IMC) points""")

    parser.add_argument(
        '--frame_id', type=str, required=True,
        help="the frame ID of both source and target inputs")
    
    parser.add_argument(
        '--slide_id', type=str, required=True,
        help="the frame ID of both source and target inputs")

    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help="path to tab-delimited transformed (x,y) coordinates of source")

    parser.add_argument(
        '-r', type=int, default=15,
        help="""Radius threshold for distance calculation.""")

    parser.add_argument(
        '-s', type=float, default=25,
        help="""Variance of each pdf in GMM.""")

    parser.add_argument(
        '-w', type=float, default=0,
        help="""Contribution of the uniform distribution to account for 
                outliers. Valid values span 0 (inclusive) and 1 (exclusive).""")
    
    parser.add_argument(
        "--mode", type=str, default='test',
        help="test mode or process mode. test mode will plot the graph")
    
    parser.add_argument(
        "--output_img_dir", type=str,
        help="directory to store the plots of matched pairs for each slide")

    args = parser.parse_args()
    main(args)
