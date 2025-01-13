import pandas as pd
from tqdm import tqdm
from sys.path import exists

from lightcurve_data import *

def main():
    # Create dataframe for flare
    flaredf = pd.read_csv('flare_data.csv')
    flaredf = flaredf[flaredf['Number of fitted flare profiles'] >= 1.0]

    # Create a place to store flares
    flare_dir = 'flares.csv'

    # Iterate through each dataframe
    for _, row in tqdm(flaredf.iterrows(), 'Processing flares', len(flaredf)):
        # Pull name
        flare_name = f'TIC{row['TIC']}'
        flare_time = row['Flare peak time (BJD)']

            


        # Check if current tic and flare is already in the csv


if __name__ == '__main__':
    main()