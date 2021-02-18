import numpy as np
import json
import pandas as pd
import argparse
import xk_mllib as xkml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='None', help='input file of .csv')
    args = parser.parse_args()
    if args.input == 'None':
        raise Exception('input file is not found')

    df = pd.read_csv(args.input, index_col=0)
    print(json.dumps(xkml.df2json(df)))
