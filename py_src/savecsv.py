import numpy as np
import json
import pandas as pd
import argparse
import xk_mllib as xkml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='None', help='input file of .csv')
    parser.add_argument('--xlabels', type=str, default='None', help='input file of .csv')
    parser.add_argument('--output', type=str, default='None', help='input file of .csv')
    args = parser.parse_args()
    if args.input == 'None':
        raise Exception('input file is not found')

    x_labels = (json.loads(args.xlabels))
    data = xkml.save_csv(args.input, args.output, x_labels)
    print(json.dumps(xkml.df2json(data)))
