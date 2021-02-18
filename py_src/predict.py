import numpy as np
import json
import pandas as pd
import argparse
import xk_mllib as xkml


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='None', help='input file of .csv')
    parser.add_argument('--ylabel', type=str, default='None', help='input file of .csv')
    parser.add_argument('--xlabels', type=str, default='None', help='input file of .csv')
    parser.add_argument('--ml-method', type=str, default='None', help='input file of .csv')
    parser.add_argument('--output', type=str, default='None', help='input file of .csv')
    args = parser.parse_args()
    if args.input == 'None':
        raise Exception('input file is not found')

    x_labels = (json.loads(args.xlabels))
    xkml.start_experiment(args.input, args.output, args.ylabel, x_labels, args.ml_method)
