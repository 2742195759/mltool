import numpy as np
import json
import pandas as pd
import argparse
import xk_mllib as xkml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='None', help='input file of .csv')
    parser.add_argument('--output', type=str, default='None', help='input file of .csv')
    parser.add_argument('--column', type=str, default='None', help='input file of .csv')
    parser.add_argument('--dict', type=str, default='None', help='input file of .csv')
    args = parser.parse_args()
    if args.input == 'None':
        raise Exception('input file is not found')

    value2id = json.loads(args.dict)
    print(json.dumps(xkml.df2json(xkml.to_enumerate(
        args.input, args.output, 'int', 
        args.column, value2id
    ))))
