import numpy as np
import glob
import os
import argparse

this_dir = 'outputs/results/'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='name of nn model')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    file_results = sorted(glob.glob(os.path.join(this_dir, args.model, '*.txt')))
    for file in file_results:
        result = np.loadtxt(file)
        print('file:\t', file)
        print('acc-avg-subject:\t', np.mean(result, 1))
        print('acc-avg-total:\t', np.mean(np.mean(result, 1)))

if __name__ == "__main__":
    main()