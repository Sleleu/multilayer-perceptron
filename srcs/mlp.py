#!/bin/python3

from split_dataset import split_dataset
import argparse


def parse_arguments():
    desc = "Multilayer Perceptron (MLP) program for classification. \
            It can split a dataset, train a neural network model, \
            or make predictions."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-a", "--action",
                        choices=['split', 'train', 'predict'],
                        required=True,
                        help="choose the action to perform: 'split' \
                        to split the data, 'train' to train the model, \
                        or 'predict' to make a prediction.")
    parser.add_argument("-d", "--dataset", type=str,
                        required=False,
                        help="path to the CSV file containing the data.")

    args = parser.parse_args()

    if args.action in ['split'] and not args.dataset:
        parser.error("-d | --dataset is required for 'split' actions.")

    return args


if __name__ == "__main__":
    args = parse_arguments()
    if args.action == 'split':
        if args.dataset:
            split_dataset(args.dataset)
