#!/bin/python3

from split_dataset import split_dataset
from training import training
import argparse


def parse_arguments():
    desc = "Multilayer Perceptron (MLP) program for classification. \
            It can split a dataset, train a neural network model, \
            or make predictions."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-a", "--action",
                        choices=['split', 'train', 'predict'],
                        required=True,
                        help="Choose the action to perform: 'split' \
                        to split the data, 'train' to train the model, \
                        or 'predict' to make a prediction.")
    parser.add_argument("-d", "--dataset", type=str,
                        required=False,
                        help="Path to the CSV file containing the data.")
    parser.add_argument("-e", "--epochs", type=int, default=84,
                        required=False,
                        help="Number of epochs for training. Default: 84")
    parser.add_argument("-l", "--layer", nargs='+', type=int, default=[24, 24],
                        required=False,
                        help="Number and density of layers in the network. Default: [24, 24]")
    parser.add_argument("-b", "--batch_size", type=int, default=8,
                        required=False,
                        help="Batch size for training. Default: 8")
    parser.add_argument("-r", "--learning_rate", type=float, default=0.0314,
                        required=False,
                        help="Learning rate for training. Default: 0.0314")
    parser.add_argument("-o", "--loss", type=str,
                        choices=["binaryCrossentropy",
                                 "categoricalCrossentropy",
                                 "sparseCategoricalCrossentropy"],
                        default="sparseCategoricalCrossentropy",
                        required=False,
                        help="Loss function to use. Default: 'sparseCategoricalCrossentropy'")

    args = parser.parse_args()

    if args.action in ['split'] and not args.dataset:
        parser.error("-d | --dataset <dataset name> is required for 'split' action.")

    return args


if __name__ == "__main__":
    args = parse_arguments()
    if args.action == "split":
        split_dataset(args.dataset)
    elif args.action == "train":
        training(args.layer, args.epochs, args.loss, args.batch_size, args.learning_rate)
