from srcs.split_dataset import split_dataset
from srcs.training import training
from srcs.predict import predict
import numpy as np
import argparse


def validate_positive_int(value):
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"{value} must be a positive integer greater than 0")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} must be a valid integer")

def validate_layers(value_list):
    if len(value_list) < 2:
        raise argparse.ArgumentTypeError("At least 2 layers must be specified")
    try:
        int_list = [int(x) for x in value_list]
        if any(x <= 0 for x in int_list):
            raise argparse.ArgumentTypeError("All layer sizes must be positive integers")
        return int_list
    except ValueError:
        raise argparse.ArgumentTypeError("All layer values must be valid integers")

def validate_positive_float(value):
    try:
        fvalue = float(value)
        if fvalue <= 0:
            raise argparse.ArgumentTypeError(f"{value} must be a positive number greater than 0")
        return fvalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} must be a valid number")

def parse_arguments():
    desc = """Multilayer Perceptron (MLP) program for classification.
            It can split a dataset, train a neural network model,
            or make predictions."""
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
    parser.add_argument("-e", "--epochs", 
                        type=validate_positive_int,
                        default=84,
                        required=False,
                        help="Number of epochs for training. Must be positive. Default: 84")
    parser.add_argument("-l", "--layer",
                        nargs='+',
                        type=int,
                        default=[24, 24, 24],
                        required=False,
                        help="Number and density of layers in the network. Minimum 2 layers required. Default: [24, 24, 24]")
    parser.add_argument("-b", "--batch_size",
                        type=validate_positive_int,
                        default=8,
                        required=False,
                        help="Batch size for training. Must be positive. Default: 8")
    parser.add_argument("-r", "--learning_rate",
                        type=validate_positive_float,
                        default=0.0314,
                        required=False,
                        help="Learning rate for training. Must be positive. Default: 0.0314")
    parser.add_argument("-o", "--loss",
                        type=str,
                        choices=["binaryCrossentropy",
                                "categoricalCrossentropy",
                                "sparseCategoricalCrossentropy"],
                        default="sparseCategoricalCrossentropy",
                        required=False,
                        help="Loss function to use. Default: 'sparseCategoricalCrossentropy'")
    parser.add_argument("-s", "--seed", 
                        type=validate_positive_int,
                        default=None,
                        required=False,
                        help="Generate a random seed to track results. Default: None")
    parser.add_argument("-w", "--weight_initializer", type=str,
                        required=False,
                        default="HeUniform",
                        choices=["HeNormal", "HeUniform", "GlorotNormal", "GlorotUniform"],
                        help="Choose which weight initialisation method will be used for training. Default: 'HeUniform'")
    parser.add_argument("--standardize", type=str,
                        required=False,
                        default="z_score",
                        choices=["z_score", "minmax"],
                        help="Choose which standardization method will be used for training. Default: 'z_score'")
    parser.add_argument("--solver", type=str,
                        required=False,
                        default="sgd",
                        choices=["sgd", "adam"],
                        help="Choose wich solver will be used for training. Default: 'sgd'")
    parser.add_argument("-p", "--patience", type=validate_positive_int,
                        required=False,
                        default=5,
                        help="Number of epochs to wait before early stopping. Default: 5")
    parser.add_argument("--activation", type=str,
                        required=False,
                        default='sigmoid',
                        choices=["sigmoid", "relu", "leakyrelu", "tanh"],
                        help="Choose the activation function for hidden layers. Default: 'sigmoid'")
    parser.add_argument("--output_activation", type=str,
                        required=False,
                        default='softmax',
                        choices=["sigmoid", "softmax"],
                        help="Choose the activation function for output layer. Default: 'softmax'")
    parser.add_argument("-m", "--model", type=str,
                        required=False,
                        help="Path to saved model file for prediction")

    args = parser.parse_args()

    if args.action == 'split' and not args.dataset:
        parser.error("-d | --dataset <dataset name> is required for 'split' action.")
    elif args.action == "predict":
        if not args.model or not args.dataset:
            parser.error("-d | --dataset <dataset name> AND -m | --model <model name> is required for 'predict' action.")
    try:
        args.layer = validate_layers(args.layer)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    return args


if __name__ == "__main__":
    args = parse_arguments()
    if args.action == "split":
        split_dataset(args.dataset)
    elif args.action == "train":
        training(layer=args.layer, 
                 epochs=args.epochs, 
                 loss=args.loss, 
                 batch_size=args.batch_size, 
                 learning_rate=args.learning_rate, 
                 seed=args.seed,
                 standardize=args.standardize,
                 weight_initializer=args.weight_initializer,
                 solver=args.solver,
                 patience=args.patience,
                 activation=args.activation,
                 output_activation=args.output_activation)
    elif args.action == "predict":
        try:
            scaler_params = np.load('scaler_params.npy', allow_pickle=True).item()
            predictions = predict(args.model, args.dataset, scaler_params)
        except FileNotFoundError:
            print("Error: Model file or scaler parameters not found")
