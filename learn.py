#!/usr/bin/env python

import argparse

import loader

from pybrain.datasets import SupervisedDataSet
from rnn import RecurrentNeuralNetwork

out = ['T3 P', 'T3 L', 'T4 P', 'T4 L']


def learn(input, output):
    """
    Learn nn from data.
    """
    nn = RecurrentNeuralNetwork(14, 4)
    dataset = SupervisedDataSet(14, 4)
    for ins, out in zip(input, output):
        dataset.addSample(ins, out)
        
    nn.set_learning_data(dataset)
    nn.train(1)


def run(path):
    data = loader.from_csv(path)

    ins = zip(*[data[x] for x in data if x not in out])
    outs = zip(*[data[x] for x in data if x in out])

    learn(ins, outs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learn some neural network')
    parser.add_argument("data", type=str,
                        help="Path to CSV file containing data to learn on")
    args = parser.parse_args()
    run(args.data)
