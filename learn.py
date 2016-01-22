#!/usr/bin/env python3

import argparse

import loader
import matplotlib.pyplot as plot

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

    learning, validating = dataset.splitWithProportion(0.8)
    nn.set_learning_data(learning)
    nn.train(1)

    result = nn.calculate(validating)

    return result, validating['target']


def run(path):
    data = loader.from_csv(path)

    ins = zip(*[data[x] for x in data if x not in out])
    outs = zip(*[data[x] for x in data if x in out])

    y_mod, y = learn(ins, outs)
    e = sum(sum(x) for x in y-y_mod)
    er = sum(sum(x**2) for x in y-y_mod)

    print("Błąd sumaryczny {e}".format(e=e))
    print("Suma średniokwadratowy {e}".format(e=er))

    plot.figure(1)
    plot.plot(y_mod)
    plot.plot(y)

    plot.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learn some neural network')
    parser.add_argument("data", type=str,
                        help="Path to CSV file containing data to learn on")
    args = parser.parse_args()
    run(args.data)
