#!/usr/bin/env python3

import argparse

import loader
import matplotlib.pyplot as plot

from pybrain.datasets import SupervisedDataSet, SequentialDataSet
from rnn import RecurrentNeuralNetwork
from smoother import gaussian_filter

out = ['T3 P', 'T3 L', 'T4 P', 'T4 L']

def learn(input, output):
    """
    Learn nn from data.
    """
    nn = RecurrentNeuralNetwork(13, 4)
    dataset = SupervisedDataSet(13, 4)
    for ins, out in zip(input, output):
        dataset.addSample(ins, out)

    learning, validating = dataset.splitWithProportion(0.8)
    nn.set_learning_data(learning)
    nn.train(75)

    result = nn.calculate(validating)

    return result, validating['target']


def run(path):
    data = loader.from_csv(path)
    norm = loader.normalize(data)
    data = norm['normalized']
    if type(data) is not dict:
       raise TypeError('Normalized data is not a dict')
    average = norm['average']
    if type(average) is not dict:
       raise TypeError('Normalized average is not a dict')
    variance = norm['variance']
    if type(variance) is not dict:
       raise TypeError('Normalized variance is not a dict')
    del data["Podciśnienie w komorze spalania"]
    for x in ["Ciśnienie pary wtórnej", "Położenie zaworu A prawa", "Przepływ wody wtryskowej do pary wtórnej",
              "T4 P", "T2 L", "T1 P", "T3 L", "Położenie zaworu B prawa", "T3 P", "T4 L", "T2 P", "Średnie obroty podajników paliwa",
              "Ciśnienie wody wtryskowej do wtrysków wtórnych", "Położenie zaworu A lewa", "T1 L"]:
              data[x] = gaussian_filter(data[x])

    ins = zip(*[data[x] for x in data if x not in out])
    outs = zip(*[data[x] for x in data if x in out])

    y_mod, y = learn(ins, outs)
    e = sum(sum(x) for x in y-y_mod)
    er = sum(sum(x**2) for x in y-y_mod)

    print("Błąd sumaryczny {e}".format(e=e))
    print("Suma średniokwadratowy {e}".format(e=er))
    i = 0
    res = {}
    mod = {}
    for key in data:
        if key in out:
            res[key] = y_mod.T[i]
            mod[key] = y.T[i]
            i += 1

    res = loader.denormalize(res, average, variance)
    mod = loader.denormalize(mod, average, variance)

    plot.figure(1)
    for x in res:
        plot.plot(res[x])
        plot.plot(mod[x])

    # plot.plot(y_mod)
    # plot.plot(y)
    plot.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learn some neural network')
    parser.add_argument("data", type=str,
                        help="Path to CSV file containing data to learn on")
    args = parser.parse_args()
    run(args.data)
