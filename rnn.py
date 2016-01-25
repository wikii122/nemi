from pybrain.structure import RecurrentNetwork, BiasUnit
from pybrain.structure.modules import LinearLayer, SigmoidLayer, TanhLayer, LSTMLayer
from pybrain.structure.connections import FullConnection
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from  pybrain.tools.validation import Validator
from pybrain.tools.shortcuts import buildNetwork
from singleton import singleton
class RecurrentNeuralNetwork:
    """
    Recurent neural network.
    """
    def __init__(self, nin, nout):
        singleton.append(self)

        self.inn = nin
        self.outn = nout

        self.n = buildNetwork(nin, 20, nout, bias=False, recurrent=True)
        self.n = RecurrentNetwork()
        self.n.addInputModule(LinearLayer(nin, name='in'))
        self.n.addOutputModule(LinearLayer(nout, name='out'))

        self.n.addModule(SigmoidLayer(8, name='hidden2'))
        self.n.addModule(TanhLayer(nin+nout/2, name='hidden1'))
        self.n.addModule(BiasUnit(name='bias'))
        self.n.addModule(LSTMLayer(5, name='memory'))

        self.n.addConnection(FullConnection(self.n['in'], self.n['hidden1']))
        self.n.addConnection(FullConnection(self.n['bias'], self.n['hidden1']))
        self.n.addConnection(FullConnection(self.n['hidden1'], self.n['hidden2']))
        self.n.addConnection(FullConnection(self.n['hidden2'], self.n['out']))
        self.n.addConnection(FullConnection(self.n['hidden1'], self.n['memory']))
        self.n.addConnection(FullConnection(self.n['memory'], self.n['hidden2']))
        self.n.addConnection(FullConnection(self.n['in'], self.n['hidden2']))
        self.n.addConnection(FullConnection(self.n['hidden2'], self.n['out']))
        
        self.n.addRecurrentConnection(FullConnection(self.n['hidden1'], self.n['hidden1']))
        self.n.addRecurrentConnection(FullConnection(self.n['memory'], self.n['hidden1']))
        self.n.sortModules()

    def set_learning_data(self, dataset):
        """
        Set dataset used to train network.
        """
        self.ds_learn = dataset

    def train(self, epochs=100):
        """
        Train the network
        """
        #self.n.reset()
        trainer = BackpropTrainer(self.n, self.ds_learn, verbose=True)
        # trainer.setData(self.ds_learn)
        return trainer.trainEpochs(epochs=epochs)

    def validate_error(self, dataset):
        """
        Return error value for given dataset
        """
        v = Validator()
        #self.n.reset()
        return v.MSE(self.n, dataset)

    def calculate(self, dataset):
        """
        Return network response for given dataset
        """
        #self.n.reset()
        return self.n.activateOnDataset(dataset)

