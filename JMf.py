import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from functools import partial

def reseed(seed=12):
    global rand
    rand = np.random.RandomState(seed)

activations = {
    'linear': ('linear', lambda x: x, lambda x: np.ones(np.array(x).shape)),
    'tanh': ('tanh', lambda x: np.tanh(x), lambda x: 1. - (np.tanh(x) ** 2.)),
    'sigma': ('sigma', lambda x: 1. / (1. + np.exp(-x)), lambda x: activations['sigma'][1](x) * (1. - activations['sigma'][1](x))),
    'eswish': ('eswish', lambda x, beta=1: beta * x * activations['sigma'][1](x), lambda x, beta=1.: activations['eswish'][1](x) + activations['sigma'][1](x) * (beta - activations['eswish'][1](x))),
    'mish': ('mish', lambda x: x * np.tanh(np.log1p(np.exp(x))), lambda x: np.tan(np.log1p(np.exp(x))) + x * (1 / np.cos(np.log1p(np.exp(x)))) ** 2),
    'relu': ('relu', lambda x: np.maximum(np.zeroes(np.array(x).shape), x), lambda x: 1. * (x > .0)),
    # 'ustep': ('ustep', lambda x: 1. * (x > .0), lambda x: np.zeroes(np.array(x).shape)),
    'sin': ('sin', lambda x: np.sin(x * np.pi), lambda x: np.pi * np.cos(np.pi * x)),
    'gaussian': ('gaussian', lambda x: np.exp(-np.multiply(x, x) / 2.), lambda x: -np.exp(-(x * x) / 2.) * x),
    'usigma': ('usigma', lambda x: (np.tanh( x / 2.) + 1.) / 2., lambda x: ((1. / np.cosh(x / 2)) ** 2.) / 4),
    # 'inverse': ('inverse', lambda x: -x, lambda x: -np.ones(np.array(x).shape)),
    'cosine': ('cosine', lambda x: np.cos(x * np.pi), lambda x: -x * np.sin(x * np.pi))
    }

def unique(olist):
    return list(dict.fromkeys(olist))

class Layer():
    def __init__(self, nodes, shape, type=None, activation=None, sparsity=.5, recurrence=.2, dropout=.2, bounds=(-.25, .75), spectral_radius=.95, leakage=.8, noise=.05):
        self.type = type
        self.spectral_radius = spectral_radius
        self.dropout = dropout
        self.leakage = leakage
        self.sparsity = sparsity
        self.recurrence = recurrence
        self.bounds = bounds
        self.noise = noise

        activation = activation or activations.get(rand.choice(list(activations.keys())))
        self.activation_name, self.activation, self.dactivation = activation
        if self.activation_name == 'eswish':
            self.beta = rand.uniform(1, 2)
            self.activation, self.dactivation = partial(self.activation, beta=self.beta), partial(self.dactivation, beta=self.beta)

        self.shape = shape
        row, col = shape
        if type is 'input':
            self.ishape = (row, col)
            self.wshape = (col, row)
            self.oshape = (row, row)
        elif type is 'output':
            self.ishape = (row, row)
            self.wshape = (row, col)
            self.oshape = (row, col)
        else:
            self.ishape = self.wshape = self.oshape = shape
        self.input = np.zeros(self.ishape)
        self.output = np.zeros(self.oshape)

        self.nodes = np.array([Node(self.oshape, activation=activation) for _ in range(nodes)])
        self.connections = self.connect()

        self.pipeline = [c for c in self.connections if c.active]
        self.destination_nodes = unique([c.destination for c in self.pipeline])
        self.lenpipeline = len(self.pipeline)

    def feedforward(self, input):
        if self.type is not 'input':
            if np.multiply(*input.shape) > np.multiply(*self.wshape) and self.type is not 'output':
                pca = PCA(n_components=self.ishape[-1], random_state=12)
                input = pca.transform(pca.fit_transform(input).T)
            else:
                input = np.resize(input, self.ishape)

        self.input = input
        noise = rand.normal(self.input.mean(), self.input.std(), self.oshape) * self.noise

        for c in self.pipeline: c.source.state = self.input

        [c.propagate(hidden=self.type) for c in self.pipeline]

        propagation = np.mean([self.activate(node.state) for node in self.destination_nodes], axis=0) + noise

        propagation = (1 - self.leakage) * self.output + self.leakage * propagation
        self.output = propagation

        return self.output

    def activate(self, w):
        return self.activation(w)

    def dactivate(self, w):
        return self.dactivation(w)

    def connect(self, sources=None, destinations=None):
        c = np.array([])
        if self.type is 'input' or self.type is 'output':
            c = [Connection(self.wshape, Node(self.oshape), n, self.bounds, self.spectral_radius, self.dropout) for n in self.nodes]
        else:
            for s in sources or self.nodes:
                for d in destinations or self.nodes:
                    shape = self.wshape if ((d is not s) and (self.sparsity < rand.rand())) or ((d is s) and (self.recurrence > rand.rand())) else np.zeros(self.wshape)
                    c = np.append(c, Connection(shape, s, d, self.bounds, self.spectral_radius, self.dropout))
        return c


class Connection():
    def __init__(self, weight=None, source=None, destination=None, bounds=(-.25, .75), spectral_radius=.95, dropout=.2):
        wmin, wmax = bounds
        wmean = (wmin + wmax) / 2
        wstd = (wmax - wmin) / 2
        self.triggered = False
        self.active = False
        self.weight = weight if type(weight) is np.ndarray else rand.normal(wmean, wstd, weight) if type(weight) is tuple else np.array([rand.normal(wmean, wstd)])
        if self.weight.any():
            row, col = self.weight.shape
            self.active = True
            if not (row - col):
                mask = rand.random((row, col)) > dropout
                self.weight[~mask] = .0
                radius = max(abs(np.linalg.eigvals(self.weight)))
                if not radius: print(self.weight, mask)
                self.weight = self.weight * (spectral_radius / radius)
        self.source = source
        self.destination = destination

    def propagate(self, hidden=False):#, noise=.001):
        self.triggered = True
        i = np.dot(self.source.state, self.weight)
        return self.destination.activate(i)



class Node():
    def __init__(self, shape, activation=None):#, bias=0):
        self.active = True
        self.state = np.zeros(shape)
        self.activation_name, self.activation, self.dactivation = activation or activations.get(rand.choice(list(activations.keys())))

    def activate(self, w):
        w = np.array(w)
        self.state = w
        self.state = np.clip(np.nan_to_num(self.state), -5, 5)
        return self.state


class Population():
    def __init__(self, inputs, outputs, population=None, input_step=1, sparsity=.5, recurrence=.2, spectral_radius=.95, dropout=.2, leakage=.8, alpha=1e-3, noise=.05, seed=12):
        reseed(seed)

        self.alpha = alpha
        oshape = outputs.shape[1:]
        self.inputs = inputs
        ishape = inputs.shape[1:]
        self.outputs = outputs

        input, *hidden, output = population

        itopologie = [input, (input_step, ishape[-1])]
        htopologie = [[nodes, (shape, shape)] for nodes, shape in hidden]
        otopologie = [output, (input_step, oshape[-1])]
        self.initial_topologie = [itopologie, *htopologie, otopologie]

        self.current_step = 0


        self.ilayer = Layer(*itopologie, type='input', activation=activations.get('linear'), sparsity=1., recurrence=.0, bounds=(-.1, .1), spectral_radius=spectral_radius, dropout=.0, leakage=1., noise=noise)

        self.layers = [Layer(nodes, shape, type='hidden', activation=activations.get('tanh'), \
            sparsity=sparsity, recurrence=recurrence, bounds=(-.1, .1), spectral_radius=spectral_radius, dropout=dropout, leakage=leakage, noise=noise) for nodes, shape in htopologie]

        self.olayer = Layer(*otopologie, type='output', activation=activations.get('eswish'), sparsity=1., recurrence=.0, bounds=(-.1, .1), spectral_radius=spectral_radius, dropout=.0, leakage=1., noise=noise)

    # def remove_node(self, node):
    #     node.active = False
    #     for _ in self.connections[[_ for _ in self.connections if node in [_.source, _.destination]]]:
    #         self.remove_connection(_)
    #
    # def create_connection(self, source, destination, activation):
    #     self.connections = self.connections.append(Connection(self.update_step(), source, destination))
    #
    # def remove_connection(self, connection):
    #     connection.active = False
    #     if(connection.source not in [_.source for _ in self.connections]):
    #         self.remove_node(connection.source)

    # def update_step(self):
    #     _ = self.current_step
    #     self.current_step += 1
    #     return _

    def feedforward(self, x):
        input = x
        input = self.ilayer.feedforward(input)
        for l in self.layers:
            input = l.feedforward(input)
        output = self.olayer.feedforward(input)
        return output

    def backpropagation(self, y):
        ...

    @staticmethod
    def loss(y, z):
        return sum((z - y)**2)


class SimpleScaler():
    def __init__(self):
        self.mean = 0
        self.std = 1

    @staticmethod
    def transform(array):
        mean = array.mean()
        std = array.std()

        w = (array - mean) / std
        return w

    def fit_transform(self, array):
        self.mean = array.mean()
        self.std = array.std()

        w = (array - self.mean) / self.std
        return w

    def fit(self, array):
        self.mean = array.mean()
        self.std = array.std()

    def reverse(self, array):
        w = (array * self.std) + self.mean

        return w






















'''___'''
