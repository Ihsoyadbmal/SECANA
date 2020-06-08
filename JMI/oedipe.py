import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale, MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam, Adagrad, SGD, Adadelta
from tensorflow.keras.utils import Sequence
import math
import random
from datetime import datetime
from tensorflow.keras.utils import plot_model


temps = pd.read_csv('D:/Documents/workspace/DIA/Projets/temperatures/data/daily-minimum-temperatures-in-me.csv', parse_dates=['Date'], index_col='Date')

# temps.info()

# _, ax = plt.subplots(figsize=(100,10))
# sns.lineplot(data=temps, ax=ax)

# temps.isna().any().sum()

temps = temps.asfreq('1D', method='ffill')

_X, _x, _x_ = temps['1981 01':'1986 02'], temps['1987':'1988'], temps['1989':'1990']
spe = temps['1981':'1986'].shape[0]

ssc = MinMaxScaler()
ssc.fit(temps.values)

X, x, x_ = ssc.transform(_X), ssc.transform(_x), ssc.transform(_x_)

class gen(Sequence):
    def __init__(self, x, y, size=90):
        self.x = x
        self.y = y
        self.size = size
        self.len = len(x)
    def __len__(self):
        return self.len - self.size
    def __getitem__(self, idx):
        X = self.x[idx: self.size + idx]
        Y = self.y[self.size + idx:self.size + idx + 1]
        return np.array(X).reshape(1, -1, 1), [np.array(Y), np.array(Y), np.array(Y)]

class pred(Sequence):
    def __init__(self, x, size=90):
        self.x = x
        self.size = size
        self.len = len(x)
    def __len__(self):
        return self.len - self.size
    def __getitem__(self, idx):
        X = self.x[idx: self.size + idx]
        # X = X.reshape(-1, (self.size * 2) + 1, X.shape[1])
        # Y = Y.reshape(-1, Y.shape[0])
        return np.array(X).reshape(1, -1, 1)


tra = gen(X, X)
val = gen(x, x)

input = tf.keras.layers.Input(shape=(tra.size, 1), name='input')

# gru = tf.keras.layers.SimpleRNN(16, name='g1')(input)

dense = tf.keras.layers.Dense(50, activation=tf.nn.relu, name='12')(input)
dp = tf.keras.layers.Dropout(.1, name='dp1')(dense)
dense2 = tf.keras.layers.Dense(25, activation=tf.nn.relu, name='13')(dp)
dp2 = tf.keras.layers.Dropout(.2, name='dp2')(dense2)
output = tf.keras.layers.Dense(1, activation=tf.keras.activations.linear, name='o1')(dp2)

# #####
# gru2 = tf.keras.layers.SimpleRNN(16, name='g2')(input)

dense3 = tf.keras.layers.Dense(50, activation=tf.nn.relu, name='22')(input)
dp3 = tf.keras.layers.Dropout(.2, name='dp3')(dense3)
dense4 = tf.keras.layers.Dense(25, activation=tf.nn.relu, name='23')(dp3)
dp4 = tf.keras.layers.Dropout(.3, name='dp4')(dense4)
output2 = tf.keras.layers.Dense(1, activation=tf.keras.activations.linear, name='o2')(dp4)

# gru3 = tf.keras.layers.SimpleRNN(16, name='g3')(input)

dense5 = tf.keras.layers.Dense(50, activation=tf.nn.relu, name='32')(input)
dp5 = tf.keras.layers.Dropout(.3, name='dp5')(dense5)
dense6 = tf.keras.layers.Dense(25, activation=tf.nn.relu, name='33')(dp5)
dp6 = tf.keras.layers.Dropout(.4, name='dp6')(dense6)
output3 = tf.keras.layers.Dense(1, activation=tf.keras.activations.linear, name='o3')(dp6)

model = tf.keras.models.Model(inputs=input, outputs=[output, output2, output3], name='Jean-Michel')

dnow = datetime.now().strftime("%Y%m%d-%H%M%S")
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
# mc = ModelCheckpoint('best_model.h5', monitor='val_dense_2_mean_squared_error', mode='min', verbose=1, save_best_only=True)

res = '_JM_'
fpath = './checkpoints/{val_loss:.6f}'+ res + dnow + '.h5'

mc = MC(fpath, monitor=['val_o1_loss', 'val_o2_loss', 'val_o3_loss'], mode='min', verbose=1, save_best_only=True)
ga = GA(nets=['12', '13', '22', '23', '32', '33'], monitor=['val_o1_loss', 'val_o2_loss', 'val_o3_loss'], crossover=.75, mutation=.1, dropout=.0)

model.compile(optimizer=Adam(0.01), loss='mse')

model.summary()

plot_model(model, to_file='oedipe.png')

ohist = model.fit_generator(tra, validation_data=val, epochs=10, callbacks=[mc, ga])

sns.lineplot(data=np.array(ohist.history['val_loss']))

plt.subplots(figsize=(20,10))
sns.lineplot(data=np.array(ohist.history['val_o1_loss']), label='output 1')
sns.lineplot(data=np.array(ohist.history['val_o2_loss']), label='output 2')
sns.lineplot(data=np.array(ohist.history['val_o3_loss']), label='output 3')
plt.title('Validation loss - Œdipe')


tes = pred(x_)
model = tf.keras.models.load_model('./checkpoints/0.058382_JM_20200602-122740.h5')

z = model.predict(tes)

zp = np.r_[z[0][0, :, :], z[0][1:, -1, :]]

_, ax = plt.subplots(figsize=(0o24,0b1010))
# sns.lineplot(data=ssc.inverse_transform(z[0]), ax=ax, palette='Reds')
sns.lineplot(data=ssc.inverse_transform(zp)[-500:], ax=ax, palette='Greens')
# sns.lineplot(data=ssc.inverse_transform(z[2]), ax=ax, palette='Purples')
sns.lineplot(data=ssc.inverse_transform(x_[:, -1:])[-500:], ax=ax)


_, ax = plt.subplots(figsize=(0o24,0b1010))
# sns.lineplot(data=ssc.inverse_transform(z[0]), ax=ax, palette='Reds')
sns.lineplot(data=zp[-500:], ax=ax, palette='Greens')
# sns.lineplot(data=ssc.inverse_transform(z[2]), ax=ax, palette='Purples')
sns.lineplot(data=x_[:, -1:][-500:], ax=ax)




# Tests
# class GA(Callback):
#     def __init__(self, mode='auto', monitor=['val_loss'], nets=[], cm=(0x32, 0b1010)):
#         super(GA, self).__init__()
#         self.monitor = monitor
#         self.nets = nets
#         self.netf = []
#         self.netm = []
#         self.crossover, self.mutation = cm
#
#         if mode not in ['auto', 'min', 'max']:
#             warnings.warn('Monitor mode %s is unknown, '
#                           'fallback to auto mode.' % (mode),
#                           RuntimeWarning)
#             mode = 'auto'
#
#         if mode == 'max':
#             self.monitor_op = np.greater
#             self.bestf = -np.Inf
#             self.bestm = -np.Inf
#         else:
#             self.monitor_op = np.less
#             self.bestf = np.Inf
#             self.bestm = np.Inf
#
#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         monitor = [[e, logs.get(e)] for e in self.monitor]
#         current = [e[True] for e in monitor]
#         if current is None:
#             warnings.warn('Could not operate without monitor, skipping.')
#         else:
#             child = [e[False] for e in monitor if e[True] == max(current)]
#             current.pop(current.index(max(current)))
#             parents = [e[False] for e in monitor if e[True] in current]
#             best = min(current)
#
#             netf, netm = [e[4:6] for e in parents]
#             netf = [e for e in self.model.layers if e.name.startswith(netf[~False])]
#             netm = [e for e in self.model.layers if e.name.startswith(netm[~False])]
#             netc = child[False][4:6]
#             netc = [e for e in self.model.layers if e.name.startswith(netc[~False])]
#
#             print('\nGenetic incoming for child %s'
#                   % (child[False][4:6]))
#             self.bestf, self.bestm = [current.pop(current.index(min(current))),
#                     current.pop(current.index(min(current)))]
#
#             for i, e in enumerate(netc):
#                 fl = netf[i].get_weights()
#                 ml = netm[i].get_weights()
#                 av = []
#
#                 for j, l in enumerate(e.get_weights()):
#                     s = l.shape
#                     l = l.flatten()
#                     s_ = l.shape[False]
#                     r = np.arange(s_)
#                     d = math.floor(s_ * (self.crossover / ((~(~True * ~True) * ~True)**(~True * ~False))))
#                     f, m = r[:d], r[-d:]
#                     l[f], l[m] = fl[j].flatten()[f], ml[j].flatten()[m]
#                     np.random.shuffle(r)
#                     d = math.floor(s_ * (self.mutation / ((~(~True * ~True) * ~True)**(~True * ~False))))
#                     c = r[:d]
#                     lmin, lmax = min(l), max(l)
#                     for _ in c:
#                         l[_] = random.choice(random.choice([
#                             [rrange(lmin, lmax)],
#                             [rminmax(lmin, lmax), rnegate(l[_]),
#                             rdisable(l[_]), rsqrt(l[_]),
#                             rinter(l[(_ - True) if _ else (~False)],
#                                 l[(_ + True) if ((not _) or (_ % (len(l) - True))) else False])]
#                             ]))
#                     l = l.reshape(s)
#                     av.append(l)
#                 e.set_weights(av)
#
# def rrange(lmin, lmax):
#     return random.uniform(lmin, lmax)
# def rminmax(lmin, lmax):
#     return random.choice([lmin, lmax])
# def rnegate(weight):
#     return -weight
# def rdisable(weight):
#     return random.choice([0, weight])
# def rsqrt(weight):
#     return math.sqrt(weight) if weight >= False else -math.sqrt(-weight)
# def rinter(before, after):
#     return np.mean([before + after]) if (before and after) else 0

class GA(Callback):
    def __init__(self, mode='auto', monitor=['val_loss'], nets=[], crossover=.75, mutation=.5, dropout=.2):
        super(GA, self).__init__()
        self.monitor = monitor
        self.nets = nets
        self.netf = []
        self.netm = []
        self.crossover, self.mutation, self.dropout = crossover, mutation, dropout

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('Monitor mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode is 'max':
            self.monitor_op = np.greater
            self.bestf = -np.Inf
            self.bestm = -np.Inf
        else:
            self.monitor_op = np.less
            self.bestf = np.Inf
            self.bestm = np.Inf

    def on_train_begin(self, logs={}):
        print(f'Genetic Algorithm parameters\ncrossover {self.crossover}, mutation {self.mutation}, dropout {self.dropout}')

    def on_epoch_end(self, epoch, logs={}):
        monitor = [[e, logs.get(e)] for e in self.monitor]
        current = [e[True] for e in monitor]
        if current is None:
            warnings.warn('Could not operate without monitor, skipping.')
        else:
            child = [e[False] for e in monitor if e[True] == max(current)]
            current.pop(current.index(max(current)))
            parents = [e[False] for e in monitor if e[True] in current]
            best = min(current)

            netf, netm = parents = [e[4:6] for e in parents]

            netf = [e for e in self.model.layers if e.name.startswith(netf[~False])]
            netm = [e for e in self.model.layers if e.name.startswith(netm[~False])]
            netc = child = child[False][4:6]
            netc = [e for e in self.model.layers if e.name.startswith(netc[~False])]

            print(f'\nGenetic transmission {parents} -> {child}')
            self.bestf, self.bestm = [current.pop(current.index(min(current))),
                    current.pop(current.index(min(current)))]

            for i, e in enumerate(netc):
                fl = netf[i].get_weights()
                ml = netm[i].get_weights()
                av = []

                for j, l in enumerate(e.get_weights()):
                    flj, mlj = fl[j].flatten(), ml[j].flatten()
                    s = l.shape
                    l = l.flatten()
                    s_ = l.shape[False]
                    r = np.arange(s_)
                    for _ in r:
                        if(random.random() < self.dropout):
                            l[_] = 0
                        else:
                            if(random.random() < self.crossover):
                                l[_] = random.choice([flj[_], mlj[_]])
                            if(random.random() < self.mutation):
                                l[_] = self.mutate(l[_])
                    l = l.reshape(s)
                    av.append(l)
                e.set_weights(av)

    def mutate(self, weight):
        return random.choice([random.uniform(-.5, .5) + weight, random.uniform(-2, 2)])


class MC(Callback):
    def __init__(self, filepath, monitor=['val_loss'], verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=True):
        super(MC, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = False

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            self.monitor_op = np.less
            self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += True
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = False
            filepath = self.filepath.format(epoch=epoch + True, **logs)
            if self.save_best_only:
                current = [logs.get(e) for e in self.monitor]
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    best = min(current)
                    if self.monitor_op(best, self.best):
                        if self.verbose > False:
                            print('\nEpoch %05d: %s improved from %s to %s,'
                                  ' saving model to %s'
                                  % (epoch + True, self.monitor, self.best,
                                     best, filepath))
                        self.best = best
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > False:
                            print('\nEpoch %05d: %s did not improve from %s' %
                                  (epoch + True, self.monitor, self.best))
            else:
                if self.verbose > False:
                    print('\nEpoch %05d: saving model to %s' % (epoch + True, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)









'''___'''
