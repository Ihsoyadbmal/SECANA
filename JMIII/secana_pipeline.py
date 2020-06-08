import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from secana import Population, activations
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from tqdm import tqdm
from scipy.signal import savgol_filter

import sqlite3

# import sounddevice as sd
# from scipy.io.wavfile import write
#
# _ = MinMaxScaler((-1, 1)).fit_transform(pred.reshape(-1, 1))
#
# len(_)
#
# sd.play(_, 2557)
#
# scaled = np.int16(_/np.max(np.abs(_)) * 32767)
# write('JM.wav', 2257, scaled)

scaler = MinMaxScaler()
# df = pd.read_pickle('D:/Documents/SMBT/SMBT_Nappe_Astienne/datas/pickles/smbt_day.pkl')
df = pd.read_pickle('D:/Documents/SMBT/Larzac/data/larzac_Cjass_sc1_Pblaq.pickle')
df = df[1500:]
df = df.rename(columns={'cote piézométrique': 'cote', 'Cote piézométrique' : 'cote', 'Température en °C': 'températures', 'Précipitations en mm': 'précipitations'})
df = df[['précipitations', 'températures', 'cote']]

df['températures'] = df['températures'].rolling('90D').mean()
df['précipitations'] = df['précipitations'].rolling('90D').sum()
df = df.iloc[90:]

X = scaler.fit_transform(df.values[:, :-1])
Y = scaler.fit_transform(df.values[:, -1].reshape(-1, 1))

X = np.c_[X[20:], Y[:-20]]
X = Y[:-20]
Y = Y[20:]

##########
conn = sqlite3.connect('data/yahoo.db')
cur = conn.cursor()

yahoo = cur.execute('select * from btc').fetchall()

col, *_ = zip(*cur.description)
btc = pd.DataFrame(yahoo, columns=col)

btc.loc[:, 'Date'] = pd.to_datetime(btc.Date)
btc = btc.set_index('Date')

sns.color_palette()

_, ax = plt.subplots(figsize=(20, 10))
h = [ax]
sns.lineplot(data=btc.iloc[-400:, :-1], dashes=False, ax=ax)
ax.set_ylabel('Dollars')
ax.legend_.remove()
ax = plt.twinx()
h += [ax]
sns.lineplot(data=btc.iloc[-400:, -1], dashes=False, ax=ax, color='goldenrod')
ax.set_ylabel('Volume')
handle = list(np.append(*[e.lines for e in h]))
plt.legend(handles=handle, labels=list(btc.columns), loc='upper right')
plt.title('Bitcoin market 30.04.20 - 02.06.20')

scaler = MinMaxScaler()

X = scaler.fit_transform(btc.dropna()[:-50][-800:].values)
Y = scaler.fit_transform(btc.dropna()['Close'][50:][-800:].values.reshape(-1, 1))

sns.lineplot(data=X)
sns.lineplot(data=Y)

X.shape
Y.shape

##########



from CTSGenerator import mackey_glass

mg = mackey_glass(length=2000, tau=200)

plt.subplots(figsize=(20,10))
sns.lineplot(data=mg)

X = mg[:-50].reshape(-1, 1)
Y = mg[50:].reshape(-1, 1)

alpha = .05
beta = .9
epochs = 5
batch = 2
step = 5
preheat = 50


x = np.array([X[i: i + step] for i in range(len(X) - step)])

offset = len(x) % step
x_, y_ = X[offset:len(X)], Y[offset:len(Y)]
validation = len(x_) - 300
seed = 9604
# p = Population(x_, y_, ((20, 25), (10, 50), (5, 100), (5, 5)), input_step=batch, oshape=1, sparsity=.7, recurrence=.2, dropout=.2, alpha=alpha, seed=seed, leakage=.2)
p = Population(x_, y_, [5, (20, 5), 5], input_step=step, sparsity=.7, recurrence=.2, dropout=.2, alpha=alpha, seed=seed, leakage=.05, noise=.00)

loss = np.array([])
# pca = PCA(n_components=25)

for _ in tqdm(range(epochs)):
    pred = np.array([])
    watchdog = np.array([])
    bpwatchdog = np.array([])
    deltawd = np.array([])
    for i, x in enumerate(range(0, len(p.inputs), step)):
        pred = np.r_[pred, p.feedforward(p.inputs[x:x + step]).reshape(-1)]
        if not (i % batch) and x > preheat and x < validation:

            lr = mse(p.outputs[:x + step].reshape(-1), pred)
            # dlr = (lr + p.olayer.dactivate(pred[x:x + batch])) * alpha
            # dact = p.olayer.dactivate(lr)
            dact = p.outputs[:x + step].reshape(-1) - p.olayer.dactivate(pred)
            # dact = p.olayer.dactivate(pred[:x])
            bp = dact * alpha
            # bp = (p.outputs[:x].reshape(-1) - pred[:x]) * dact * alpha
            for connection in p.olayer.connections:
                # np.linalg.pinv(np.c_[connection.weight, p.outputs[x - batch:x]]).T * p.olayer.dactivate(pred[x - batch:x]).reshape(-1, 1)
                delta = connection.destination.dactivation(bp[-step:].reshape(-1, 1) * connection.weight)
                # delta = connection.destination.dactivation(bp) * alpha
                # connection.source.dactivation(bp) * alpha
                # connection.destination.activation_name
                connection.weight += delta
            watchdog = np.r_[watchdog, lr]
            bpwatchdog = np.r_[bpwatchdog, np.zeros(((step * batch) - step)), bp[-step:]]
            deltawd = np.r_[deltawd, np.zeros(((step * batch) - step)), delta.reshape(-1)]
            # dwatchdog = np.r_[dwatchdog, dlr]
    loss = np.r_[loss, np.inf if np.isnan(pred).any() else mse(p.outputs, savgol_filter(pred, 91, 3))]

len(x_) // step // batch
len(bpwatchdog)

connection.weight

p.ilayer.input.shape
p.ilayer.output.shape
p.ilayer.pipeline[0].weight.shape
p.ilayer.pipeline[0].source.state.shape
p.ilayer.pipeline[0].destination.state.shape
p.layers[0].pipeline[0].weight.shape

np.dot(p.olayer.input, p.olayer.pipeline[0].weight)
p.olayer.pipeline[0].propagate(hidden=False)
p.olayer.input
p.olayer.pipeline[0].weight
p.olayer.pipeline[0].source.state
p.olayer.pipeline[0].destination.state
p.olayer.pipeline[1].source.state
p.olayer.pipeline[1].destination.state

[n.state for n in p.layers[-1].destination_nodes]
[n.state for n in p.olayer.destination_nodes]
[c.weight for c in p.olayer.pipeline]

p.ilayer.pipeline[0].weight

sns.lineplot(data=np.c_[watchdog, bpwatchdog], dashes=False)
sns.lineplot(data=watchdog, dashes=False)
sns.lineplot(data=bpwatchdog, dashes=False)
sns.lineplot(data=deltawd, dashes=False)

# plt.subplots(figsize=(20,10))
sns.lineplot(data=loss)

_, ax = plt.subplots(figsize=(20,10))
# sns.lineplot(data=np.c_[MinMaxScaler().fit_transform(pred.reshape(-1, 1)), MinMaxScaler().fit_transform(savgol_filter(pred, 91, 3).reshape(-1, 1))], palette='Reds', dashes=False)
sns.lineplot(data=MinMaxScaler().fit_transform(savgol_filter(pred, 91, 3).reshape(-1, 1)), palette='Reds', dashes=False, ax=ax)
sns.lineplot(data=p.outputs, palette='Blues', ax=ax)
sns.lineplot(data=MinMaxScaler().fit_transform(x_[:, :]), palette='Greys', dashes=False, ax=ax)
sns.lineplot(data=np.c_[np.r_[np.zeros((preheat)), bpwatchdog, np.zeros((len(x_) - validation))], np.r_[np.zeros((preheat)), deltawd, np.zeros((len(x_) - validation))]], dashes=False, palette='Greens', ax=ax)
h, l = ax.get_legend_handles_labels()
handles = np.r_[h[:2], h[6], h[-1:]]
plt.legend(handles=list(handles), labels=['Prediction', 'True'], loc='upper left')
plt.title('SECANA - BTC')

# sns.lineplot(data=pred)
plt.subplots(figsize=(20,10))
sns.lineplot(data=p.outputs[:], dashes=False, palette='Blues')
sns.lineplot(data=np.c_[pred, savgol_filter(pred, 91, 3)], dashes=False, palette='Reds')
# sns.lineplot(data=pred[:])
# sns.lineplot(data=x_[:, :])

plt.subplots(figsize=(20,10))
sns.lineplot(data=p.inputs, dashes=False, palette='Blues')
sns.lineplot(data=savgol_filter(pred, 91, 3), dashes=False, palette='Reds')

plt.subplots(figsize=(20,10))
sns.lineplot(data=savgol_filter(pred, 91, 3))


plt.subplots(figsize=(20, 10))
sns.lineplot(data=btc.Close['2018':])
plt.title('Bitcoin - 01.2018 06.2020')

_, ax = plt.subplots(figsize=(20,10))
sns.lineplot(data=MinMaxScaler().fit_transform(savgol_filter(pred, 91, 3).reshape(-1, 1)), palette='Reds', dashes=False, ax=ax)
# sns.lineplot(data=MinMaxScaler().fit_transform(pred.reshape(-1, 1)), palette='Reds', dashes=False, ax=ax)
# sns.lineplot(data=MinMaxScaler().fit_transform(x_), palette='Greens', dashes=False, ax=ax)
sns.lineplot(data=MinMaxScaler().fit_transform(y_), palette='Blues', dashes=False, ax=ax)
h, l = ax.get_legend_handles_labels()
ax.legend(h, ['pred', 'true'])
plt.title('SECANA - BTC')





'''___'''
