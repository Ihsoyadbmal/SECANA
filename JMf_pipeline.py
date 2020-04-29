import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from JMf import Population, activations
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from tqdm import tqdm
from scipy.signal import savgol_filter

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
df = pd.read_pickle('D:/Documents/SMBT/SMBT_Nappe_Astienne/datas/pickles/smbt_day.pkl')
df = df.rename(columns={'cote piézométrique': 'cote', 'Cote piézométrique' : 'cote', 'Température en °C': 'températures', 'Précipitations en mm': 'précipitations'})
df = df[['précipitations', 'températures', 'cote']]

df['températures'] = df['températures'].rolling('90D').mean()
df['précipitations'] = df['précipitations'].rolling('90D').sum()
df = df.iloc[90:]

X = scaler.fit_transform(df.values[:, :-1])
Y = scaler.fit_transform(df.values[:, -1].reshape(-1, 1))

X = np.c_[X[20:], Y[:-20]]
Y = Y[20:]


from CTSGenerator import mackey_glass

mg = mackey_glass(length=2000, tau=200)


sns.lineplot(data=mg)

X = mg[:-50].reshape(-1, 1)
Y = mg[50:].reshape(-1, 1)

alpha = .01
beta = .9
epochs = 20
batch = 2
step = 25
preheat = 500

x = np.array([X[i: i + step] for i in range(len(X) - step)])

offset = len(x) % step
x_, y_ = X[offset:len(X)], X[offset:len(X)]
seed = 9604
# p = Population(x_, y_, ((20, 25), (10, 50), (5, 100), (5, 5)), input_step=batch, oshape=1, sparsity=.7, recurrence=.2, dropout=.2, alpha=alpha, seed=seed, leakage=.2)
p = Population(x_, y_, [2, (20, 5), 5], input_step=step, sparsity=.7, recurrence=.2, dropout=.2, alpha=alpha, seed=seed, leakage=.2, noise=.00)
# p = Population(x_, y_, (), input_step=step, oshape=1, sparsity=.7, recurrence=.2, dropout=.2, alpha=alpha, seed=seed, leakage=.2)

loss = np.array([])

# pca = PCA(n_components=25)

for _ in tqdm(range(epochs)):
    pred = np.array([])
    watchdog = np.array([])
    bpwatchdog = np.array([])
    for i, x in enumerate(range(0, len(p.inputs), step)):
        pred = np.r_[pred, p.feedforward(p.inputs[x:x + step]).reshape(-1)]
        if not (i % batch) and x > preheat:

            # pred.shape
            ##########
            # if iw is None:
            #     iw = (2. * np.random.binomial(1, .5, [len(p.olayer.connections), step]) - 1.) * .2
            #
            # # N_samples = res_states.shape[0]
            # # res_states = res_states.reshape(-1, 1)
            # # # ..transform..
            # # red_states = pca.fit_transform(res_states)
            # # # ..and put back in tensor form
            # # red_states = red_states.reshape(-1,1)
            #
            # coeff_tr = []
            # biases_tr = []
            #
            # for i in range(res_states.shape[0]):
            #     ridge_embedding.fit(res_states[i].reshape(1, -1), Y[i:i + 1])
            #     coeff_tr.append(ridge_embedding.coef_.ravel())
            #     biases_tr.append(ridge_embedding.intercept_.ravel())
            # print(np.array(coeff_tr).shape,np.array(biases_tr).shape)
            # input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
            ##########


            lr = ((p.outputs[x -  step:x].reshape(-1) - pred[x - step:x]) ** 2).mean()
            # dlr = (lr + p.olayer.dactivate(pred[x:x + batch])) * alpha
            dact = p.olayer.dactivate(pred[x - step:x])
            bp = (p.outputs[x - step:x].reshape(-1) - pred[x - step:x]) * dact * alpha
            # alpha *= .999
            for connection in p.olayer.connections:
                # np.linalg.pinv(np.c_[connection.weight, p.outputs[x - batch:x]]).T * p.olayer.dactivate(pred[x - batch:x]).reshape(-1, 1)
                delta = connection.destination.dactivation(bp.reshape(-1, 1) * connection.weight).reshape(-1, 1)
                # delta = connection.destination.dactivation(bp) * alpha
                # connection.source.dactivation(bp) * alpha
                # connection.destination.activation_name
                connection.weight += delta
            # for i, c in enumerate(p.olayer.pipeline):
            #     c.weight = c.weight + lr[i]
            #     watchdog = np.concatenate([watchdog, lr], axis=None)
            # noise += lr
            watchdog = np.r_[watchdog, lr]
            bpwatchdog = np.r_[bpwatchdog, np.zeros(((step * batch) - step)), bp]
            # dwatchdog = np.r_[dwatchdog, dlr]
    loss = np.r_[loss, np.inf if np.isnan(pred).any() else mse(p.outputs, savgol_filter(pred, 91, 3))]

connection.weight
connection.destination.activation_name

np.dot(np.linalg.pinv(np.c_[connection.weight, p.outputs[x - batch:x]]).T,
p.olayer.dactivate(pred[x - batch:x]).reshape(-1, 1))
connection.weight.shape
p.outputs[x - batch:x].shape

np.dot(np.linalg.pinv(np.c_[pred[500:], p.inputs[500:]]), p.olayer.dactivate(pred[500:]))

np.linalg.pinv(np.c_[connection.weight, p.outputs[-20:]]).T * p.olayer.dactivate(pred[-20:]).reshape(-1, 1)

1. - (np.tanh(pred[x - batch:x]) ** 2.)


x_.shape
y_.shape

p.ilayer.input.shape
p.ilayer.output.shape
p.ilayer.pipeline[0].weight.shape
p.ilayer.pipeline[0].source.state.shape
p.ilayer.pipeline[0].destination.state.shape
p.layers[0].pipeline[0].weight.shape
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

# plt.subplots(figsize=(20,10))
sns.lineplot(data=loss)

np.argmin(loss) + 1
loss


plt.subplots(figsize=(20,10))
sns.lineplot(data=np.c_[MinMaxScaler().fit_transform(pred.reshape(-1, 1)), MinMaxScaler().fit_transform(savgol_filter(pred, 91, 3).reshape(-1, 1))], palette='Reds', dashes=False)
# sns.lineplot(data=p.outputs, palette='Blues')
sns.lineplot(data=MinMaxScaler().fit_transform(np.c_[x_[:, :-1], y_]), palette='Blues', dashes=False)
sns.lineplot(data=np.r_[np.zeros((500)), bpwatchdog], dashes=False, palette='Greens')

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


p.connections[-1].destination.bias
_ =

np.mean(-alpha  * np.dot(x[:20].T, pred[:20] - y[:20]))# / len(x)
return - self.alpha * _




(1. / (1. + np.exp(-.8))) * (1.0 - (1. / (1. + np.exp(-.8))))



[array([7.99, 8.12, 5.1 , 4.81]) 1029]
[array([ 8.01,  8.01, 11.23,  7.57]) 15669]
[array([9.6 , 7.29, 5.96, 7.93]) 16625]
[array([80.82,  6.25,  7.36,  8.4 ]) 7813]
[array([13.84,  6.38,  7.08,  6.96]) 12205]
[array([7.01, 4.9 , 5.28, 6.7 ]) 29985]
[array([30.32,  6.71,  6.68,  6.41]) 25361]
[array([8.75, 8.76, 7.11, 4.55]) 22263]
[array([7.22, 5.7 , 6.7 , 6.31]) 1163]
[array([37.07, 18.32,  7.25,  6.63]) 31365]
[array([7.22, 5.82, 6.46, 7.19]) 1402]
[array([10.74,  9.52,  9.39,  8.64]) 18151]

[array([[44.92,  8.56],
       [20.64,  3.72],
       [17.79,  7.26],
       [ 8.82,  3.11]])
 389]

[array([[10.79, 28.9 ],
       [ 6.54, 14.11],
       [ 5.76, 11.8 ],
       [ 6.14,  7.64]])
 31577]







'''___'''
