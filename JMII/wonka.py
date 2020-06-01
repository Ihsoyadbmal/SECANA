import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt, networkx as nx
from datetime import datetime
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from hyperopt import hp, tpe, rand, fmin, STATUS_OK, Trials
from pyESN import ESN

class Hyperopt():

    def __init__(self, mdf, features=None, feature_range=(0, 1)):
        '''
        :param mdf: dataframe with target on df[:, -1]
        :type mdf: pd.DataFrame
        :param feature_range: range of the scaler default=(0, 1)
        :type feature_range: tuple(,)
        '''

        try:
            self.name = mdf.name
        except:
            self.name = None

        mdf = self.transform(mdf, features) if features else mdf

        self.mms = MinMaxScaler(feature_range=feature_range)
        self.mdf = self.mms.fit_transform(mdf.iloc[90:])
        _ = self.mms.fit(mdf.values[:, -1:])

        lim = self.mdf[:, -1:]
        self.m = (max(lim) - min(lim)) *.05
        self.mmin = min(lim) - min(lim) * .5
        self.mmax = max(lim) + max(lim) * .5

    def supesn(self, n_inputs, n_outputs, n_reservoir, sparsity, random_state, epochs, spectral_radius, noise, step=20, trainratio=1, blind=False, oneshot=False, preheat=False):
        params = {
            'n_inputs': int(n_inputs),
            'n_outputs': int(n_outputs),
            'n_reservoir': int(n_reservoir),
            'sparsity': sparsity,
            'random_state': int(random_state),
            'spectral_radius': spectral_radius,
            'noise': noise}
        esn = ESN(**params, silent=True)

        esn.epochs = int(epochs)
        esn.step = int(step)
        esn.trainlen = int(trainratio) if trainratio // 0b1010 else int((((len(self.mdf) // 3) * 2)) * trainratio)
        esn.totallen = (len(self.mdf) - esn.trainlen) - (len(self.mdf) - esn.trainlen) % esn.step
        esn.padding = len(self.mdf) - (esn.trainlen + esn.totallen)
        # esn.pmdf = self.mdf[esn.padding:]
        esn.pmdf = self.mdf[:esn.trainlen + esn.totallen]
        esn.trainratio = trainratio
        esn.blind = blind
        esn.oneshot = oneshot
        esn.preheat = preheat
        esn.name = self.name
        params.update({'epochs':esn.epochs, 'step':esn.step, 'trainratio':trainratio})

        return esn, params

    def model(self, params, rpred=False, rmmse=True):
        esn, params = self.supesn(**params)
        esn.current = 'MSE' if rmmse else 'MAX'

        for epoch in range(esn.epochs):
            pred_tot = np.zeros(esn.totallen)
            esndfcp = esn.pmdf[:esn.trainlen]

            if(esn.blind and esn.oneshot):
                for _ in range(esn.preheat):
                    _ = esn.fit(esndfcp[:esn.trainlen, :-1], esndfcp[:esn.trainlen, -1:])
                _ = esn.fit(esndfcp[:esn.trainlen, :-1], esndfcp[:esn.trainlen, -1:])
                prediction = esn.predict(esn.pmdf[esn.trainlen:, :-1])
                pred = prediction[:, 0]
                pred_tot = pred.copy()
            elif(esn.blind):
                for _ in range(esn.preheat):
                    _ = esn.fit(esndfcp[:esn.trainlen - esn.step, :-1], esndfcp[esn.step:esn.trainlen, -1:])
                for i in range(0, esn.totallen, esn.step):
                    _ = esn.fit(esndfcp[i:i + esn.trainlen - esn.step], esndfcp[i + esn.step:i + esn.trainlen, -1:])
                    prediction = esn.predict(esndfcp[i + esn.trainlen - esn.step: i + esn.trainlen, :-1])
                    pred = prediction[:, 0]
                    pred_tot[i:i + esn.step] = pred
                    esndfcp = np.append(esndfcp, np.c_[esn.pmdf[i + esn.trainlen: i + esn.trainlen + esn.step, :-1], pred], axis=0)
            else:
                for _ in range(esn.preheat):
                    _ = esn.fit(esndfcp[:esn.trainlen - esn.step], esndfcp[esn.step:esn.trainlen, -1:])
                for i in range(0, esn.totallen, esn.step):
                    _ = esn.fit(esndfcp[i:i + esn.trainlen - esn.step], esndfcp[i + esn.step:i + esn.trainlen, -1:])
                    prediction = esn.predict(esndfcp[i + esn.trainlen - esn.step: i + esn.trainlen])
                    pred = prediction[:, 0]
                    pred_tot[i:i + esn.step] = pred
                    esndfcp = np.append(esndfcp, np.c_[esn.pmdf[i + esn.trainlen: i + esn.trainlen + esn.step, :-1], pred], axis=0)

        pred_clip = pred_tot.copy()

        pred_clip[np.where((pred_clip < self.mmin) | (pred_clip > self.mmax))] = np.NaN
        count = np.isnan(pred_clip).sum()
        while(True):
            pred_clip = pd.Series(pred_clip).interpolate().bfill().ffill().values
            if(np.isnan(pred_clip).all()): break
            pred_clip[np.where((pred_clip[1:] < pred_clip[:-1] - self.m) | (pred_clip[1:] > pred_clip[:-1] + self.m))] = np.NaN
            if(not np.isnan(pred_clip).sum()): break

        s = pred_clip if np.isnan(pred_clip).any()\
            else self.mms.inverse_transform(savgol_filter(pred_clip, 91, 3).reshape(-1, 1))

        sdfcp = self.mms.inverse_transform(savgol_filter(esn.pmdf[esn.trainlen:esn.trainlen+esn.totallen, -1], 91, 3).reshape(-1, 1))

        esn.mmse, esn.mmax = (np.inf, np.inf) if np.isnan(s).any() else self.getmse(s, sdfcp)

        return (esn, s) if rpred else {'loss': (esn.mmse if rmmse else esn.mmax), **params, 'status': STATUS_OK}

    @staticmethod
    def getmse(s, sdfcp):
        return (round(np.sqrt(mse(s, sdfcp)) * 100, 2),\
            np.round(max(abs(s - sdfcp)) * 100, 2)[0])

    @staticmethod
    def bsearch(args, blind=False, oneshot=False):
        return dict([(k,\
            v if k in ['n_reservoir', 'random_state', 'n_inputs', 'n_outputs'] else\
            hp.quniform(k, v - 1, v + 1, 1) if ((k is 'epochs' and v - 1) and (type(v) is int)) else\
            hp.quniform(k, 1, 3, 1) if (k is 'epochs') else\
            hp.uniform(k, v - .1, v + .1) if v // 1 else\
            hp.uniform(k, v - .01, v + .01) if v // .1 else\
            hp.uniform(k, v - .001, v + .001)) for k, v in args] +\
            [('blind', bool(blind)), ('oneshot', bool(oneshot))])

    @staticmethod
    def rsearch(args, blind=False, oneshot=False):
        return dict([(k,\
            hp.quniform(k, 2, 4, 1) if k is 'epochs' else\
            v if ((type(v) is int) or (k is 'trainratio')) else\
            hp.uniform(k, v - .5, v + .5) if v // 1 else\
            hp.uniform(k, v - .05, v + .05) if v // .1 else\
            hp.uniform(k, v - .005, v + .005)) for k, v in args] +\
            [('blind', bool(blind)), ('oneshot', bool(oneshot))])

    @staticmethod
    def fullrsearch(n_inputs, n_outputs, blind=False, oneshot=False):
        return {
            'n_inputs': int(n_inputs),
            'n_outputs': int(n_outputs),
            'n_reservoir' : hp.quniform('n_reservoir', 100, 250, 25),
            'sparsity' : hp.uniform('sparsity', .3, .6),
            # 'sparsity' : hp.uniform('sparsity', .003, .20),
            'random_state' : hp.randint('random_state', 2**15-1),
            'epochs' : 1,
            'spectral_radius' : hp.uniform('spectral_radius', .5, 2.5),
            'step': hp.quniform('step', 5, 100, 1),
            'trainratio': hp.uniform('trainratio', .2, 1.1),
            'noise' : hp.uniform('noise', 0.003, .25),
            'blind' : bool(blind),
            'oneshot': bool(oneshot)
            }

    @staticmethod
    def transform(df, features):
        mdf = df.copy()

        mdf[features[0]] = df[features[0]].rolling(window=('90D'), min_periods=1).sum()
        mdf[features[1]] = df[[features[1]]].rolling(window=('90D'), min_periods=1).mean()

        mdf = mdf[features]

        return mdf

    def esnplot(self, pred, esn, bonus=None, bonus_label='', offset=0, scale=1, save=None):
        legend = []
        plt.figure(figsize=(16, 8))
        legend += plt.plot(range(0, esn.trainlen + esn.totallen),\
            (self.mms.inverse_transform(savgol_filter(esn.pmdf[:esn.trainlen + esn.totallen, -1], 91, 3).reshape(-1, 1)) + offset) * scale,\
            'b', label="True", alpha=.5, color='blue')
        legend += plt.plot(range(esn.trainlen, esn.trainlen + esn.totallen),\
            (pred + offset) * scale,\
            'k', alpha=.8, label='Walk Forward prediction', color='orange')

        lo, hi = plt.ylim()
        plt.plot([esn.trainlen, esn.trainlen], [lo + np.spacing(1), hi - np.spacing(1)], 'k:', linewidth=2)

        plt.title(esn.name or r'Ground Truth and Echo State Network Output', fontsize=25)
        plt.xlabel(r'Time (Days)', fontsize=20, labelpad=10)
        plt.ylabel(r'NP', fontsize=20, labelpad=10)
        if(bonus):
            plt.twinx()
            legend += plt.plot(range(0, esn.trainlen + esn.totallen),\
                bonus.values, label=bonus.name, color='green', alpha=.5)
            plt.ylabel(bonus_label or bonus.name, fontsize=20, labelpad=10)
        plt.legend(legend, [e.get_label() for e in legend], fontsize=15, loc='upper left')

        if(save):
            dnow = datetime.now().strftime('%Y%m%d-%H%M%S')
            model = {
                'date': dnow,
                'epochs': esn.epochs,
                'spectral_radius': esn.spectral_radius,
                'noise': esn.noise,
                'n_reservoir': esn.n_reservoir,
                'sparsity': esn.sparsity,
                'random_state': esn.random_state,
                'n_inputs': esn.n_inputs,
                'n_outputs': esn.n_outputs,
                'step': esn.step,
                'trainratio': esn.trainratio}
            res = '_ESN_'

            fullpath = save + str(esn.mmse) + '_' + str(esn.mmax) + res + dnow
            plt.savefig(fullpath + '.png')
            with open(fullpath + '.json', 'w') as f:
                f.write(str(model))

    @staticmethod
    def plotweights(w, innull=None):
        w = nx.DiGraph(w)
        esmall = [(u, v) for (u, v, d) in w.edges(data=True) if d['weight'] < (0 if not innull else innull[0])]
        elarge = [(u, v) for (u, v, d) in w.edges(data=True) if d['weight'] > (0 if not innull else innull[1])]

        pos = nx.spring_layout(w)
        plt.subplots(figsize=(20,10))
        _ = nx.draw_networkx_nodes(w, pos, node_size=150, node_color='k')
        _ = nx.draw_networkx_edges(w, pos, edgelist=elarge,
                               width=2, alpha=.5, edge_color='r')
        _ = nx.draw_networkx_edges(w, pos, edgelist=esmall,
                               width=2, alpha=.5, edge_color='b', style='dashed')







'''___'''
