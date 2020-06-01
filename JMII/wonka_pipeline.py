import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from pyESN import ESN
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler
from hyperopt import hp, tpe, rand, fmin, STATUS_OK, Trials
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from wonka import Hyperopt


# Winning lottery ticket

df = pd.read_pickle('data/smbt_day.pkl')
df = df.rename(columns={'Précipitations en mm': 'précipitations', 'Température en °C': 'températures', 'Cote piézométrique': 'cote'})

opt = Hyperopt(df,['précipitations', 'températures', 'cote'])

life = beautiful = None
# while pseudo-true fullrandom
winners = np.array([])
blind = False
oneshot = False

fs = Hyperopt.fullrsearch
bs = Hyperopt.bsearch


while(life is beautiful):
    mevals, algo, goldenticket = (100, rand.suggest, fs(3, 1, blind=blind, oneshot=oneshot))

    factory = Trials()
    try:
        chocolate = fmin(fn=opt.model, space=goldenticket, algo=algo,
                    max_evals=mevals, trials=factory, verbose=1)
    except:
        print('error full random')
        ...

    if(factory.results):
        args = list(factory.results[np.argmin([_['loss'] for _ in factory.results])].items())[1:-1]

        mevals, algo, goldenticket = (50, tpe.suggest, bs(args, blind=blind, oneshot=oneshot))
        factory = Trials()
        try:
            chocolate = fmin(fn=opt.model, space=goldenticket, algo=algo,
                        max_evals=mevals, trials=factory, verbose=1)
        except:
            print('error bayesian search')
            ...

        wonka = factory.results[np.argmin([_['loss'] for _ in factory.results])].items()
        inargs = dict(wonka).copy()
        inmin = inargs['loss']
        del inargs['loss'], inargs['status']
        inargs.update({'blind':False, 'oneshot':False, 'preheat':False})

        print(inmin)
        if(inmin <= 5):
            with open('data/Thau.json', 'a') as f :
                f.write(str(dict(wonka)) + '\n')
            print(dict(wonka))
            winners = np.append(winners, dict(wonka))

{'loss': 232.01, 'n_inputs': 3, 'n_outputs': 1, 'n_reservoir': 8, 'sparsity': 0.1387303878048083, 'random_state': 2788, 'spectral_radius': 1.655989960459667, 'noise': 0.12875804450216377, 'epochs': 1, 'step': 6, 'trainratio': 0.8916175440055693, 'status': 'ok'}

winners

# blind = True, oneshot = False
{'loss': 5.7, 'n_inputs': 2, 'n_outputs': 1, 'n_reservoir': 15, 'sparsity': 0.15705898659729958, 'random_state': 31428, 'spectral_radius': 2.2990281899386447, 'noise': 0.06813801702513478, 'epochs': 3, 'step': 26, 'trainratio': 0.9838501327189767, 'status': 'ok'}
{'loss': 9.41, 'n_inputs': 2, 'n_outputs': 1, 'n_reservoir': 8, 'sparsity': 0.10207334551327688, 'random_state': 2707, 'spectral_radius': 1.6906518389274818, 'noise': 0.22236706428603434, 'epochs': 1, 'step': 21, 'trainratio': 0.9822023608162257, 'status': 'ok'}

# blind = False, oneshot = False
{'loss': 4.09, 'n_inputs': 3, 'n_outputs': 1, 'n_reservoir': 9, 'sparsity': 0.18272586102190053, 'random_state': 22934, 'spectral_radius': 1.1730987200886898, 'noise': 0.07266047492797412, 'epochs': 2, 'step': 14, 'trainratio': 0.8531751272855373, 'status': 'ok'}
{'loss': 4.89, 'n_inputs': 3, 'n_outputs': 1, 'n_reservoir': 14, 'sparsity': 0.1892387811771885, 'random_state': 10726, 'spectral_radius': 1.5071150215457052, 'noise': 0.2366751529618585, 'epochs': 2, 'step': 11, 'trainratio': 0.9514292751549067, 'status': 'ok'}
{'loss': 4.37, 'n_inputs': 3, 'n_outputs': 1, 'n_reservoir': 15, 'sparsity': 0.21436557058294745, 'random_state': 25446, 'spectral_radius': 1.4586936126385823, 'noise': 0.13040745646627627, 'epochs': 2, 'step': 7, 'trainratio': 0.9450616683307573, 'status': 'ok'}


_ = [_['loss'] for _ in lottery.results]
np.array(lottery.results)[np.argpartition(_, 5)][:5]
tickets = np.array(lottery.results)[np.argpartition(_, 3)[:3]].tolist()
tickets = tickets if type(tickets) is list else [tickets]
ticket = list(tickets[0].items())[1:-1]


params = {
    'n_inputs': 3,
    'n_outputs': 1,
    'n_reservoir': 14,
    'sparsity': 0.1003518795869285,
    'random_state': 67,
    'spectral_radius': 1.7620377890517072,
    'noise': 0.051664953429020166,
    'epochs': 1,
    'step': 15,
    'trainratio': 1.0797105393589097}

params = {'n_inputs': 3, 'n_outputs': 1, 'n_reservoir': 12, 'sparsity': 0.1168209879921305, 'random_state': 15035, 'spectral_radius': 2.418830575772738, 'noise': 0.06257661804521956, 'epochs': 1, 'step': 5, 'trainratio': 1.143117538322185}

esn, s = opt.model(params, True)

opt.esnplot(opt.mdf, s, esn)









'''___'''
