from __future__ import print_function
import numpy as np
import pandas as pd
from dvahedging import DVAHedging

np.set_printoptions(suppress=True)

# load data -------------------------------------------------------------------
my_dir = 'C:/Users/Giorgio/Desktop/DVA_hedging/python/'
filename = my_dir + '20170223_dataset_v7.csv'
data = pd.read_csv(filename, sep=';')

# INPUT : ---------------------------------------------------------------------
start_point = 3
allocazione_iniziale = np.array([10., 10., -32., 20., 1000000])

# scelgo tre azioni
action1 = np.array([0., 30., 5.])
action2 = np.array([10., 5., 10.])
action3 = np.array([10., 2., 0.])

# creazione oggetto -----------------------------------------------------------
mdp = DVAHedging(market_datafile=filename, horizon=100, start_point=start_point,
                 Initial_allocation_tot=allocazione_iniziale,
                 window_offset=0, verbose=0, sep=';')

# step ------------------------------------------------------------------------
state1, reward1, final, info = mdp.step(action1)
state2, reward2, final, info = mdp.step(action2)
state3, reward3, final, info = mdp.step(action3)
