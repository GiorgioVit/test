from __future__ import print_function
import gym
from gym import spaces
from gym.utils import seeding
from gym.spaces import prng
import numpy as np
import pandas as pd
from datetime import datetime
import time
import random
import os
import xlsxwriter
from openpyxl import load_workbook
from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
from rllab.envs.env_spec import EnvSpec
import collections
from cached_property import cached_property



class DVAHedging(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    """
    This class simulates the evolution of the DVA and portfolio
    given a set of market data.

    The most relevant components of this class are

    - window_offset: how many step we consider in the past. Zero means that we consider only the
                     information associated to the current date.
    - allocation_delta: store the changes in the allocation of the
                        instruments as a consequence of the performed action.
                        Actually it is composed by changes in
                        [SX7E, BTP, BUND, ITRAXX, Cash Flow (i.e., bank account)]
                        Shape: [(window_offset + 1) * n_securities x 1]
    - allocation_tot: is the overall amount or quantity associated to each instrument.
                      It is the consequence of the performed action and the interest rates.
                      [SX7E, BTP, BUND, ITRAXX, Cash Flow (i.e., bank account)]
                      Shape: [n_securities x 1]
    - prices: [SX7E, SPREAD BTP/BUND, ITRAXX, spread 5y]
              Shape: [5 x (window_offset + 1)] 
    - portfolio_sensitivities: contains the sensitivities*multipliers and tha DVA sensitivity 
                               [SX7E, BTP, BUND, ITRAXX, DVA]
    - state: contains the information that can be perceived by the agent.
             [portfolio_sensitivities]
    """

    alpha = 0.09           # costo del capitale
    number_securities = 5  # sx7e, btp, bund, iTraxx, Cash Flow
    n_prices = 4           # sx7e, btp-bund spread, iTraxx, intesa CDS spread 5y
    n_step_day = 96        # number of step each day
    multiplier = np.array([50, 1000, 1e6, 1])      # array contenente la dimensione dei lotti di sx7e, BTP,ITRAXX e cassa (definita a 1) - non viene definita la dim dei lotti di BUND perche' e' uguale a quella dei BTP
    max_allocation = np.array([11500, 2100, 700])  # array contenente la dimensione massima di allocazione per SX7E,BTP,ITRAXX- non viene definito il max BUND perche' viene calcolato implicitamente successivamente come max bTP*(sensitivityBTP/sensitivityBUND)
    max_actions = np.array(max_allocation)         # stessa dimensione del vettore sopra perche' l'agente non sceglie la quantita' di BUND, ma quella di BTP. e' il simulatore a "forzare" la scelta della quantita' di BUND
  

    def __init__(self, market_datafile, horizon=100, window_offset=3, window_offset_day=0, risk_factor=1,
                 window_offset_week=0, gamma=1.0, dva_nominal=400e6, start_point=None, grid_horizon=0,state_flag='BTP_baseline',
                 Initial_alloc_tot=None, empty_allocation=1, training=True, split=3/4, verbose=0, low_bound=1, sep=','):

        # validate inputs
        assert horizon > 0, 'horizon must be greater than zero'
        assert dva_nominal > 0, 'DVA nominal must be greater than zero'
        assert window_offset >= 0, 'window_offset must in [0,{}'.format(horizon)
        assert window_offset_day >= 0, 'window_offset_day must be greater then zero'
        assert window_offset_week >= 0, 'window_offset_week must be greater then zero'

        self.state_flag = state_flag
        self.risk_factor = risk_factor
        self.horizon = horizon  # dimensione dell'episodio di simulazione, viene passato come parametro nel momento in cui si crea un istanza di questa classe
        self.window_offset = window_offset  # quanto indietro voglio che l'agente guardi ossia di quanti step prima di oggi voglio tener traccia nello stato. Se window_offset=0 => solo oggi
        self.window_offset_day = window_offset_day  #
        self.window_offset_week = window_offset_week
        self.market_datafile = market_datafile
        self.data = pd.read_csv(market_datafile, sep=sep)

        self.data['delete'] = [str(self.data['data_column'].iloc[i]) == "nan" for i in range(len(self.data))]
        self.data = self.data[self.data['delete'] == False]

        self.price_dim = self.n_prices * (self.window_offset + 1)  # l'array price sara' 5-1 perche' non terra' traccia separatamente di BTP e BUND * il numero di giorni di cui voglio tenere traccia
        self.daily_price_dim = self.n_prices * (self.window_offset_day)
        self.state_dim = self.get_state_dim()
        self.start_point = start_point  # parametro passato alla classe quando si crea un istanza di questo tipo. Se start_point=None l'algoritmo inizia da un punto casuale del dataset.

        self.low_bound = low_bound  # if I don't want to consider tha data from 2009, set the low bound (ex 846-->2013)
        self.training = training
        self.split = split
        self.verbose = verbose  # se verbose=1 vengono stampate informazioni aggiuntive che possono essere utili in fase di debug
        self.gamma = gamma  # utile all'algoritmo
        self.dva_nominal = dva_nominal  # dato in input
        self.empty_allocation = empty_allocation  # se empty allocation = 1 viene creato un portafoglio vuoto
                                                  # se empty allocation = 0 viene creato un portafoglio iniziale come definito dal metodo _initial_allocation()
                                                  # se empty allocation = 2 viene creato un portafoglio iniziale sulla base di Initial_allocation_tot
        self.grid_horizon = grid_horizon
        if grid_horizon > 0:
            rows = self.data.shape[0] - low_bound - 1
            l_bound = low_bound + max([96 * 5 * window_offset_week, 96 * window_offset_day, window_offset]) + 1
            u_bound = low_bound + round(split * rows) -1
            grid_train = []
            n_grid = 1
            while (l_bound + horizon * n_grid) < u_bound:
                grid_train.append(l_bound + horizon * (n_grid-1))
                n_grid = n_grid + 1
            self.grid_train = grid_train
            self.initial_grid_train = grid_train


        if Initial_alloc_tot is None:
            self.allocation_tot = np.zeros(self.number_securities)
        else:
            self.allocation_tot = Initial_alloc_tot
            self.empty_allocation = 2
            self.Initial_alloc_tot = Initial_alloc_tot
        self.step_gain_cumulato = 0
        self.hedge_cumulato = 0
        self.reward_cumulata = 0
        self.step_PL = 0
        self.reward = 0
        self.pl_cumulato = 0
        self.delta_DVA_cumulato = 0
        self.allocation_size = (self.number_securities * (self.window_offset + 1))

        # gym attributes
        self.viewer = None
        self.action_space = Box(low=-self.max_actions,
                                high=self.max_actions)
        self.observation_space = Box(low=np.array([-1e15] * self.state_dim),
                                     high=np.array([1e15] * self.state_dim)
                                     )

        # Utilities per print / scrittura output-------------------------------------------
        self.print_output = 0
        self.write_to_file = 0

        if self.write_to_file:
            self.output_folder = './'
            if os.path.isfile(self.output_folder + 'output.xlsx'):
                os.remove(self.output_folder + 'output.xlsx')
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
            workbook = xlsxwriter.Workbook(self.output_folder + 'output.xlsx')
            workbook.close()

            self.row_index_output = 0  # per tenere traccia della cella su cui viene salvato l'output
            # Aggiunta dataframe di output (per salvataggio su file)
            self.output_df = pd.DataFrame(
                columns=['BTP', 'Bund', 'FINSNR', 'SX7E', 'Cassa', 'CDS Intesa', 'DVA', 'P&L(dailygain)', 'Reward'],
                # Modifica 28-02: aggiunta reward
                index=['Sensitivity', 'Delta lotti', 'Lotti tot', 'Delta Prezzo', 'Valore mercato'])

        # initialize state-------------------------------------------------------------------
        self.seed()  # metodo per la generazione di un seme casuale
        self.reset()  # metodo richiamato all'inizio dell'episodio

    """
    metodo per la generazione di un seme casuale
    """

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    """
    Metodo centrale del simulatore.
    Esegue un passo alla volta i calcoli necessari a simulare l'evolversi dell'ambiente in seguito alla scelta dell'agente
    """

    def step(self, action):
        # salvataggio variabili temporanee--------------------------------------------
        if len(action) == 1:
            action = [0.0, action[0], 0.0]
        elif len(action) == 2:
            action = [0.0, action[0], action[1]]

        number_securities = self.number_securities
        multiplier = self.multiplier
        dva_nominal = self.dva_nominal
        price_dim = self.price_dim
        allocation_size = self.allocation_size
        index = self.index
        new_capital = self.capital

        assert len(self.allocation_tot) == number_securities

        # last_series è utile per navigare allocation_delta
        last_series = (number_securities * (self.window_offset + 1)) - number_securities

        # all'inizio, reinizializzo il daily_hedge e la variaziazione di cassa a 0
        daily_hedge = 0
        self.allocation_delta[last_series + number_securities - 1] = 0

        # aggiorno prezzi e sensitivities dello stato i-1
        sensitivities = self._compute_sensitivity(index=index)

        price = self._compute_price(index=index)

        # controllo se sono a fine giornata
        orario = self.data.ix[index, 'orario_column']

        if orario == "17:25":
            chiusura = 1
        else:
            chiusura = 0
        self.chiusura = chiusura

        # traslo allocation_delta di una serie a sx
        for i in range(last_series):
            self.allocation_delta[i] = self.allocation_delta[i + number_securities]

            # check allocation range
        tmp_alloc_tot = [self.allocation_tot[0], self.allocation_tot[1],
                         self.allocation_tot[3]]  # vettore temporaneo per le allocazioni btp
        self.tmp_alloc_tot = tmp_alloc_tot
        delta_action_p = self.max_allocation - tmp_alloc_tot
        delta_action_m = - self.max_allocation - tmp_alloc_tot
        clip_allocation = np.clip(action, delta_action_m, delta_action_p)

        self.allocation_delta[-number_securities] = clip_allocation[0]  # SX7E
        self.allocation_delta[-number_securities + 1] = clip_allocation[1]  # BTP
        # DA CONTROLLARE
        self.allocation_delta[-number_securities + 2] = - round(
            clip_allocation[1] * sensitivities[-4] / sensitivities[-3])  # BUND
        self.allocation_delta[-number_securities + 3] = clip_allocation[2]  # ITRAXX

        # sistemo l'allocazione totale che ho su ogni securities. (non il bank account).
        # N.B.: allocation_tot sono quindi le allocazioni dopo che l'azione è stata presa!
        self.allocation_tot[:number_securities - 1] = self.allocation_tot[:number_securities - 1] \
                                                      + self.allocation_delta[
                                                        last_series:last_series + number_securities - 1]

        # gestione tassi interesse (r1 è temporeaneo...)--------------------------

        rate = float(self.data.ix[index, "CDS_Spread_1Y"]) / 100.0
        r1 = 0.00  # SCELTO ARBITRARIAMENTE PER TEST, SARà DA MODIFICARE
        r0 = r1 + rate

        if chiusura:
            # calcolo variazione temporale -------------------------------------------
            current_day = datetime.strptime(self.data.ix[index, 'data_column'], '%d/%m/%Y')
            next_day = datetime.strptime(str(self.data.ix[index + 1, 'data_column']), '%d/%m/%Y')
            delta_days = np.round((next_day - current_day).total_seconds() / 86400.0)

        # calcolo il collaterale in (i-1)+
        self.collateral = self._compute_collat(index)
        # calcolo i dividendi del collaterale se sono a fine giornata
        if chiusura:
            dividend_collateral = -(self.collateral * r1 * (delta_days / 360.0))


        # DVA di ieri
        DVA_tm1 = self._compute_dva(index)

        # VADO AVANTI DI UN GIORNO --------------------------------------------------------
        index += 1
        # ---------------------------------------------------------------------------------

        self.orario = self.data.ix[index, 'orario_column']
        # aggiorno vettore price
        price = self._compute_price(index=index)
        sensitivities = self._compute_sensitivity(index=index)
        self.time_to_roll = self.compute_time_to_roll(index=index)

        DVA = self._compute_dva(index)

        # gestione indice sx7e, che tiene conto delle roll-dates --------------------------
        if self.data.ix[index - 1, "SX7E_Time_To_Roll"] == 0 and chiusura:
            sx7e_price_tm1 = self.data.ix[index - 1, "SX7E_Price_NewSerie"]
        else:
            sx7e_price_tm1 = self.data.ix[index - 1, "SX7E_Price"]
        self.sx7e_price_tm1 = sx7e_price_tm1  # lo salvo perché lo re-utilizzo per calcolare l'RC
        sx7e_price = self.data.ix[index, "SX7E_Price"]

        self.delta_sx7e = (sx7e_price - sx7e_price_tm1)  # Aggiunto per output su file
        new_value = self.allocation_tot[0] * multiplier[0] * self.delta_sx7e

        # update cassa
        daily_hedge += new_value

        # gestione BTP, che tiene conto delle roll-dates ----------------------------------
        if self.data.ix[index - 1, "BTP_Time_To_Roll"] == 0 and chiusura:
            btp_tm1 = self.data.ix[index - 1, "BTP_Price_NewSerie"]
        else:
            btp_tm1 = self.data.ix[index - 1, "BTP_Price"]
        btp_tm = self.data.ix[index, "BTP_Price"]
        delta_btp = btp_tm - btp_tm1
        btp_value = self.allocation_tot[1] * multiplier[1] * delta_btp
        self.delta_btp = delta_btp

        # gestione BUND, che tiene conto delle roll-dates ----------------------------------
        if self.data.ix[index - 1, "BUND_Time_To_Roll"] == 0 and chiusura:
            bund_tm1 = self.data.ix[index - 1, "BUND_Price_NewSerie"]
        else:
            bund_tm1 = self.data.ix[index - 1, "BUND_Price"]
        bund_tm = self.data.ix[index, "BUND_Price"]
        delta_bund = bund_tm - bund_tm1
        bund_value = self.allocation_tot[2] * multiplier[1] * delta_bund
        self.delta_bund = delta_bund

        new_value = btp_value + bund_value
        # update cassa
        daily_hedge += new_value

        # gestione itraxx -------------------------------------------------------------------
        # self.delta_itraxx = self.data.ix[index,"Itraxx_onPrice_BID"] - self.data.ix[index-1,"Itraxx_onPrice_BID"] #used only for printing output

        # itraxx_dividend=self.data.ix[index,"ITRAXX_Dividend"]*chiusura
        # new_value = self._compute_delta_iTraxx(index) - self.allocation_tot[3] * multiplier[2] * itraxx_dividend

        # gestione itraxx -------------------------------------------------------------------

        self.delta_itraxx = (self.data.ix[index, "Itraxx_onPrice_BID"] + self.data.ix[index, "Itraxx_onPrice_ASK"]) / 2 \
                            - (self.data.ix[index - 1, "Itraxx_onPrice_BID"] + self.data.ix[index - 1, "Itraxx_onPrice_ASK"]) / 2
        if self.data.ix[index, "ITRAXX_Time_To_Roll"] == 0 and chiusura:
            itraxx_dividend = self.data.ix[index, "ITRAXX_Dividend"]
        else:
            itraxx_dividend = 0

        new_value = self._compute_delta_iTraxx(index) + self.allocation_tot[3] * multiplier[2] * itraxx_dividend

        # update cassa

        daily_hedge += new_value

        # RC-------------------------------------------------------------------------------
        # Update valore di classe
        self.sensitivities = sensitivities
        self.prices = price

        self.RC = self._compute_RC(price, sensitivities, init=0)


        # se sono a fine giornata considerare anche l'increment_for_interest, il  dva_dividend e il costo del capitale
        if chiusura:
            self.increment_for_interest = self.allocation_tot[4] * r0 * (
            delta_days / 360.0)  # allocation_tot[4] a questo punto non è ancora stata aggiornata (i-1)
            self.cost_RC = self.RC * self.alpha * (delta_days / 360.0)
            if self.data.ix[index, "CDS_Time_To_Roll"] == 0:
                dva_dividend = self._compute_dva_dividend(index)
            else:
                dva_dividend = 0

        # calcolare la variazione del DVA
        delta_DVA = DVA - DVA_tm1
        self.delta_DVA = delta_DVA


        # allocation_delta[last_series + 4] viene sempre reinizializzata a 0
        if chiusura:
            self.allocation_delta[
                last_series + 4] = daily_hedge + dva_dividend + self.increment_for_interest + dividend_collateral - self.cost_RC
        else:
            self.allocation_delta[last_series + 4] = daily_hedge

            # Update cassa totale
        self.allocation_tot[4] += self.allocation_delta[last_series + 4]

        # guadagno step


        if chiusura:
            step_gain = daily_hedge + self.increment_for_interest + dividend_collateral - self.cost_RC
        else:
            step_gain = daily_hedge

        assert np.isfinite(step_gain), 'daily gain is infinite... stopping'
        # Update new_capital
        new_capital += step_gain

        # calcolo del reward
        if chiusura:
            self.step_PL = step_gain +  delta_DVA + dva_dividend
            reward = self.f_reward(self.step_PL)
        else:
            self.step_PL = step_gain + delta_DVA
            reward = self.f_reward(self.step_PL)


        if self.verbose > 0:
            # np.set_printoptions(threshold=np.inf)
            if chiusura:
                print('difference = {}'.format(delta_days))
            print('delta allocation: {}'.format(self.allocation_delta[-5:]))
            print('total allocation: {}'.format(self.allocation_tot))
            print('prices : {}'.format(self.prices[-self.n_prices:]))
            print('step_gain: {}'.format(step_gain))

        # Aggiornamento variabili di classe ---------------------------------------------
        self.daily_hedge = daily_hedge
        self.daily_dva = DVA
        self.delta_DVA = delta_DVA
        self.delta_DVA_cumulato += delta_DVA
        self.reward = reward
        self.capital += new_capital
        self.index = index

        self.step_gain = step_gain
        self.step_gain_cumulato += step_gain
        self.hedge_cumulato += self.daily_hedge
        self.reward_cumulata += reward
        self.pl_cumulato += self.step_PL
        if chiusura:
            self.current_date = next_day.date()  # update data attuale
            if self.window_offset_day > 0:
                self.daily_prices = self.find_daily_prices(index)
            if self.window_offset_week > 0:
                self.weekly_prices = self.find_weekly_prices(index)
        self.V2X = self.data.ix[index, "V2X_VALUE"]
        self.VIX = self.data.ix[index, "VIX_VALUE"]

        # Update state ---------------------------------------------------------------------
        # nota: per ora lo stato non contiene i prezzi della nuova serie
        daily_prices_dim = self.n_prices * self.window_offset_day
        weekly_prices_dim = self.n_prices * self.window_offset_week
        assert daily_prices_dim == len(self.daily_prices), 'Wrong dimension of daily_prices'
        assert weekly_prices_dim == len(self.weekly_prices), 'Wrong dimension of weekly_prices'

        self.state = self.get_state()

        # STAMPA -----------------------------------------------------------------------------
        if self.print_output:
            print("------------ UNO STEP IN AVANTI ------------\n\ndata attuale: ", self.current_date)
            print("orario:  ", self.orario)
            print("index:  ", self.index)
            #            self.print_pretty()
            print('\nlotti SX7E-BTP-ITRAXX')
            print('delta allocation: {}'.format(self.allocation_delta[-5:-1]))
            print('total allocation: {}'.format(self.allocation_tot))
            print('\nprices : [SX7E, SPREAD BTP/BUND, ITRAXX bid, ITRAXX ask, Cash Flow (i.e., bank account)] ')
            print(self.prices[-self.n_prices:])

            print("\nP&L = dailygain + delta DVA: ", self.step_PL)  # corretto "-" con "+"
            print("delta DVA: ", self.delta_DVA)
            print("daily gain: ", step_gain)

            if chiusura:
                print("DVA dividend: ", dva_dividend)
                print("Costi RC : ", self.cost_RC)
                print("Iteresse Bank Account: ", self.increment_for_interest)
                print("Interessi Collateral : ", dividend_collateral)
            print("")
            print("---------------------------------------------------------------------")
            print("---------------------------------------------------------------------")

        # TEST OUTPUT_DF
        if self.write_to_file:
            self.itraxx_on_spread = self.data.ix[index, "Itraxx_Spread_BID"]  # Aggiunto per output su file
            self.itraxx_on_spread_tm1 = self.data.ix[index - 1, "Itraxx_Spread_BID"]  # Aggiunto per output su file
            self.cds_spread_5y_tm1 = self.data.ix[index - 1, "Spread_cds_5Y_BID"]  # Aggiunto per output su file
            self.cds_spread_5y = self.data.ix[index, "Spread_cds_5Y_BID"]  # Aggiunto per output su file
            self._compute_output()
            self._compute_output_step()
            self._write_output_df()
            self.row_index_output += 7
            print(self.output_df)
        # return self.get_state(), reward, False, {}
        return Step(observation=self.get_state(), reward=reward, done=False)

    def reset(self,
              state=None):  # definisce l'indice di partenza casualmente se questo non viene passato come parametro. Inoltre, se esiste uno stato lo restituisce altrimenti richiama il metodo init_price
        if state is None:
            if self.grid_horizon>0:
                if len(self.grid_train)>1:
                    self.index = self.grid_train[0]
                    self.grid_train = self.grid_train[1:]
                else:
                    self.index = self.grid_train[0]
                    self.grid_train = self.initial_grid_train

            elif self.start_point is not None:

                self.index = self.start_point
            else:
                # number of rows in our dataset
                rows = self.data.shape[0] - self.low_bound - 1
                # splitting the dataset: 3/4 for training and 1/4 for testing
                if self.training:
                    l_bound = self.low_bound + np.max([self.window_offset, self.window_offset_day * self.n_step_day,
                                                       self.window_offset_week * self.n_step_day * 5])
                    u_bound = self.low_bound + round(self.split * rows)
                else:
                    l_bound = self.low_bound + round(self.split * rows) + 1
                    u_bound = self.data.shape[0]
                self.index = prng.np_random.randint(low=l_bound, high=u_bound - self.horizon)

            self._init_price(index=self.index)

        else:
            state_dim = self.state_dim
            assert len(state) == state_dim, 'Wrong dimension of the provided state'
            self.state = np.array(state)

        return self.get_state()

    """
    Restituisce lo spazio di osservazione
    """

    def observation_space(self):
        return Box(low=np.array([-1e15] * self.state_dim), high=np.array([1e15] * self.state_dim))

    """
    Restituisce lo spazio di azione
    """

    def action_space(self):
        return Box(low=-self.max_actions, high=self.max_actions)

    def action_dim(self):
        return 3

    def render(self):
        print('current state:', self.state)

    def spec(self):
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

    """
    Horizon
    """

    def horizon(self):
        """
        Horizon of the environment, if it has one
        """
        return self.horizon

    """
    Restituisce lo stato del simulatore
    """

    def get_state(self):
        # passare dalla sensitivy per lotti a quella totale del portafoglio
        state_flag = self.state_flag
        sensitivities = self.sensitivities
        allocation_tot = self.allocation_tot

        if state_flag == 'bsl':
            baseline_features = np.array([round(sensitivities[-1]/sensitivities[-4]), round(sensitivities[-1]/sensitivities[-2])])
            return np.concatenate((baseline_features, allocation_tot))
        elif state_flag == 'bsl_prices':
            baseline_features = np.array([round(sensitivities[-1]/sensitivities[-4]), round(sensitivities[-1]/sensitivities[-2])])
            return np.concatenate((baseline_features, allocation_tot, self.prices))
        elif state_flag == 'bsl_prices_w':
            baseline_features = np.array([round(sensitivities[-1]/sensitivities[-4]), round(sensitivities[-1]/sensitivities[-2])])
            return np.concatenate((baseline_features, allocation_tot, self.prices))
        elif state_flag == 'bsl_prices_dprices':
            baseline_features = np.array([round(sensitivities[-1]/sensitivities[-4]), round(sensitivities[-1]/sensitivities[-2])])
            return np.concatenate((baseline_features, allocation_tot, self.prices, self.daily_prices))
        elif state_flag == 'BTP_baseline':
            return [round(sensitivities[-1]/sensitivities[-4]), allocation_tot[1]]
        elif state_flag == 'BTP-iTraxx_baseline':
            return [allocation_tot[1], round(sensitivities[-1]/sensitivities[-4]), allocation_tot[3], round(sensitivities[-1]/sensitivities[-2])]
        else:
            baseline_sensitivities = [sensitivities[0] / sensitivities[4], sensitivities[1] / sensitivities[4],
                                       sensitivities[3] / sensitivities[4]]
            return np.concatenate((np.array([allocation_tot[0], allocation_tot[1],allocation_tot[3]]), baseline_sensitivities))

    def get_state_dim(self):
        state_flag = self.state_flag
        if state_flag == 'bsl':
            return 2 + self.number_securities
        elif state_flag == 'bsl_prices':
            return 2 + self.number_securities + self.price_dim
        elif state_flag == 'bsl_prices_w':
            return 2 + self.number_securities + self.price_dim
        elif state_flag == 'bsl_prices_dprices':
            return 2 + self.number_securities + self.price_dim + self.daily_price_dim
        elif state_flag == 'BTP_baseline':
            return  2
        elif state_flag == 'BTP-iTraxx_baseline':
            return  4
        else:
            return 6


    def _initial_allocation(self, empty_allocation=1):
        initial_allocation = []
        if empty_allocation == 1:
            self.capital = 0
            for i in range(self.number_securities):
                self.allocation_tot[i] = 0.0
        elif empty_allocation == 2:
            self.capital = self.allocation_tot[-1]
            for i in range(self.number_securities):
                self.allocation_tot[i] = self.Initial_alloc_tot[i]
        else:
            rd= random.random()
            if rd < 0.5:
                initial_allocation.append(random.randint(0, 11500))
                tmp = random.randint(0, 2100)  # btp
                initial_allocation.append(tmp)
                initial_allocation.append(
                    -tmp + (random.randint(round(-tmp / 100), round(tmp / 100))))  # Fix segno bund (negativo)
                initial_allocation.append(random.randint(0, 700))
                initial_allocation.append(random.randint(-5e6, +5e6))
                self.capital = initial_allocation[-1]
                for i in range(self.number_securities):
                    self.allocation_tot[i] = initial_allocation[i]
            else:
                self.capital = 0
                for i in range(self.number_securities):
                    self.allocation_tot[i] = 0.0
    """
    Metodo di inizializzazione dei valori
    """

    def _init_price(self, index):

        self.current_date = datetime.strptime(self.data.ix[index, 'data_column'],
                                              '%d/%m/%Y').date()  # update data attuale
        self.orario = self.data.ix[index, 'orario_column']

        number_securities = self.number_securities
        state_dim = self.state_dim
        self.state = np.zeros((state_dim,))

        self.increment_for_interest = 0.0
        price = self._compute_price(index=index)
        self.prices = price
        sensitivities = self._compute_sensitivity(index=index)
        self.sensitivities = sensitivities

        time_to_roll = self.compute_time_to_roll(index=index)
        self.time_to_roll = time_to_roll

        self.RC = self._compute_RC(price, sensitivities, init=1)

        allocation_delta = np.zeros(number_securities * (self.window_offset + 1))
        self.allocation_delta = allocation_delta

        self._initial_allocation(
            empty_allocation=self.empty_allocation)  # di default self.empty_allocation = 1 (true) e quindi il portafoglio viene lasciato vuoto

        self.step_PL = 0

        price_dim = len(price)

        self.prices = price

        self.daily_prices = self.find_daily_prices(index)
        self.weekly_prices = self.find_weekly_prices(index)
        self.V2X = self.data.ix[index, "V2X_VALUE"]
        self.VIX = self.data.ix[index, "VIX_VALUE"]
        self.DVA = self._compute_dva(index)

        # Aggiornamento stato
        self.state = self.get_state()
        #
        # STAMPE UTILI A CONSOLE
        # -----------------------------------------------------------------------------
        if self.print_output:
            print("------------ STATO INIZIALE ------------\n\ndata attuale: ", self.current_date)
            print("orario:  ", self.orario)
            print("index:  ", self.index)
            #                self.print_pretty()
            print('\nlotti SX7E-BTP-ITRAXX')
            print('delta allocation: {}'.format(self.allocation_delta[-5:-1]))
            print('total allocation: {}'.format(self.allocation_tot))
            print('\nprices : [SX7E, spread BTP/BUND, ITRAXX bid, ITRAXX ask, spread 5Y] ')
            print(self.prices[-self.n_prices:])

            print("")
            print("---------------------------------------------------------------------")
            print("---------------------------------------------------------------------")
        # if self.start_point is None:
        #                print("----------------------------------------------------")
        #                print("----------------- INIZIALIZZAZIONE -----------------")
        #                print("----------------------------------------------------")
        #            self.print_pretty()


        # TEST OUTPUT_DF
        if self.write_to_file:
            self.itraxx_on_spread = self.data.ix[index, "Itraxx_Spread_BID"]  # Aggiunto per output su file
            self.cds_spread_5y = self.data.ix[index, "Spread_cds_5Y_BID"]  # Aggiunto per output su file
            self._compute_output()
            self._write_output_df()
            self.row_index_output += 7
            print(self.output_df)

    """
    Metodo per aggiornare i prezzi degli strumenti
    """

    def _compute_price(self, index):
        price_dim = self.price_dim
        price = np.zeros(price_dim)
        idx = 0
        for i in range(index - self.window_offset, index + 1):  # range: [index - self.window_offset, index]
            cdata = self.data.iloc[i, :]
            # SX7E
            price[idx] = cdata['SX7E_Price']
            idx += 1

            # Spread BTP-BUND
            price[idx] = cdata['spread_BTP_BUND']
            idx += 1
            self.bund_price = cdata['BUND_Price']  # Aggiunta bund_price come attributo di classe
            self.btp_price = cdata['BTP_Price']

            # ITRAXX
            price[idx] = (cdata["Itraxx_onPrice_BID"] + cdata["Itraxx_onPrice_ASK"])/2
            idx += 1

            # CDS
            price[idx] = cdata["Spread_cds_5Y_BID"]
            idx += 1

        return price

    """
    Metodo per aggiornare le sensitivities
    """

    def _compute_sensitivity(self, index):

        sensitivities        = np.zeros(self.number_securities)

        cdata = self.data.iloc[index, :]
        # SX7E - percent sensitivity (per unità di lotto)

        sx7E_sensitivity = cdata['SX7E_Price'] * self.multiplier[0] * 0.01  # *self.allocation_tot[0]
        sensitivities[0] = sx7E_sensitivity

        # BTP - bp sensitivity (per unità di lotto)
        btp_sensitivity = cdata['BTP_Sensitivity'] * 10  # *self.allocation_tot[1]
        sensitivities[1] = btp_sensitivity

        # BUND - bp sensitivity (per unità di lotto)
        bund_sensitivity = cdata['BUND_Sensitivity'] * 10  # *self.allocation_tot[2]
        sensitivities[2] = bund_sensitivity

        # ITRAXX - bp sensitivity (per unità di lotto)
        itraxx_sensitivity = (cdata['Itraxx_Sensitivity_BID'] + cdata['Itraxx_Sensitivity_ASK'])/2 * self.multiplier[2] / 10000  # *self.allocation_tot[3]
        sensitivities[3] = itraxx_sensitivity

        # DVA -bp sensitivity
        dva_sensitivity = cdata['DVA_Sensitivity_BID']
        sensitivities[4] = ((dva_sensitivity * self.dva_nominal) / 10000)

        return sensitivities

    """
    Metodo per calcolare il Regulaory Capital
    """

    def _compute_RC(self, price, sensi, init=1):
        allocation_tot = self.allocation_tot
        multiplier = self.multiplier

        sensitivities_RC = np.zeros(4)
        sensitivities_RC[0] = allocation_tot[0] * sensi[-5] * 100  # da percent sensitivity a sensitivity unitaria
        sensitivities_RC[1] = allocation_tot[1] * sensi[-4] * multiplier[1] *10          # da bp sensitivity a sensitivity unitaria
        sensitivities_RC[2] = allocation_tot[2] * sensi[-3] * multiplier[1] *10         # da bp sensitivity a sensitivity unitaria
        sensitivities_RC[3] = allocation_tot[3] * sensi[-2] * 10000 * (-1)  # da bp sensitivity a sensitivity unitaria

        SX7E_price =   price[-4]
        ITRAXX_price = price[-2]
        prices = np.array([SX7E_price, self.btp_price, self.bund_price, ITRAXX_price])


        # Delta ---------------------------------------------------------------
        # Risk weights
        RW = np.array([0.5, 0.005, 0.005, 0.05])
        # Risk position for each bucket
        # (Equity bucket 8, CSR bucket 1, CSR bucket 1, CSR bucket 3)
        WS_sx7e = RW[0] * sensitivities_RC[0]
        WS_btp = RW[1] * sensitivities_RC[1]
        WS_bund = RW[2] * sensitivities_RC[2]
        WS_iTraxx = RW[3] * sensitivities_RC[3]

        # Low-Medium-High correlation scenarios
        rho_stress_test = [0.35 * 0.75, 0.35, 0.35 * 1.25]
        gamma_stress_test = [0.1 * 0.75, 0.1, 0.1 * 1.25]

        K = np.zeros(3)
        K[0] = np.sqrt(WS_sx7e ** 2)
        Delta_max = 0
        for i in range(len(rho_stress_test)):
            rho = rho_stress_test[i]
            gamma = gamma_stress_test[i]
            K[1] = np.sqrt(WS_btp ** 2 + WS_bund ** 2 + 2 * rho * WS_btp * WS_bund)
            K[2] = np.sqrt(WS_iTraxx ** 2)

            if K[1] ** 2 + K[2] ** 2 + 2 * gamma * (WS_btp + WS_bund) * WS_iTraxx >= 0:
                Delta = K[0] + np.sqrt(np.max(K[1] ** 2 + K[2] ** 2 + 2 * gamma * (WS_btp + WS_bund) * WS_iTraxx, 0))
            else:
                S1 = max(min(WS_btp + WS_bund, K[1]), -K[1])
                S2 = max(min(WS_iTraxx, K[2]), -K[2])
                Delta = K[0] + np.sqrt((K[1] ** 2 + K[2] ** 2) + 2 * gamma * S1 * S2)
            Delta_max = max(Delta_max, Delta)

            # DRC -----------------------------------------------------------------
        conv_fact_btp = 0.6
        conv_fact_bund = 0.9
        JTD_sx7e = 1 * SX7E_price * allocation_tot[0] * multiplier[0] / 4
        JTD_btp = 0.75 * conv_fact_btp * allocation_tot[1] * multiplier[1]
        JTD_bund = 0.75 * conv_fact_bund * allocation_tot[2] * multiplier[1]
        JTD_iTraxx = 0.75 * allocation_tot[3] * multiplier[2] + prices[3] * allocation_tot[3] * multiplier[2]

        # print(JTD_btp, JTD_bund, JTD_iTraxx)
        # Risk weights
        RW = np.array([0.045, 0.06, 0.005, 0.045])
        DRC_corporates = np.max(RW[3] * JTD_iTraxx + RW[0] * JTD_sx7e, 0)
        if abs(JTD_btp + abs(JTD_bund)) > 0:
            WtS = JTD_btp / (JTD_btp + abs(JTD_bund))
        else:
            WtS = 0
        DRC_sovereigns = RW[1] * JTD_btp - WtS * RW[2] * abs(JTD_bund)
        DRC = DRC_corporates + DRC_sovereigns
        # Regulatory Capital --------------------------------------------------
        # Per ora considero solo l'RC dovuto alle sensitivities (calcolo del JTD da ricontrollare)
        #RC = Delta_max  # + DRC

        RC = 0

        return RC

    """
    Metodo per aggiornare il time to roll degli strumenti
    """

    def compute_time_to_roll(self, index):
        number_timetoroll = (self.number_securities) * (self.window_offset + 1)
        time_to_roll = np.zeros(number_timetoroll)
        idx = 0
        for i in range(index - self.window_offset, index + 1):  # range: [index - self.window_offset, index]
            cdata = self.data.iloc[i, :]
            # SX7E
            sx7E_ttr = cdata['SX7E_Time_To_Roll']
            time_to_roll[idx] = sx7E_ttr
            idx += 1
            # BTP
            btp_ttr = cdata['BTP_Time_To_Roll']
            time_to_roll[idx] = btp_ttr
            idx += 1
            # BUND
            bund_ttr = cdata['BUND_Time_To_Roll']
            time_to_roll[idx] = bund_ttr
            idx += 1
            # ITRAXX
            itraxx_ttr = cdata['ITRAXX_Time_To_Roll']
            time_to_roll[idx] = itraxx_ttr
            idx += 1
            # CDS
            dva_ttr = cdata['CDS_Time_To_Roll']
            time_to_roll[idx] = dva_ttr
            idx += 1
        return time_to_roll

    """
    Metodo per calcolare il collaterale subito dopo che l'azione è stata presa (i-1)^+
    """

    def _compute_collat(self, index):
        ITRAXX_notional = self.allocation_tot[3] * self.multiplier[2]
        mid_price = (self.data.ix[index, "Itraxx_onPrice_ASK"] + self.data.ix[index, "Itraxx_onPrice_BID"]) / 2
        return ITRAXX_notional * mid_price

    """
    Metodo per calcolare il dva a partire dal CDS spread 5Y
    """

    def _compute_dva(self, index):
        lambda_CDS = self.data.ix[index, "Spread_cds_5Y_BID"] / 0.6 / 100
        delta_days_CDS = datetime.strptime(self.data.ix[index, 'maturity ISPIM'], '%d/%m/%Y') - datetime.strptime(
            self.data.ix[index, 'data_column'], '%d/%m/%Y')
        return 0.6 * (1 - np.exp(-lambda_CDS * delta_days_CDS.days / 360)) * self.dva_nominal

    """
    Metodo per calcolare il dva dividend a partire dal CDS spread 5Y
    """

    def _compute_dva_dividend(self, index):
        lambda_CDS = self.data.ix[index - 1, "Spread_cds_5Y_BID"] / 0.6 / 100
        delta_days_CDS = datetime.strptime(self.data.ix[index - 1, 'maturity ISPIM'], '%d/%m/%Y') - datetime.strptime(
            self.data.ix[index, 'data_column'], '%d/%m/%Y')
        return 0.6 * (1 - np.exp(-lambda_CDS * delta_days_CDS.days / 360)) * self.dva_nominal - self._compute_dva(index-1)

    def _compute_delta_iTraxx(self, index):
        multiplier = self.multiplier
        allocation_delta = self.allocation_delta
        tmp_alloc_tot = self.tmp_alloc_tot

        iTraxx_old = (self.data.ix[index - 1, "Itraxx_onPrice_ASK"] + self.data.ix[index - 1, "Itraxx_onPrice_BID"]) / 2
        iTraxx = (self.data.ix[index, "Itraxx_onPrice_ASK"] + self.data.ix[index, "Itraxx_onPrice_BID"]) / 2

        if allocation_delta[-2] > 0:
            delta_iTraxx = tmp_alloc_tot[2] * multiplier[2] * (iTraxx - iTraxx_old) \
                           + allocation_delta[-2] * multiplier[2] * (iTraxx - self.data.ix[index - 1, "Itraxx_onPrice_BID"])
        else:
            delta_iTraxx = + tmp_alloc_tot[2] * multiplier[2] * (iTraxx - iTraxx_old) \
                           + allocation_delta[-2] * multiplier[2] * (iTraxx - self.data.ix[index - 1, "Itraxx_onPrice_ASK"])

        return delta_iTraxx

    """
    Metodo che calcola la funzione Reward
    """

    def f_reward(self, x):
        if x > -0.1 or x < -0.5*(10**6):
            return x
        else:
            return max(1 - (1 - x) ** self.risk_factor, -0.5*(10**6))
    #
    # def f_reward(self, x):
    #     if x > -0.1:
    #         return x
    #     else:
    #         return 1 - (1 - x) ** self.risk_factor
    """
    Metodo che restituisce l'azione della Baseline bpt, iTraxx, btp-iTraxx
    """

    def baseline_action(self, flag="BTP_baseline"):
        action = np.zeros(3)  # [sx7e, btp, iTraxx]

        sensitivity_dva = self.sensitivities[-1]
        sensitivity_btp = self.sensitivities[-4]
        sensitivity_iTraxx = self.sensitivities[-2]

        if flag == "BTP_iTraxx_baseline":
            btp = round(-0.5 * sensitivity_dva / sensitivity_btp)  # long btp
            delta_btp = btp - self.allocation_tot[1]
            iTraxx = round(-1* sensitivity_dva / sensitivity_iTraxx)  # long risk
            delta_iTraxx = iTraxx - self.allocation_tot[3]
            action[1] = delta_btp
            action[2] = delta_iTraxx

        elif flag == "BTP_baseline":
            btp = round(-sensitivity_dva / sensitivity_btp)  # long btp
            delta_btp = btp - self.allocation_tot[1]
            action[1] = delta_btp

        elif flag == "iTraxx_baseline":
            iTraxx = round(-2*sensitivity_dva / sensitivity_iTraxx)  # long risk
            delta_iTraxx = iTraxx - self.allocation_tot[3]
            action[2] = delta_iTraxx
        elif flag == "DVA":
            return action # don't do nothing
        else:
            print("error : please select sx7e-btp-iTraxx-naked strategy")
        return action

    """
    Metodo per calcolare daily_prices
    """

    def find_daily_prices(self, index):
        if self.window_offset_day == 0:
            return []
        stop = 0
        closing_index = index
        daily_price = np.zeros(self.window_offset_day * self.n_prices)
        while stop == 0:
            if self.data.ix[closing_index, 'orario_column'] == "17:25":
                stop = 1
            else:
                closing_index = closing_index - 1

        # gestire il caso in cui mi trovo a fine giornata
        if closing_index == index:
            closing_index = closing_index - self.n_step_day
        idx = 0

        for i in np.arange(self.window_offset_day - 1, -1, -1):
            cdata = self.data.iloc[closing_index - i * self.n_step_day, :]
            # SX7E
            daily_price[idx] = cdata['SX7E_Price']
            idx += 1
            # Spread BTP-BUND
            daily_price[idx] = cdata['spread_BTP_BUND']
            idx += 1
            # ITRAXX
            daily_price[idx] = (cdata["Itraxx_onPrice_BID"] + cdata["Itraxx_onPrice_ASK"])/2
            idx += 1
            # CDS
            daily_price[idx] = cdata["Spread_cds_5Y_BID"]
            idx += 1


        return daily_price

    """
    Metodo per calcolare weekly_prices
    """

    def find_weekly_prices(self, index):
        if self.window_offset_week == 0:
            return []
        stop = 0
        closing_index = index
        n_prices = self.n_prices
        weekly_price = np.zeros(self.window_offset_week * n_prices)

        while stop == 0:
            if self.data.ix[closing_index, 'orario_column'] == "17:25":
                stop = 2
            else:
                closing_index = closing_index - 1

        # gestire il caso in cui mi trovo a fine giornata
        if closing_index == index:
            closing_index = closing_index - self.n_step_day

        idx = 0
        for i in np.arange(self.window_offset_week, 0, -1):
            closing_prices = np.zeros((n_prices, 5))

            for j in np.arange(5):
                cdata = self.data.iloc[closing_index + (j + 1 - i * 5) * self.n_step_day, :]
                # SX7E
                closing_prices[0][j] = cdata['SX7E_Price']
                # Spread BTP-BUND
                closing_prices[1][j] = cdata['spread_BTP_BUND']
                # ITRAXX
                closing_prices[2][j] = (cdata["Itraxx_onPrice_BID"] + cdata["Itraxx_onPrice_ASK"])/2
                # CDS
                closing_prices[3][j] = cdata["Spread_cds_5Y_BID"]

            weekly_price[idx:idx + n_prices] = [closing_prices[k][:].mean() for k in range(n_prices)]
            idx = idx + n_prices

        return weekly_price

