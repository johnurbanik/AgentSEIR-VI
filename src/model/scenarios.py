from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import pickle

class BaseScenarioParams:
    '''
    Basic disease parameters taken from
    '''

    # From https://mrc-ide.github.io/global-lmic-reports/parameters.html
    hosp_by_age = pd.Series({
        14: 0.1,
        19: 0.2,
        24: 0.5,
        29: 1.0,
        34: 1.6,
        39: 2.3,
        44: 2.9,
        49: 3.9,
        54: 5.8,
        59: 7.2,
        64: 10.2,
        69: 11.7,
        74: 14.6,
        79: 17.7,
        np.inf: 18.0
    }) / 100

    hosp_death_by_age = pd.Series({
        39: 1.3,
        44: 1.5,
        49: 1.9,
        54: 2.7,
        59: 4.2,
        64: 6.9,
        69: 10.5,
        74: 14.9,
        79: 20.3,
        np.inf: 58.0
    }) / 100

    age

    # R0 in NY as high as 4, but in China it was around 3.2
    # Source: https://covid19-projections.com/infections-tracker/, among others
    R0 = 3.2

    '''
    Incubation/infectious time variously taken from several sources:
    https://cmmid.github.io/topics/covid19/pre-symptomatic-transmission.html
    https://github.com/HopkinsIDD/ncov_incubation
    https://github.com/dirkschumacher/covid-19-indicators
    https://www.medrxiv.org/content/10.1101/2020.04.25.20079889v1
    '''
    # NB: Actually comprises both exposed and presymptomatic, but useful to divide this way as agent symptoms will change awareness.
    INCUBATION_PERIOD = (6.47, 0.71)  # Gamma distribution scale and shape. Mean 4.58.
    PROPORTION_PRESYMPTOMATIC_TRANSMISSION = (.45, .2, 0, 1) # Truncated normal on range (0,1) with std dev of .2. May actually be more like a mixture?
    SYMPTOMATIC_RATE = 0.6  # rate of people that have the virus and will manifest symptoms

    # TODO: These are really age dependent!
    PRESYMPTOMATIC_CONTAGIOUS_PERIOD = (4/3, 5/4) # Gamma distribution. Mean 2, SD 1.74.
    ASYMPTOMATIC_CONTAGIOUS_PERIOD = (8.45, 0.77)  # Gamma distribution. Mean 6.5
    SYMPTOMATIC_CONTAGIOUS_PERIOD = (9.8, 0.7)  # Gamma distribution. Mean 7.
    # RECOVERY_PERIOD = ()  # TODO: Account for # longcovid, instead of considering agents recovered immediately after they

    SYMPTOM_TO_HOSP_PERIOD = (5, 2, 0, 20)  # TODO: Truncated normal, should probably be gamma distributed instead.
    HOSP_DEATH_PERIOD = 9.5
    HOSP_RECOVERY_PERIOD = 7.6

    def __init__(
        self,
    ):
        pass
