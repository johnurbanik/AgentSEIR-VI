import jax.numpy as np
import numpyro.distributions as dist

class BaseScenarioParams:
    '''
    Basic disease parameters.
    '''

    # From https://mrc-ide.github.io/global-lmic-reports/parameters.html
    HOSP_AGE_BRACKETS = np.array([14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 79, 74, 79, 100])
    HOSP_BY_BRACKET = np.array([0.1, 0.2, 0.5, 1.0, 1.6, 2.3, 2.9, 3.9, 7.2, 10.2, 11.7, 14.6, 17.7, 18.0]) / 100
    DEATH_AGE_BRACKETS = np.array(39, 44, 49, 54, 59, 64, 69, 74, 79)
    HOSP_DEATH_BY_BRACKET = np.array(1.3, 1.5, 1.9, 2.7, 4.2, 6.9, 10.5, 14.9, 20.3, 58.0) / 100
    HOSP_BY_AGE = {
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
        100: 18.0
    }

    HOSP_DEATH_BY_AGE = {
        39: 1.3,
        44: 1.5,
        49: 1.9,
        54: 2.7,
        59: 4.2,
        64: 6.9,
        69: 10.5,
        74: 14.9,
        79: 20.3,
        100: 58.0
    }

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
    INCUBATION_PERIOD = (6.47, 1.41)  # Gamma distribution scale and rate. Mean 4.58.
    PROPORTION_PRESYMPTOMATIC_TRANSMISSION = (.45, .2, 0, 1) # Truncated normal on range (0,1) with std dev of .2. May actually be more like a mixture?
    SYMPTOMATIC = (1, 0.6)  # Binomial. Rate of people that have the virus and will manifest symptoms

    # TODO: These are really age dependent!
    PRESYMPTOMATIC_CONTAGIOUS_PERIOD = (4/3, .8) # Gamma distribution shape and rate. Mean 2, SD 1.74.
    ASYMPTOMATIC_CONTAGIOUS_PERIOD = (8.45, 1.3)  # Gamma distribution shape and rate. Mean 6.5
    SYMPTOMATIC_CONTAGIOUS_PERIOD = (9.8, 1.43)  # Gamma distribution shape and rate. Mean 7.
    # RECOVERY_PERIOD = ()  # TODO: Account for # longcovid, instead of considering agents recovered immediately after they are no longer infectious.

    SYMPTOM_TO_HOSP_PERIOD = (5, 2, 0, 20)  # TODO: Truncated normal, should probably be gamma distributed instead.
    HOSP_DEATH_PERIOD = (9.5, 1)  # TODO: better gamma distribution
    HOSP_RECOVERY_PERIOD = (7.6, 1)  # TODO: better gamma distribution

    def __init__(self, params):
        '''
        Pull from passed parameters and initialize numpyro samplers.
        '''
        # if params.R0 is not None:
        #     self.R0 = params.R0
        # self.R0 = numpyro.sample('r0', dist.Normal(*self.R0))  # NB: Unused right now

        # if params.INCUBATION_PERIOD is not None:
        #     self.INCUBATION_PERIOD = params.INCUBATION_PERIOD
        # self.INCUBATION_PERIOD = numpyro.sample('iP', dist.Gamma(*self.INCUBATION_PERIOD))  # NB: unused right now
        if params.PROPORTION_PRESYMPTOMATIC_TRANSMISSION is not None:
            self.PROPORTION_PRESYMPTOMATIC_TRANSMISSION = params.PROPORTION_PRESYMPTOMATIC_TRANSMISSION
        self.PROPORTION_PRESYMPTOMATIC_TRANSMISSION = numpyro.sample('pP', dist.TruncatedNormal(*self.PROPORTION_PRESYMPTOMATIC_TRANSMISSION))
        if params.SYMPTOMATIC is not None:
            self.SYMPTOMATIC = params.SYMPTOMATIC
        self.SYMPTOMATIC = numpyro.sample('rS', dist.Beta(*self.SYMPTOMATIC))

        if params.PRESYMPTOMATIC_CONTAGIOUS_PERIOD is not None:
            self.PRESYMPTOMATIC_CONTAGIOUS_PERIOD = params.PRESYMPTOMATIC_CONTAGIOUS_PERIOD
        self.PRESYMPTOMATIC_CONTAGIOUS_PERIOD = numpyro.sample('cP', dist.Gamma(*self.PRESYMPTOMATIC_CONTAGIOUS_PERIOD))
        if params.ASYMPTOMATIC_CONTAGIOUS_PERIOD is not None:
            self.ASYMPTOMATIC_CONTAGIOUS_PERIOD = params.ASYMPTOMATIC_CONTAGIOUS_PERIOD
        self.ASYMPTOMATIC_CONTAGIOUS_PERIOD = numpyro.sample('cA', dist.Gamma(*self.ASYMPTOMATIC_CONTAGIOUS_PERIOD))
        if params.SYMPTOMATIC_CONTAGIOUS_PERIOD is not None:
            self.SYMPTOMATIC_CONTAGIOUS_PERIOD  = params.SYMPTOMATIC_CONTAGIOUS_PERIOD
        self.SYMPTOMATIC_CONTAGIOUS_PERIOD = numpyro.sample('cS', dist.Gamma(*self.SYMPTOMATIC_CONTAGIOUS_PERIOD))

        if params.SYMPTOM_TO_HOSP_PERIOD is not None:
            self.SYMPTOM_TO_HOSP_PERIOD = params.SYMPTOM_TO_HOSP_PERIOD
        self.SYMPTOM_TO_HOSP_PERIOD = numpyro.sample('pH', dist.TruncatedNormal(*self.SYMPTOM_TO_HOSP_PERIOD))
        if params.HOSP_DEATH_PERIOD is not None:
            self.HOSP_DEATH_PERIOD = params.HOSP_DEATH_PERIOD
        self.HOSP_DEATH_PERIOD = numpyro.sample('hD', dist.Gamma(*self.HOSP_DEATH_PERIOD))
        if params.HOSP_RECOVERY_PERIOD is not None:
            self.HOSP_RECOVERY_PERIOD  = params.HOSP_RECOVERY_PERIOD
        self.HOSP_RECOVERY_PERIOD = numpyro.sample('hR', dist.Gamma(*self.HOSP_RECOVERY_PERIOD))

        # Derived variables
        self.EXPOSED_PERIOD = self.INCUBATION_PERIOD - self.PRESYMPTOMATIC_CONTAGIOUS_PERIOD
        self.RECOVERY_PERIOD = self.INCUBATION_PERIOD + np.where(self.SYMPTOMATIC > 0, self.SYMPTOMATIC_CONTAGIOUS_PERIOD, self.ASYMPTOMATIC_CONTAGIOUS_PERIOD)
