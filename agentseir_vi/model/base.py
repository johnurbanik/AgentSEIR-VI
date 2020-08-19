import json

import mesa
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector

import jax.numpy as np
import jax.pandas as pd

from agentseir_vi.model.utils import log_to_file

class Epidemic(BaseScheduler):
    '''
    A scheduler that steps through time for the agents of a model.

    As opposed to modeling individual agents, as in standard mesa models, agents and their properties are
    kept in a set of numpy arrays. This allows for use of numpyro and drastically improves the performance.

    Thus, a scheduler over a model is used, and agent actions are controlled directly.
    '''
    def __init__(self, model):
        # For now, just use super.
        # If we want to change how time is stepped through (i.e. different phases)
        # this would be where to do it.
        super().__init__(model)

    def step(self):
        self.update_agent_states()  # Update the counts of the various states
        if self.model.log_file:
            self.model.write_logs()  # Write some logs, if enabled

        # NB: At some point, it may make sense to make agents move. For now, they're static.

        ############################# Transmission logic ###################################
        # Get all of the newly contagious agents, swap them to being contagious
        contagious = np.where(
            (self.model.contagious_start < self.time) &
            (self.model.epidemic_state < self.model.STATE_CONTAGIOUS)
        )

class Population(Model):
    '''
    A population is a square grid of agents.
    For now, they stay in the same place, and infect other agents around them
    based on their risk tolerance, social radius, and their neighbors hygiene.

    Additionally, agents modify their social radius based on sampling their neighborhood and their awareness.
    '''
    STATE_SUSCEPTIBLE = 0
    STATE_EXPOSED = 1
    STATE_PRESYMPTOMATIC = 2
    STATE_SYMPTOMATIC = 3
    STATE_ASYMPTOMATIC = 4
    STATE_HOSPITALIZED = 5
    STATE_DEAD = 6
    STATE_RECOVERED = 7

    def __init__(self, params, log_file=None):
        '''
        params: class or dict containing the global parameters for the model.
        '''
        self.log_file = log_file
        self.params = params

        # Agents
        # self.people = None  # ID array for easy reference
        self.age = None  # Float 0-100
        self.sex = None  # Boolean, since we don't have good data for intersex people
        self.risk_tolerance = None # Float, 0-1
        self.risk_factors = None # Integer 0-5, number of co-morbidities
        self.hygiene = None  # Float, 0-1
        '''
        0 -> can stay home
        1 -> private commuter
        2 -> public transit commuter
        3 -> essential worker
        '''
        self.worker_class = None
        '''
            0-> susceptible, 1-> exposed, 2-> presymptomatic,
            3 -> symptomatic, 4-> asymptomatic, 5 -> hospitalized, 6 -> dead, 7 -> recovered
        '''
        self.epidemic_state = None
        # self.location = None  # Position on grid
        self.social_radius = None  # Int 0-10, how far out grid do they socialize with
        self.awareness = None  # How much attention they are paying to growth of epidemic locally

        # Dates are all ints for the step number
        self.date_infected = None
        self.date_contagious = None
        self.date_recovered = None
        self.date_hospitalized = None
        self.date_died = None


        # Global params:
        self.lockdown_level = 0.0  # Float 0-1

        # Counters
        self.infected_count = 0
        self.presymptomatic_count = 0
        self.asymptomatic_count = 0
        self.symptomatic_count = 0
        self.hospitalized_count = 0
        self.recovered_count = 0
        self.died_count = 0

        self.scheduler = Epidemic(self)


    def init_agents(self, pop_size=(1e3, 1e3)):
        # Numpyro code will go here, passed in via self.params
        # For a first pass, we'll initialize the numpy arrays.

        self.age = np.random.uniform(0, 100, size=pop_size)  # Terrible distribution, we should use random.choice from a discrete age distribution
        self.sex = np.random.binomial(1, .5, size=pop_size)
        self.risk_tolerance = np.random.beta(2,5, size=pop_size)  # Random values for now
        self.risk_factors = np.random.binomial(5, .3, size=pop_size)
        self.hygiene = np.random.beta(2,5, size=pop_size)  # Random values for now
        self.worker_class = np.random.choice(3, size=pop_size, p=(.6, .1, .2, .1))


        self.epidemic_state = np.zeros(shape=pop_size)
        self.social_radius = np.random.binomial(10, .2, size=pop_size)
        self.awareness =  np.random.beta(2, 2, size=pop_size)

        self.date_infected = np.full(shape=pop_size, fill_value=np.inf)
        self.date_contagious = np.full(shape=pop_size, fill_value=np.inf)
        self.date_recovered = np.full(shape=pop_size, fill_value=np.inf)
        self.date_hospitalized = np.full(shape=pop_size, fill_value=np.inf)
        self.date_died = np.full(shape=pop_size, fill_value=np.inf)

    def step(self):
        self.scheduler.step()

    def set_status(self, neighbors_to_infect):
        pass

    def write_logs(self):
        current_time =  self.scheduler.time
        data = dict(
            time=current_time,

            current_exposed_cases=int((self.epidemic_state == self.STATE_EXPOSED).sum()),
            current_contagious_cases=int(np.where(np.logical_and(
                self.epidemic_state >= self.STATE_PRESYMPTOMATIC,
                self.epidemic_state <= self.STATE_HOSPITALIZED
            )).sum()),
            current_hospitalized_cases=int((self.epidemic_state == self.STATE_HOSPITALIZED).sum()),

            presymptomatic_count=int((self.epidemic_state == self.STATE_PRESYMPTOMATIC).sum()),
            asymptomatic_count=int((self.epidemic_state == self.STATE_ASYMPTOMATIC).sum()),
            symptomatic_count=int((self.epidemic_state == self.STATE_SYMPTOMATIC).sum()),

            total_contagious_count=int((self.date_contagious < current_time).sum()),
            total_infected_count=int((self.date_infected < current_time).sum()),
            total_hospitalized_count=int((self.date_hospitalized < current_time).sum()),
            total_died_count=int((self.date_died < current_time).sum()),
            total_recovered_count=int((self.date_recovered < current_time).sum()),
        )
        info = json.dumps(data)
        log_to_file(self.log_file, info, as_log=True, verbose=False)