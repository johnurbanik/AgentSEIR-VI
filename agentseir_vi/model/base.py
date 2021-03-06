import json

import mesa
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector

import jax.numpy as np
import jax.random as jrand

import numpyro as ny
import numpyro.distributions as dist

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
        self.PRNGKey = jrand.PRNGKey(0)  # TODO: Better seed
        self.params = self.model.params
        super().__init__(model)

    def step(self):
        self.update_agent_states()  # Update the counts of the various states
        self.update_counts()
        if self.model.log_file:
            self.model.write_logs()  # Write some logs, if enabled

        # NB: At some point, it may make sense to make agents move. For now, they're static.

        ############################# Transmission logic ###################################
        # Get all the currently contagious agents, and have them infect new agents.
        # TODO: include hospital transmission, vary transmissability by state.
        contagious = np.asarray(
            (self.model.epidemic_state == self.model.STATE_PRESYMPTOMATIC) |
            (self.model.epidemic_state == self.model.STATE_ASYMPTOMATIC) |
            (self.model.epidemic_state == self.model.STATE_SYMPTOMATIC)
        ).nonzero()

        # For each contagious person, infect some of its neighbors based on their hygiene and the contagious person's social radius.
        # Use jax.random instead of numpyro here to keep these deterministic.
        # TODO: figure out a way to do this in a (more) vectorized manner. Probably some sort of kernel convolution method with each radius. Should also look into numpyro's scan.
        for x,y in zip(*contagious):
            radius = self.model.social_radius[x, y]
            base_isolation = self.model.base_isolation[x, y]
            nx, ny = np.meshgrid(   # pylint: disable=unbalanced-tuple-unpacking
                np.arange(x - radius, x + radius),
                np.arange(y - radius, y + radius)
            )
            neighbor_inds = np.vstack([nx.ravel(), ny.ravel()])
            # Higher base_isolation leads to less infection.
            # TODO: modify isolation so that symptomatic agents isolate more.
            infection_attempts = jrand.choice(
                self.PRNGKey,
                neighbor_inds,
                shape=int(len(neighbor_inds) * (1 - base_isolation))
            )
            potentially_infected_hygiene = self.model.hygiene[infection_attempts[:, 0], infection_attempts[:, 1]]
            susceptible = self.model.epidemic_state[infection_attempts[:, 0], infection_attempts[:, 1]] == self.model.STATE_SUSCEPTIBLE

            indexer = jrand.bernoulli(
                self.PRNGKey,
                potentially_infected_hygiene.ravel(),
                len(infection_attempts)
            )
            got_infected = np.zeros(self.model.epidemic_state.shape, dtype=np.bool_)
            got_infected[potentially_infected_hygiene[indexer]] = True

            # Set the date to become infectious
            self.model.epidemic_state[got_infected & susceptible] = self.model.STATE_EXPOSED
            self.model.date_infected[got_infected & susceptible] = self.time
            self.model.date_contagious[got_infected & susceptible] = self.time + self.params.EXPOSED_PERIOD

    def update_agent_states(self):
        # Get all of the newly contagious agents, swap them to being contagious
        newly_contagious = np.asarray(
            (self.model.date_contagious <= self.time) &
            (self.model.epidemic_state < self.model.STATE_PRESYMPTOMATIC)
        ).nonzero()
        self.model.epidemic_state[newly_contagious] = self.model.STATE_PRESYMPTOMATIC

        # Also set the time in which they will become symptomatic, recover, die, etc.
        # This could also be done at transmission time.
        self.model.date_symptomatic[newly_contagious] = self.time + self.params.INCUBATION_PERIOD
        self.model.date_recovered[newly_contagious] = self.time + self.params.RECOVERY_PERIOD
        self.model.date_hospitalized[newly_contagious] = self.time + self.params.INCUBATION_PERIOD + self.params.SYMPTOM_TO_HOSP_PERIOD
        self.model.date_died[newly_contagious] = self.time + self.params.RECOVERY_PERIOD + self.params.SYMPTOM_TO_HOSP_PERIOD + self.params.HOSP_DEATH_PERIOD

        # Progress presymptomatic to asymptomatic/symptomatic
        newly_asymptomatic = np.asarray(
            (self.model.epidemic_state == self.model.STATE_PRESYMPTOMATIC) &
            (self.model.date_symptomatic <= self.time) &
            ~(self.params.SYMPTOMATIC)
        ).nonzero()
        self.model.epidemic_state[newly_asymptomatic] = self.model.STATE_ASYMPTOMATIC

        newly_symptomatic = np.asarray(
            (self.model.epidemic_state == self.model.STATE_PRESYMPTOMATIC) &
            (self.model.date_symptomatic <= self.time) &
            ~(self.params.SYMPTOMATIC)
        ).nonzero()
        self.model.epidemic_state[newly_symptomatic] = self.model.STATE_SYMPTOMATIC

        # Progress symptomatic to hospital
        newly_hospitalized = np.asarray(
            (self.model.will_be_hospitalized) &
            (self.model.date_hospitalied <= self.time) &
            (self.model.epidemic_state == self.model.STATE_SYMPTOMATIC)
        ).nonzero()
        self.model.epidemic_state[newly_hospitalized] = self.model.STATE_HOSPITALIZED

        # Progress hospitalized to death
        newly_dead = np.asarray(
            (self.model.epidemic_state == self.model.STATE_HOSPITALIZED) &
            (self.model.will_die) &
            (self.model.date_died <= self.time)
        ).nonzero()
        self.model.epidemic_state[newly_dead] = self.model.STATE_DEAD

        # Progress recovered to recovered if they won't die
        newly_recovered = np.asarray(
            ~(self.model.will_die) &
            (self.model.date_recovered <= self.time) &
            (self.model.epidemic_state < self.model.STATE_RECOVERED)  # Small optimization?
        ).nonzero()
        self.model.epidemic_state[newly_recovered] = self.model.STATE_RECOVERED

    def update_counts(self):
        self.model.susceptible_count[self.time] = (self.model.epidemic_state == self.model.STATE_SUSCEPTIBLE).sum()
        self.model.exposed_count[self.time] = (self.model.epidemic_state == self.model.STATE_EXPOSED).sum()
        self.model.presymptomatic_count[self.time] = (self.model.epidemic_state == self.model.STATE_PRESYMPTOMATIC).sum()
        self.model.asymptomatic_count[self.time] = (self.model.epidemic_state == self.model.STATE_ASYMPTOMATIC).sum()
        self.model.symptomatic_count[self.time] = (self.model.epidemic_state == self.model.STATE_SYMPTOMATIC).sum()
        self.model.hospitalized_count[self.time] = (self.model.epidemic_state == self.model.STATE_HOSPITALIZED).sum()
        self.model.infected_count[self.time] = self.model.presymptomatic_count[self.time] + self.model.asymptomatic_count[self.time] + self.model.symptomatic_count[self.time] + self.model.hospitalized_count[self.time]

        self.model.recovered_count[self.time] = (self.model.epidemic_state == self.model.STATE_RECOVERED).sum()
        self.model.died_count[self.time] = (self.model.epidemic_state == self.model.STATE_DEAD).sum()


class Population(Model):
    '''
    A population is a square grid of agents.
    For now, they stay in the same place, and infect other agents around them
    based on their risk tolerance, social radius, and their neighbors hygiene.

    Additionally, agents modify their social radius based on sampling their neighborhood and their base_isolation.
    '''
    STATE_SUSCEPTIBLE = 0
    STATE_EXPOSED = 1
    STATE_PRESYMPTOMATIC = 2
    STATE_SYMPTOMATIC = 3
    STATE_ASYMPTOMATIC = 4
    STATE_HOSPITALIZED = 5
    STATE_DEAD = 6
    STATE_RECOVERED = 7

    def __init__(self, N, params, log_file=None):
        '''
        params: class or dict containing the global parameters for the model.
        '''
        self.log_file = log_file
        self.params = params

        self.num_steps = N

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
        self.base_isolation = None  # How much attention they are paying to growth of epidemic locally

        # Dates are all ints for the step number
        self.date_infected = None
        self.date_contagious = None
        self.date_recovered = None
        self.date_hospitalized = None
        self.date_died = None


        # Global params:
        # self.lockdown_level = 0.0  # Float 0-1

        # Counters
        self.susceptible_count = np.zeros(shape=self.num_steps, dtype=np.int_)
        self.exposed_count = np.zeros(shape=self.num_steps, dtype=np.int_)
        self.presymptomatic_count = np.zeros(shape=self.num_steps, dtype=np.int_)
        self.asymptomatic_count = np.zeros(shape=self.num_steps, dtype=np.int_)
        self.symptomatic_count = np.zeros(shape=self.num_steps, dtype=np.int_)
        self.hospitalized_count = np.zeros(shape=self.num_steps, dtype=np.int_)
        self.infected_count = np.zeros(shape=self.num_steps, dtype=np.int_)
        self.recovered_count = np.zeros(shape=self.num_steps, dtype=np.int_)
        self.died_count = np.zeros(shape=self.num_steps, dtype=np.int_)

        self.scheduler = Epidemic(self)


    def init_agents(self, num_steps=1000, pop_size=(1e2, 1e2), initial_infections=2):
        self.age = ny.sample('age', dist.Uniform(0, 100))
        self.sex =  ny.sample('sex', dist.Binomial(1, .5))
        self.risk_tolerance = ny.sample('risk', dist.Beta(2, 5))
        # self.risk_factors = ny.sample('health', dist.Binomial(5, .3))
        self.hygiene = ny.sample('hygiene', dist.Beta(2, 5))
        # self.worker_type = ny.sample('worker_type', dist.Categorical((.6, .1, .2, .1)))

        self.epidemic_state = ny.sample('state', dist.Binomial(1, initial_infections/pop_size[0]*pop_size[1]))
        self.social_radius = ny.sample('radius', dist.Binomial(10, .2))
        self.base_isolation = ny.sample('base_isolation', dist.Beta(2, 2))

        # TODO: make these depend on risk factors as well
        self.will_be_hospitalized = ny.sample('hosp', dist.Binomial(1, self.params.HOSP_AGE_MAP[self.age]))
        self.will_die = ny.sample('die', dist.Binomial(1, self.params.DEATH_MAP[self.age]))

        # The lengths of the infection are handled on a per agent basis via scenarios, these are just placeholders.
        self.date_infected = np.where(self.epidemic_state > 0, np.zeros(shape=pop_size), np.full(shape=pop_size, fill_value=np.inf))

        self.date_contagious = np.where(self.epidemic_state > 0, np.ceil(self.params.EXPOSED_PERIOD), np.full(shape=pop_size, fill_value=np.inf))
        self.date_symptomatic = np.full(shape=pop_size, fill_value=np.inf)
        self.date_recovered = np.full(shape=pop_size, fill_value=np.inf)
        self.date_hospitalized = np.full(shape=pop_size, fill_value=np.inf)
        self.date_died = np.full(shape=pop_size, fill_value=np.inf)

    def step(self):
        self.scheduler.step()

    def write_logs(self):
        current_time =  self.scheduler.time
        data = dict(
                    current_susceptible=int(self.susceptible_count),
                    current_exposed=int(self.exposed_count),
                    current_presymptomtic=int(self.presymptomatic_count),
                    current_asymptomatic=int(self.asymptomatic_count),
                    current_symptomatic=int(self.symptomatic_count),
                    current_hospitalized=int(self.hospitalized_count),
                    current_infected=int(self.infected_count),
                    current_recovered=int(self.recovered_count),
                    current_dead=int(self.died_count),
                    total_contagious_count=int((self.date_contagious < current_time).sum()),
                    total_infected_count=int((self.date_infected < current_time).sum()),
                    total_hospitalized_count=int((self.date_hospitalized < current_time).sum()),
        )
        info = json.dumps(data)
        log_to_file(self.log_file, info)