### WARNING: Work in progress.

This package is not yet functional + has not been tested. If you're interested in contributing, feel free to contact the author on [twitter](http://twitter.com/johnurbanik).

# AgentSEIR-VI


AgentSEIR-VI is a proof of concept of a scalable agent based model in Python. The model uses [Mesa](https://mesa.readthedocs.io/en/master/) as scaffolding, but as opposed to implementing individual agents, it uses a set of arrays to back the agent, allowing most operations to be vectorized and avoid the overhead of Python.

This serves multiple purposes:
- allow for more complex modeling and larger scales.
- allow for Bayesian inference over the model parameters using data.

As an example of more complex modeling, consider the possibility of an agent that adapts its behavior based on the state of its spatial neighborhood, as well as agents making up a complex adaptive system wherein their contact network changes reactively based on information.

Here, we utilize the PPL [numpyro](https://github.com/pyro-ppl/numpyro) for inference, where both group level and individual level parameters are modeled.

In the future, the model could be abstracted to allow more flexible specification of the states, decision heuristics, interaction topology, and environment. This would allow for modeling scenarios including different types of NPIs and behaviors.

Given the fine grained modeling possible in ABM, special cases of this approach should generalize many network theory based epidemiological models (i.e. see how [stellargraph uses matrix representations for large networks](https://medium.com/stellargraph/faster-machine-learning-on-larger-graphs-how-numpy-and-pandas-slashed-memory-and-time-in-79b6c63870ef)) as well as mean-field based approaches. Using something like [networkit](https://networkit.github.io/dev-docs/python_api/algebraic.html) to generate adjacency graphs may be useful in the former case.

 However, it may not be able to easily capture multi-scale modeling approaches (without some additional apparatus to model hierarchies of agent populations), and does not incorporate vital dynamics well.

### Data:

Data for simulations can be sourced from the API client, which sources data from [The COVID Tracking Project](https://covidtracking.com/data/api).

TODO: In the future, a generator that uses mean-field SEIR to generate counts for simulation may be useful.

### Inspirations:

This approach tries to build on work such as [Variational Inference with Agent-Based Models](https://arxiv.org/abs/1605.04360) by Wen Dong, [pyro agents](https://github.com/robertness/pyro_agents) by Robert Osazuwa Ness, and [SEIRS+ Model Framework](https://github.com/ryansmcgee/seirsplus) by Ryan Seamus McGee. It also is inspired by some investigations done by the author, on [stochastic heterogenous SEIR models](https://github.com/epi-center/planning/tree/master/modeling/stochastic-eSEIR-with-heterogeneity) using a mean-field approach.

### Thoughts:

At some point, it may be useful to start specifying agent infectiousness and behavior continuously as opposed to using an SEIR model. For example, one could use within-host models to capture *how* infectious the person is based on viral load, as was alluded to in [earlier work](https://github.com/epi-center/planning/blob/master/modeling/within-host/within_host_dynamics.ipynb) by the author.