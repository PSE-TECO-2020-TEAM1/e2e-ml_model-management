from ConfigSpace.conditions import InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (CategoricalHyperparameter, Constant,
                                         UniformFloatHyperparameter,
                                         UniformIntegerHyperparameter)

cs = ConfigurationSpace()
hidden_layer_depth = UniformIntegerHyperparameter(name="hidden_layer_depth",
                                                  lower=1, upper=3, default_value=1)

num_nodes_per_layer = UniformIntegerHyperparameter(name="num_nodes_per_layer",
                                                   lower=16, upper=264, default_value=32,
                                                   log=True)

activation = CategoricalHyperparameter(name="activation", choices=['tanh', 'relu'],
                                       default_value='relu')

alpha = UniformFloatHyperparameter(name="alpha", lower=1e-7, upper=1e-1, default_value=1e-4,
                                   log=True)

learning_rate_init = UniformFloatHyperparameter(name="learning_rate_init",
                                                lower=1e-4, upper=0.5, default_value=1e-3,
                                                log=True)

early_stopping = CategoricalHyperparameter(name="early_stopping",
                                           choices=["valid", "train"],
                                           default_value="valid")

n_iter_no_change = Constant(name="n_iter_no_change", value=32)

validation_fraction = Constant(name="validation_fraction", value=0.1)

solver = Constant(name="solver", value='adam')

tol = Constant(name="tol", value=1e-4)

batch_size = Constant(name="batch_size", value="auto")

shuffle = Constant(name="shuffle", value="True")

beta_1 = Constant(name="beta_1", value=0.9)

beta_2 = Constant(name="beta_2", value=0.999)

epsilon = Constant(name="epsilon", value=1e-8)

cs.add_hyperparameters([hidden_layer_depth, num_nodes_per_layer,
                        activation, alpha,
                        learning_rate_init, early_stopping,
                        n_iter_no_change, validation_fraction, tol,
                        solver, batch_size, shuffle,
                        beta_1, beta_2, epsilon])

validation_fraction_cond = InCondition(validation_fraction, early_stopping, ["valid"])

cs.add_conditions([validation_fraction_cond])
