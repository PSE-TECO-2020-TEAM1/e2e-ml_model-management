from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (CategoricalHyperparameter,
                                         UniformFloatHyperparameter,
                                         UniformIntegerHyperparameter,
                                         UnParametrizedHyperparameter)

cs: ConfigurationSpace = ConfigurationSpace()

C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True,
                               default_value=1.0)

kernel = CategoricalHyperparameter(name="kernel",
                                   choices=["rbf", "poly", "sigmoid"],
                                   default_value="rbf")
degree = UniformIntegerHyperparameter("degree", 2, 5, default_value=3)

gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8,
                                   log=True, default_value=0.1)
coef0 = UniformFloatHyperparameter("coef0", -1, 1, default_value=0)

shrinking = CategoricalHyperparameter("shrinking", ["True", "False"],
                                      default_value="True")

tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-3,
                                 log=True)

max_iter = UnParametrizedHyperparameter("max_iter", -1)

degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")

coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])

cs.add_hyperparameters([C, kernel, degree, gamma, coef0, shrinking,
                        tol, max_iter])

cs.add_condition(degree_depends_on_poly)
cs.add_condition(coef0_condition)
