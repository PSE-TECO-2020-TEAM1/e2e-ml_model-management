from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (CategoricalHyperparameter,
                                         UniformFloatHyperparameter,
                                         UniformIntegerHyperparameter,
                                         UnParametrizedHyperparameter)

cs: ConfigurationSpace = ConfigurationSpace()

criterion = CategoricalHyperparameter(
    "criterion", ["gini", "entropy"], default_value="gini")

bootstrap = CategoricalHyperparameter(
    "bootstrap", ["True", "False"], default_value="True")

max_features = UniformFloatHyperparameter(
    "max_features", 0., 1., default_value=0.5)

min_samples_split = UniformIntegerHyperparameter(
    "min_samples_split", 2, 20, default_value=2)

min_samples_leaf = UniformIntegerHyperparameter(
    "min_samples_leaf", 1, 20, default_value=1)

min_weight_fraction_leaf = UnParametrizedHyperparameter(
    "min_weight_fraction_leaf", 0.)

max_depth = UnParametrizedHyperparameter("max_depth", "None")

max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")

min_impurity_decrease = UnParametrizedHyperparameter(
    'min_impurity_decrease', 0.0)

cs.add_hyperparameters([criterion, max_features,
                        max_depth, min_samples_split, min_samples_leaf,
                        min_weight_fraction_leaf, max_leaf_nodes,
                        bootstrap, min_impurity_decrease])
