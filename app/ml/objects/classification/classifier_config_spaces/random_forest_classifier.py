from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (CategoricalHyperparameter,
                                         UniformIntegerHyperparameter,
                                         UniformFloatHyperparameter,
                                         Constant)

cs = ConfigurationSpace()
criterion = CategoricalHyperparameter(
    "criterion", ["gini", "entropy"], default_value="gini")

# The maximum number of features used in the forest is calculated as m^max_features, where
# m is the total number of features, and max_features is the hyperparameter specified below.
# The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
# corresponds with Geurts' heuristic.
max_features = UniformFloatHyperparameter(
    "max_features", 0., 1., default_value=0.5)

max_depth = Constant("max_depth", "None")
min_samples_split = UniformIntegerHyperparameter(
    "min_samples_split", 2, 20, default_value=2)
min_samples_leaf = UniformIntegerHyperparameter(
    "min_samples_leaf", 1, 20, default_value=1)
min_weight_fraction_leaf = Constant("min_weight_fraction_leaf", 0.)
max_leaf_nodes = Constant("max_leaf_nodes", "None")
min_impurity_decrease = Constant('min_impurity_decrease', 0.0)
bootstrap = CategoricalHyperparameter(
    "bootstrap", ["True", "False"], default_value="True")
cs.add_hyperparameters([criterion, max_features,
                        max_depth, min_samples_split, min_samples_leaf,
                        min_weight_fraction_leaf, max_leaf_nodes,
                        bootstrap, min_impurity_decrease])
