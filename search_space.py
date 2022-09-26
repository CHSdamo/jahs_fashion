import ConfigSpace as CS
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, OrdinalHyperparameter


def search_space():

    # Architecture Space
    # Convolution layers
    n_conv_layers = OrdinalHyperparameter("n_conv_layers", [1, 2, 3])
    n_conv_0 = UniformIntegerHyperparameter("n_conv_0", 8, 256, log=True)
    n_conv_1 = UniformIntegerHyperparameter("n_conv_1", 8, 256, log=True)
    n_conv_2 = UniformIntegerHyperparameter("n_conv_2", 8, 256, log=True)

    # Conditions for conv layers
    cond_conv_1 = CS.conditions.InCondition(n_conv_1, n_conv_layers, [2, 3])
    cond_conv_2 = CS.conditions.InCondition(n_conv_2, n_conv_layers, [3])

    # Kernel Size
    kernel_size = OrdinalHyperparameter("kernel_size", [3, 5])

    # Use Batch Normalization
    batch_norm = CategoricalHyperparameter("use_BN", choices=[False, True])

    # Global Avg Pooling
    global_avg_pooling = CategoricalHyperparameter("global_avg_pooling", choices=[False, True])

    # Dense layers
    n_fc_layers = OrdinalHyperparameter("n_fc_l", [1, 2])
    n_fc_0 = UniformIntegerHyperparameter("n_fc_0", 512, 2048, log=True)
    n_fc_1 = UniformIntegerHyperparameter("n_fc_1", 512, 2048, log=True)

    # Conditions for Dense layers
    cond_fc_0 = CS.conditions.InCondition(n_fc_0, n_fc_layers, [1, 2])
    cond_fc_1 = CS.conditions.InCondition(n_fc_1, n_fc_layers, [2])

    # Hyperparameters Space
    # Learning Rate
    learning_rate_init = UniformFloatHyperparameter('learning_rate_init', 1e-3, 1e0, log=True)
    weight_decay = UniformFloatHyperparameter("weight_decay", lower=1e-5, upper=1e-2, log=True)

    # Batch size
    batch_size = OrdinalHyperparameter('batch_size', [2**x for x in range(6, 12)])

    # Dropout
    dropout_rate = OrdinalHyperparameter("dropout_rate", [0.1, 0.2, 0.3, 0.4, 0.5])

    # Optimizer
    optimizer = CategoricalHyperparameter("optimizer", choices=["Adam", "SGD"])

    # Data argumentation
    random_horizontal_flip = CategoricalHyperparameter("random_horizontal_flip", choices=[True, False])
    random_rotation = CategoricalHyperparameter("random_rotation", choices=[True, False])

    # create configuration space
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameters([n_conv_layers, n_conv_0, n_conv_1, n_conv_2, kernel_size, global_avg_pooling])
    cs.add_hyperparameters([n_fc_layers, n_fc_0, n_fc_1])
    cs.add_hyperparameters([batch_norm, learning_rate_init, weight_decay, batch_size, dropout_rate, optimizer])
    cs.add_hyperparameters([random_horizontal_flip, random_rotation])
    cs.add_conditions([cond_conv_1, cond_conv_2, cond_fc_0, cond_fc_1])

    return cs
