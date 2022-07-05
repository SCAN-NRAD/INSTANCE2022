from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.seed_init = 1
    config.seed_train = 1
    config.trainset = list(range(1, 70 + 1))
    config.testset = list(range(71, 100 + 1))
    config.path_normalization = "element"
    config.gradient_normalization = "element"

    config.model = config_dict.ConfigDict()
    config.model.name = "v2"
    config.model.width = 2
    config.model.instance_norm_eps = 0.8

    config.optimizer = config_dict.ConfigDict()
    config.optimizer.lr = 1e-3
    config.optimizer.algorithm = "adam"
    config.optimizer.lr_div_step = 99_999_999
    config.optimizer.lr_div_factor = 0.1
    config.optimizer.lr_div_factor_min = 1.0

    config.augmentation = config_dict.ConfigDict()
    config.augmentation.noise = 0.0
    config.augmentation.deformation = 1.0
    config.augmentation.deformation_temperature = 5e-4
    return config
