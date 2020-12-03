import torch
import torch.nn as nn
from pytorch_lightning import Trainer

import logging
import sys
sys.path.append("./models")
sys.path.append("./utils")

from interface import *
from utils import *
from vae import VAE


def main(args):
    """ main() driver function """

    # Parameters parsing
    if filepath_is_not_valid(args.config):
        logging.error("The path {} is not a file. Aborting..".format(args.config))
        exit()

    configuration, architecture, hyperparameters = parse_config_file(args.config, args.variation)
    dataset_info = prepare_dataset(configuration)
    if (dataset_info is None):
        exit()

    # Initialization
    model = VAE(architecture, hyperparameters, dataset_info)
    trainer = Trainer(max_epochs = hyperparameters["epochs"], gpus=None, fast_dev_run=True)

    # Training and testing
    trainer.fit(model)
    result = trainer.test(model)
    model.sample(1)

if __name__ == "__main__":
    """ call main() function here """
    print()
    # configure the level of the logging and the format of the messages
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s\n")
    # parse the command line input
    args = parse_cmd_args()
    # call the main() driver function
    main(args)
    print("\n")
