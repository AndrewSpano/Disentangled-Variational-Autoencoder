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
    # first make sure that the paths to the provided config file are valid
    if filepath_is_not_valid(args.config):
        logging.error("The path {} is not a file. Aborting..".format(args.config))
        exit()

    configuration, architecture = parse_config_file(args.config)

    # train_set, test_set = load_dataset(configuration["dataset"])

    model = VAE(architecture, 784, 2, 16)

    trainer = Trainer(max_epochs = 5, gpus=None, fast_dev_run=True)
    trainer.fit(model)
    trainer.test(model)

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
