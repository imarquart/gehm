"""
Adapted from: https://github.com/moemen95/Pytorch-Project-Template
See agents folder for license!
"""
import logging


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")


    def log_loss(self, loss):
        self.logger.info("\n")
        self.logger.info(
            "Epoch: {} [Iteration {}/{} ({:.0f}%)]\tLoss: {:.6f}\n".format(
                self.current_epoch,
                self.current_epoch*len(self.dataloader.dataset),
                self.nr_epochs*len(self.dataloader.dataset),
                100.0 * (self.current_epoch*len(self.dataloader.dataset)) / (self.nr_epochs*len(self.dataloader.dataset)),
                loss,
            )
        )

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError




    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator
        :return:
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        raise NotImplementedError