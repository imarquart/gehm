import numpy as np

from tqdm.autonotebook import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from gehm.agents.base import BaseAgent

# import your classes here
import networkx as nx

from numpy import ndarray, cos, sin
from torch import tensor
from typing import Union, Optional
import torch
import numpy as np
from torch import int16
from torch.utils.data import DataLoader
import networkx as nx
from gehm.losses.sdne_loss_functions import *
from gehm.datasets.nx_datasets import *
from gehm.model.sdne import SDNEmodel
from tqdm import tqdm


# from tensorboardX import SummaryWriter

cudnn.benchmark = True


class SDNEAgent(BaseAgent):
    def __init__(self, config, G: Union[nx.Graph, nx.DiGraph]):
        super().__init__(config)

        self.config = config

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info(
                "WARNING: You have a CUDA device, so you should probably enable CUDA"
            )

        self.cuda = self.is_cuda & self.config.cuda
        if self.cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        # set the manual seed for torch
        self.manual_seed = self.config.seed
        np.random.seed(self.manual_seed)
        torch.manual_seed(self.manual_seed)

        self.nr_nodes = len(G.nodes)
        self.nr_epochs = config.nr_epochs

        # activation
        if config.activation == "Tanh":
            activation = torch.nn.Tanh
        else:
            activation = torch.nn.Tanh

        # dataset
        self.dataset = nx_dataset_onelevel(G)

        # define model
        self.model = SDNEmodel(
            dim_input=self.nr_nodes,
            dim_intermediate=config.dim_intermediate,
            dim_embedding=config.dim_embedding,
            activation=activation,
            nr_encoders=config.nr_encoders,
            nr_decoders=config.nr_decoders,
        )

        # define data_loader
        self.dataloader = DataLoader(
            self.dataset, batch_size=config.batch_size, shuffle=config.shuffle
        )

        # define loss
        self.se_loss = SDNESELoss(beta=config.beta1, device=self.device)
        self.pr_loss = SDNEProximityLoss(device=self.device)

        # define optimizers for both generator and discriminator
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            amsgrad=config.amsgrad,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.schedule_step_size,
            gamma=config.schedule_gamma,
        )

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0
        self.losses_dict = {}
        self.lr_list=[]

        if self.cuda:
            torch.cuda.manual_seed_all(self.manual_seed)
            torch.cuda.manual_seed(self.manual_seed)
            # torch.cuda.set_device(self.device)
            self.model = self.model.cuda()
            self.se_loss = self.se_loss.cuda()
            self.pr_loss = self.pr_loss.cuda()
            self.logger.info("Program will run on *****GPU-CUDA***** ")
        else:
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        # self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        # self.summary_writer = None

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        pass

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        pass

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        losses = []
        se_losses = []
        pr_losses = []
        lr_list = []
        self.model.train()

        self.current_epoch = 0

        desc = ""
        pbar = tqdm(range(0, self.nr_epochs), desc=desc, position=0, leave=False)
        for epoch in pbar:
            self.current_epoch = epoch
            epoch_loss, se_loss_epoch, pr_loss_epoch = self.train_one_epoch()

            pbar.set_description(
                "Loss: {}, LR:{}".format(epoch_loss, self.scheduler.get_last_lr())
            )

            self.scheduler.step()

            if epoch_loss > 0:
                losses.append(epoch_loss.cpu().detach().numpy())
                se_losses.append(se_loss_epoch.cpu().detach().numpy())
                pr_losses.append(pr_loss_epoch.cpu().detach().numpy())

            lr_list.append(
                self.scheduler.get_last_lr()[0]
                if isinstance(self.scheduler.get_last_lr(), list)
                else self.scheduler.get_last_lr().detach().numpy()
            )

            if epoch*len(self.dataloader.dataset) % self.config.log_interval == 0:
                self.log_loss(epoch_loss)

        self.losses_dict['total_loss']=losses
        self.losses_dict['se_loss']=se_losses
        self.losses_dict['pr_loss']=pr_losses
        self.lr_list=lr_list

        pass

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """

        desc = ""

        epoch_loss = torch.tensor(0)
        se_loss_epoch = torch.tensor(0)
        pr_loss_epoch = torch.tensor(0)

        self.current_iteration = 0

        for i, data in enumerate(self.dataloader):

            self.optimizer.zero_grad()
            node_ids, sim1, sim2 = data
            node_ids = node_ids.to(self.device)
            sim1 = sim1.to(self.device)
            positions, est_sim = self.model(sim1)

            se_loss_value = self.se_loss(est_similarity=est_sim, similarity=sim1)
            pr_loss_value = self.pr_loss(
                positions=positions, similarity=sim1, indecies=node_ids
            )

            total_loss = se_loss_value + pr_loss_value
            total_loss.backward()
            self.optimizer.step()

            epoch_loss += total_loss.cpu().detach().numpy()
            se_loss_epoch += se_loss_value.cpu().detach().numpy()
            pr_loss_epoch += pr_loss_value.cpu().detach().numpy()

            self.current_iteration += 1



        return epoch_loss, se_loss_epoch, pr_loss_epoch

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        pass

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass
