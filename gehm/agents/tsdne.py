import torch.nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
#from tqdm import tqdm
from tqdm.autonotebook import tqdm

import numpy as np
import networkx as nx

from gehm.agents.base import BaseAgent
from gehm.datasets.nx_datasets import nx_dataset_sdne, nx_dataset_tsne,batch_nx_dataset_tsne
from gehm.losses.sdne_loss_functions import *
from gehm.model.tsdne import tSDNEmodel
from gehm.agents.sdne import SDNEAgent
# import your classes here

# from tensorboardX import SummaryWriter
from gehm.utils.measurements import aggregate_measures

cudnn.benchmark = True




class tSDNEAgent(SDNEAgent):
    def __init__(self, config, G: Union[nx.Graph, nx.DiGraph]):
        super(tSDNEAgent, self).__init__(config, G)


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
        #self.dataset = nx_dataset_tsne(G)
        self.dataset=batch_nx_dataset_tsne(G, neighborhood_size=config.neighborhood_size)

        # define model
        self.model = tSDNEmodel(
            dim_input=self.nr_nodes,
            dim_intermediate=config.dim_intermediate,
            dim_embedding=config.dim_embedding,
            activation=activation,
            nr_encoders=config.nr_encoders,
            nr_decoders=config.nr_decoders,nr_heads=config.nr_heads, dropout=config.dropout, encoder_activation=config.encoder_activation
        )

        # define data_loader
        self.dataloader = DataLoader(
            self.dataset, batch_size=config.batch_size, shuffle=config.shuffle
        )
        self.predict_dataloader = DataLoader(
            self.dataset, batch_size=config.batch_size, shuffle=False
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

        # dicts
        self.losses_dict = {}
        self.lr_list = []
        self.measures={}

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

    def predict(self):
        losses = []
        se_losses = []
        pr_losses = []
        self.model.eval()
        nodes=[]
        position_list=[]
        similarity_list=[]
        with torch.no_grad():
            pbar = tqdm(enumerate(self.dataloader), desc="Predicting sample", position=0, leave=False)
            for i,data in pbar:
                node_ids, sim1, sim2 = data
                node_ids = node_ids.to(self.device)
                sim1 = sim1.to(self.device)
                positions, est_sim = self.model(sim1)


                sim1a=sim1[:,0,:]
                node1a=node_ids[:,0]

                se_loss_value = self.se_loss(est_similarity=est_sim, similarity=sim1a)
                pr_loss_value = self.pr_loss(
                    positions=positions, similarity=sim1a, indecies=node1a
                )
                total_loss = se_loss_value + pr_loss_value
                losses.append(total_loss.cpu().detach().numpy())
                se_losses.append(se_loss_value.cpu().detach().numpy())
                pr_losses.append(pr_loss_value.cpu().detach().numpy())

                nodes.append(node1a.cpu().detach().numpy())
                position_list.append(positions.cpu().detach().numpy())
                similarity_list.append(est_sim.cpu().detach().numpy())

        return self.stack_sample(nodes,position_list,similarity_list),losses

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
                sim1a=sim1[:,0,:]
                node1a=node_ids[:,0]
                se_loss_value = self.se_loss(est_similarity=est_sim, similarity=sim1a)
                pr_loss_value = self.pr_loss(
                    positions=positions, similarity=sim1a, indecies=node1a
                )

                total_loss = 0.1*se_loss_value + 0.9*pr_loss_value
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += total_loss.cpu().detach().numpy()
                se_loss_epoch += se_loss_value.cpu().detach().numpy()
                pr_loss_epoch += pr_loss_value.cpu().detach().numpy()

                self.current_iteration += 1

            return epoch_loss, se_loss_epoch, pr_loss_epoch
