from torch._C import device
import torch.nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
#from tqdm import tqdm
from tqdm.autonotebook import tqdm

import numpy as np
import networkx as nx

from gehm.agents.base import BaseAgent
from gehm.datasets.nx_hierarchical_datasets import nx_hierarchical_dataset
from gehm.losses.sdne_hierarchical_losses import *
from gehm.model.hsdne import hSDNEmodel
from gehm.agents.sdne import SDNEAgent
# import your classes here

# from tensorboardX import SummaryWriter
from gehm.utils.measurements import aggregate_measures

cudnn.benchmark = True




class hSDNEAgent(SDNEAgent):
    def __init__(self, config, G: Union[nx.Graph, nx.DiGraph], hierarchy_dict:dict=None):
        super(hSDNEAgent, self).__init__(config, G)


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
        self.dataset=nx_hierarchical_dataset(G, hierarchy_dict=hierarchy_dict)

        # define model
        self.model = hSDNEmodel(
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
        self.predict_dataloader = DataLoader(
            self.dataset, batch_size=config.batch_size, shuffle=False
        )
        # define loss
        self.se_loss = hSDNESELoss(beta=config.beta1, device=self.device)
        self.pr_loss = hSDNEProximityLoss(device=self.device)
        self.area_loss = hSDNEAreaLoss(device=self.device)
        self.var_loss = hSDNEVarianceLoss(device=self.device)

        self.area_coef = torch.tensor(config.area_loss).to(self.device)
        self.area_coef.requires_grad=False
        self.var_coef = torch.tensor(config.variance_loss).to(self.device)
        self.var_coef.requires_grad=False

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
            node_ids, hierarchy, attn,sim1, sim2= data
            node_ids = node_ids.to(self.device)
            sim1 = sim1.to(self.device)
            attn = attn.to(self.device)
            hierarchy = hierarchy.to(self.device)
            positions, est_sim = self.model(sim1,attn,hierarchy)

            se_loss_value = self.se_loss(est_similarity=est_sim, similarity=sim1, attn=attn)
            pr_loss_value = self.pr_loss(
                positions=positions, similarity=sim1, indecies=node_ids, attn=attn
            )
            area_loss_values = self.area_loss(positions)
            variance_loss= self.var_loss(positions, hierarchy)

            total_loss = se_loss_value + pr_loss_value + self.area_coef*area_loss_values+self.var_coef*variance_loss
            total_loss.backward()
            self.optimizer.step()

            epoch_loss += total_loss.cpu().detach().numpy()
            se_loss_epoch += se_loss_value.cpu().detach().numpy()
            pr_loss_epoch += pr_loss_value.cpu().detach().numpy()

            self.current_iteration += 1

        return epoch_loss, se_loss_epoch, pr_loss_epoch


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
            for i, data in pbar:
                node_ids, hierarchy, attn,sim1, sim2= data
                node_ids = node_ids.to(self.device)
                sim1 = sim1.to(self.device)
                attn = attn.to(self.device)
                hierarchy = hierarchy.to(self.device)
                positions, est_sim = self.model(sim1,attn,hierarchy)

                se_loss_value = self.se_loss(est_similarity=est_sim, similarity=sim1, attn=attn)
                pr_loss_value = self.pr_loss(
                    positions=positions, similarity=sim1, indecies=node_ids, attn=attn
                )
                total_loss = se_loss_value + pr_loss_value
                losses.append(total_loss.cpu().detach().numpy())
                se_losses.append(se_loss_value.cpu().detach().numpy())
                pr_losses.append(pr_loss_value.cpu().detach().numpy())

                nodes.append(node_ids.cpu().detach().numpy())
                position_list.append(positions.cpu().detach().numpy())
                similarity_list.append(est_sim.cpu().detach().numpy())

        return self.stack_sample(nodes,position_list,similarity_list),losses
