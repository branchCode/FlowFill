import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import math
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel

logger = logging.getLogger('base')

class InpaintModel(BaseModel):
    def __init__(self, opt):
        super(InpaintModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        self.netG = networks.define_G(opt).to(self.device)
        # 1 for batch dimension
        self.zshape = [1] + self.netG.out_shapes()
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params_G = []
            optim_params_D = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    if 'encoder' in k:
                        optim_params_D.append(v)
                    else:
                        optim_params_G.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params_G, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizer_D = torch.optim.Adam(optim_params_D, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer,
                                                         [int(x * train_opt['niter']) for x in train_opt['lr_steps_rel']],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.img_gt = data['img'].to(self.device)
        self.mask = data['mask'].to(self.device)
        self.img_masked = self.img_gt * (1 - self.mask) + self.mask

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

        # forward
        noise = torch.randn(self.img_gt.size(), device=self.img_gt.device).mul_((4 / math.sqrt(3)) / 255.0)
        z, log_p, logdet, _ = self.netG(self.img_gt+noise, imgs=self.img_masked, masks=self.mask)

        nll = (log_p + logdet).mean() * -1.

        nll.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])
        self.optimizer_G.step()

        if step > self.train_opt['train_encoder_delay']:
            self.optimizer_D.step()

        # set log
        self.log_dict['nll'] = nll.item()

    def test(self):
        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] != None:
            gaussian_scale = self.test_opt['gaussian_scale']

        self.netG.eval()
        with torch.no_grad():
            z = self.gaussian_batch(self.zshape) * gaussian_scale
            self.img_output, _ = self.netG(z, imgs=self.img_masked, masks=self.mask, rev=True)
            self.img_output = torch.clamp(self.img_output, 0, 1.)
            self.img_pred = self.img_gt * (1 - self.mask) + self.img_output * self.mask

        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['gt'] = self.img_gt.detach()[0].float().cpu()
        out_dict['masked'] = self.img_masked.detach()[0].float().cpu()
        out_dict['pred'] = self.img_pred.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
