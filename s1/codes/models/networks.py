import logging
from models.modules.flow_network import ConditionalFlow
logger = logging.getLogger('base')


####################
# define network
####################
def define_G(opt):
    opt_net = opt['network_G']
    in_nc = opt_net['in_nc']
    L = opt_net['L']
    K = opt_net['K']
    n_trans = opt_net['n_trans']
    heat = opt_net['heat']
    size = opt_net['input_size']

    netG = ConditionalFlow(in_channels=in_nc, L=L, K=K, n_trans=n_trans, heat=heat, size=size)

    return netG