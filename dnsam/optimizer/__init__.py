import torch.optim as optim

from .sam import SAM
from .usam import USAM
from .sam_faster import SAM_Faster
from .adamsam import ADAMSAM
from .gdsam import GDSAM
from .msam import MSAM
from .sama import SAMA
from .samac import SAMAC
from .sama_lbgfs import SAMA_LBGFS
from .samaf import SAMAF
from .samaccer import SAMACCER
from .samawm import SAMAWM
from .samdawm import SAMDAWM


def get_optimizer(net, cfg):
    base_opt_name = cfg['model'].get('base_opt', None)
    if base_opt_name is None:
        base_optimizer = optim.SGD
    elif base_opt_name == 'adam':
        base_optimizer = optim.Adam
    if cfg['model']['name'] == 'sam':
        return SAM(
            net.parameters(), 
            base_optimizer, 
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            rho=cfg['model']['rho'], 
            adaptive=cfg['model']['adaptive'],
            nesterov=cfg['model']['nesterov']
        )
    elif cfg['model']['name'] == 'usam':
        return USAM(
            net.parameters(), 
            base_optimizer, 
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            rho=cfg['model']['rho'], 
            adaptive=cfg['model']['adaptive'],
            nesterov=cfg['model']['nesterov'],
        )
    elif cfg['model']['name'] == 'adamsam':
        return ADAMSAM(
            net.parameters(), 
            base_optimizer, 
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            betas=cfg['model']['betas'],
            rho=cfg['model']['rho'], 
            adaptive=cfg['model']['adaptive'],
            nesterov=cfg['model']['nesterov'],
        )
    elif cfg['model']['name'] == 'gdsam':
        return GDSAM(
            net.parameters(), 
            base_optimizer=base_optimizer, 
            model=net,
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            gdsam_alpha=cfg['model']['gdsam_alpha'],
            rho=cfg['model']['rho'], 
            adaptive=cfg['model']['adaptive'],
            nesterov=cfg['model']['nesterov'],
        )
    elif cfg['model']['name'] == 'msam':
        return MSAM(
            net.parameters(), 
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            rho=cfg['model']['rho'], 
            nesterov=cfg['model']['nesterov'],
        )
    elif cfg['model']['name'] == 'sama':
        return SAMA(
            net.parameters(), 
            base_optimizer, 
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            rho=cfg['model']['rho'], 
            adaptive=cfg['model']['adaptive'],
            nesterov=cfg['model']['nesterov'],
            betas=cfg['model']['betas'],
        )
    elif cfg['model']['name'] == 'samaf':
        return SAMAF(
            net.parameters(), 
            base_optimizer, 
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            rho=cfg['model']['rho'], 
            adaptive=cfg['model']['adaptive'],
            nesterov=cfg['model']['nesterov'],
            betas=cfg['model']['betas'],
        )
    elif cfg['model']['name'] == 'sama_lbgfs':
        return SAMA_LBGFS(
            net.parameters(), 
            base_optimizer, 
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            rho=cfg['model']['rho'], 
            adaptive=cfg['model']['adaptive'],
            nesterov=cfg['model']['nesterov'],
            betas=cfg['model']['betas'],
        )
    elif cfg['model']['name'] == 'samac':
        return SAMAC(
            net.parameters(), 
            base_optimizer, 
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            rho=cfg['model']['rho'], 
            adaptive=cfg['model']['adaptive'],
            nesterov=cfg['model']['nesterov'],
        )
    elif cfg['model']['name'] == 'sam_faster':
        return SAM_Faster(
            net.parameters(), 
            base_optimizer, 
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            rho=cfg['model']['rho'], 
            adaptive=cfg['model']['adaptive'],
            nesterov=cfg['model']['nesterov']
        )
    elif cfg['model']['name'] == 'samaccer':
        return SAMACCER(
            net.parameters(), 
            base_optimizer, 
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            rho=cfg['model']['rho'], 
            adaptive=cfg['model']['adaptive'],
            nesterov=cfg['model']['nesterov'],
            betas=cfg['model']['betas'],
        )
    elif cfg['model']['name'] == 'samawm':
        return SAMAWM(
            net.parameters(), 
            base_optimizer, 
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            rho=cfg['model']['rho'], 
            adaptive=cfg['model']['adaptive'],
            nesterov=cfg['model']['nesterov'],
            betas=cfg['model']['betas'],
        )
    elif cfg['model']['name'] == 'samdawm':
        return SAMDAWM(
            net.parameters(), 
            base_optimizer, 
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            rho=cfg['model']['rho'], 
            adaptive=cfg['model']['adaptive'],
            nesterov=cfg['model']['nesterov'],
            betas=cfg['model']['betas'],
        )
    else:
        raise ValueError("Invalid optimizer!!!")