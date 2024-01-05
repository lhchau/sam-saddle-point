from ..optimizer import SAM, SAM_Faster, DNSAM, RDNSAM

def get_optimizer(net, base_optimizer, cfg):
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
    elif cfg['model']['name'] == 'dnsam':
        return DNSAM(
            net.parameters(), 
            base_optimizer, 
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            rho=cfg['model']['rho'], 
            adaptive=cfg['model']['adaptive'],
            nesterov=cfg['model']['nesterov'],
            dnsam_theta=cfg['model']['dnsam_theta']
        )
    elif cfg['model']['name'] == 'rdnsam':
        return RDNSAM(
            net.parameters(), 
            base_optimizer, 
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            rho=cfg['model']['rho'], 
            adaptive=cfg['model']['adaptive'],
            nesterov=cfg['model']['nesterov'],
            dnsam_theta=cfg['model']['dnsam_theta'],
            restart_step=cfg['model']['restart_step']
        )
    else:
        raise ValueError("Invalid optimizer!!!")