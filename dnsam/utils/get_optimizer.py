from ..optimizer import SAM, SAM_Faster, DNSAM, RDNSAM, DSAM, FDNSAM, HSAM, PSAM, ARSAM, BSAM, USAM

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
    elif cfg['model']['name'] == 'dsam':
        return DSAM(
            net.parameters(), 
            base_optimizer, 
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            rho=cfg['model']['rho'], 
            adaptive=cfg['model']['adaptive'],
            nesterov=cfg['model']['nesterov'],
            dsam_theta=cfg['model']['dsam_theta'],
        )
    elif cfg['model']['name'] == 'fdnsam':
        return FDNSAM(
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
    elif cfg['model']['name'] == 'hsam':
        return HSAM(
            net.parameters(), 
            base_optimizer, 
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            rho=cfg['model']['rho'], 
            adaptive=cfg['model']['adaptive'],
            nesterov=cfg['model']['nesterov'],
            hsam_beta=cfg['model']['hsam_beta'],
            bs=cfg['data']['batch_size']
        )
    elif cfg['model']['name'] == 'arsam':
        return ARSAM(
            net.parameters(), 
            base_optimizer, 
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            rho=cfg['model']['rho'], 
            adaptive=cfg['model']['adaptive'],
            nesterov=cfg['model']['nesterov'],
            hsam_beta=cfg['model']['hsam_beta'],
            bs=cfg['data']['batch_size']
        )
    elif cfg['model']['name'] == 'psam':
        return PSAM(
            net.parameters(), 
            base_optimizer, 
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            rho=cfg['model']['rho'], 
            adaptive=cfg['model']['adaptive'],
            nesterov=cfg['model']['nesterov'],
        )
    elif cfg['model']['name'] == 'bsam':
        return BSAM(
            net.parameters(), 
            base_optimizer, 
            lr=cfg['model']['lr'], 
            momentum=cfg['model']['momentum'], 
            weight_decay=cfg['model']['weight_decay'],
            rho=cfg['model']['rho'], 
            adaptive=cfg['model']['adaptive'],
            nesterov=cfg['model']['nesterov'],
            alpha=cfg['model']['alpha']
        )
    else:
        raise ValueError("Invalid optimizer!!!")