import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, adaptive_rho=None, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        # self.hsam_beta = 0.95
        
        self.adaptive_rho = adaptive_rho

    @torch.no_grad()
    def first_step(self, zero_grad=False):        
        # for group in self.param_groups:            
        #     for i, p in enumerate(group["params"]):
        #         if p.grad is None: continue

                # if 'hessian' not in self.state[p].keys():
                #     self.state[p]['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # self.state[p]['hessian'].mul_(self.hsam_beta).addcmul_(p.grad, p.grad, value=1 - self.hsam_beta)
        
        grad_norm = self._grad_norm()
        # hessian_norm = self._hessian_norm()
        
        for group in self.param_groups:
            # self.grad_norm, self.hessian_norm, self.scale = grad_norm, hessian_norm, scale

            for p, rho in zip(group["params"], self.adaptive_rho):
                scale = rho / (grad_norm + 1e-12)
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        # self.weight_norm1 = self.get_weight_norm()
        
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        # self.weight_norm2 = self.get_weight_norm()

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
    def _hessian_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        (self.state[p]['hessian']).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
    def get_weight_norm(self):
        total_norm = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    total_norm += torch.norm(p, p=2).item() ** 2
        return total_norm ** 0.5
    
    def get_norm(self):
        return self.grad_norm, self.hessian_norm, self.scale, self.weight_norm1, self.weight_norm2
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups