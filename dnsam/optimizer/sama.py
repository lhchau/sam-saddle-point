import torch
import math


class SAMA(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, beta=0.9, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAMA, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.state['step'] = 0
        self.beta = beta

    @torch.no_grad()
    def first_step(self, zero_grad=False):        
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                self.state[p]["old_g"] = p.grad.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                
                self.state['step'] += 1
                bias_correction = 1 - self.beta ** self.state['step']

                estimated_hess = (self.state[p]["old_g"] - p.grad) / (p.data - self.state[p]["old_p"] + 1e-12)
                if 'hess' not in self.state[p].keys():
                    self.state[p]['hess'] = estimated_hess.data.clone()
                else:
                    self.state[p]['hess'].mul_(self.beta).add_(estimated_hess, alpha=1-self.beta)
                denom = self.state[p]['hess'].abs().add_(1e-12).sqrt() / math.sqrt(bias_correction)

                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                
                p.grad.div_(denom)
                p.grad = p.grad.clamp(None, 1)

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

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
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
