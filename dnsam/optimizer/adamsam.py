import torch


class ADAMSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, betas=(0.965, 0.95), rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(ADAMSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.beta1, self.beta2 = betas

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:            
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue

                if 'hessian' not in self.state[p].keys():
                    self.state[p]['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                self.state[p]['hessian'].mul_(self.beta2).addcmul_(p.grad, p.grad, value=1 - self.beta2)
                
                exp_avg = self.state[p].get('exp_avg', None)
                if exp_avg is None:
                    exp_avg = torch.clone(p.grad).detach()
                    exp_avg.mul_(1 - self.beta1)
                else:
                    exp_avg.mul_(self.beta1).add_(p.grad, alpha=1 - self.beta1)
                self.state[p]['exp_avg'] = exp_avg
                
                ascent_grad = (exp_avg.abs() / (self.state[p]['hessian'].sqrt() + 1e-15)).clamp(None, 1)
                ascent_grad.mul_(exp_avg.sign())
                
                self.state[p]['ascent_grad'] = ascent_grad
        
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group['rho']
            self.grad_norm, self.scale = grad_norm, scale

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                
                e_w = self.state[p]['ascent_grad'] * scale
                
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

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
                        self.state[p]['ascent_grad'].norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
    def get_log(self):
        return self.grad_norm, self.scale
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
