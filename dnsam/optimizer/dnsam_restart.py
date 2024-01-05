import torch


class RDNSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, dnsam_theta=0.9, rho=0.05, restart_step=352, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(RDNSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.dnsam_theta = dnsam_theta
        self.restart_step = restart_step
        self.curr_step = 0

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            dnsam_buffer_list = []
            params_with_grad = []
            d_p_list = []

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                dnsam_buffer_list.append(
                    self.state[p].get('dnsam_buffer', None))

            for i, p in enumerate(params_with_grad):
                d_p = d_p_list[i]
                if self.dnsam_theta != 0:
                    buf = self.state[p].get('dnsam_buffer', None)

                    if buf is None or self.curr_step == self.restart_step:
                        buf = torch.clone(d_p).detach()
                        dnsam_buffer_list[i] = buf
                        self.curr_step = 0
                    else:
                        buf.mul_(1 - self.dnsam_theta).add_(d_p,
                                                            alpha=self.dnsam_theta)
            # update momentum_buffers in state
            for p, dnsam_buffer in zip(group["params"], dnsam_buffer_list):
                state = self.state[p]
                state['dnsam_buffer'] = dnsam_buffer

        sam_grad_norm = self._grad_norm()
        grad_norm = self._grad_norm_dnsam()
        self.step_length = sam_grad_norm / grad_norm

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2)
                       if group["adaptive"] else 1.0) * p.grad * scale.to(p)

                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

        self.curr_step += 1

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # get back to "w" from "w + e(w)"
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        # the closure should do a full forward-backward pass
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm_dnsam(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) *
                         self.state[p]["dnsam_buffer"]).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                        ]),
            p=2
        )
        return norm

    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0)
                         * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                        ]),
            p=2
        )
        return norm

    def _get_step_length(self):
        return self.step_length

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
