# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss_original:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss
#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5,
                 k=0.1, gamma=0.01
                 ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.k = k
        self.gamma=gamma

    def __call__(self, net, disc, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        sigma = torch.clamp(input=sigma, min=0.002, max=80) # hard coded
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)

        fake, sigma_next = edm_step(net=net, sigma=sigma, x_cur=y+n, k=self.k, class_labels=labels)
        time_cond = torch.cat((sigma.reshape(-1,1), sigma_next.reshape(-1,1)), dim=1)
        fake_pred = disc(fake, labels, time_cond).squeeze()

        return loss, fake_pred 
#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class DiscLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, k=0.1,):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.k = k
    def __call__(self, net, disc, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        sigma = torch.clamp(input=sigma, min=0.002, max=80) # hard coded
        # weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        # D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        # loss = weight * ((D_yn - y) ** 2)
        with torch.no_grad():
            fake, sigma_next = edm_step(net=net, sigma=sigma, x_cur=y+n, k=self.k, class_labels=labels)
            real = y + n * (sigma_next / sigma)
        # breakpoint()
        real.requires_grad = True
        time_cond = torch.cat((sigma.reshape(-1,1), sigma_next.reshape(-1,1)), dim=1)
        fake_pred = disc(fake, labels, time_cond).squeeze()
        real_pred = disc(real, labels, time_cond).squeeze()
        
        hinge = torch.nn.ReLU(inplace=True)
        lossD = hinge(1.0 - real_pred) + hinge(1.0 + fake_pred)

        grad_real = torch.autograd.grad(outputs=real_pred.sum(), inputs=real, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2)
        return lossD, grad_penalty
        #     terms["grad_penalty"] = grad_penalty
        #     with th.no_grad():
        #         terms["real_score"] = th.sigmoid(d_real_pred)
        #         terms["fake_score"] = th.sigmoid(d_fake_pred)
        #         terms["real_acc"] = (d_real_pred > 0.0).float()
        #         terms["fake_acc"] = (d_fake_pred < 0.0).float()
        # return  lossD
#----------------------------------------------------------------------------
# sigma(t)
def sigma_ftn(t, sigma_min=0.002, sigma_max=80, rho=7):
    assert torch.all(0 <= t) and torch.all(t <= 1)
    sigma = (sigma_max**(1/rho) + t * (sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
    return torch.clamp(sigma, min=sigma_min, max=sigma_max)
# inverse of sigma(t)
def sigma_inv(sigma, sigma_min=0.002, sigma_max=80, rho=7):
    assert torch.all(sigma_min <= sigma) and torch.all(sigma <= sigma_max)
    t = (sigma**(1/rho) - sigma_max**(1/rho)) / (sigma_min**(1/rho) - sigma_max**(1/rho))
    return torch.clamp(t, min=0, max=1)

# sample projection noise level
def sample_noise_level(sigma, k=0.1, sigma_min=0.002, sigma_max=80, rho=7):
    t = sigma_inv(sigma, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
    s = torch.minimum(torch.ones_like(sigma), t + k * torch.rand_like(sigma))
    sigma_next = sigma_ftn(s)
    return torch.clamp(sigma_next, max=sigma)

# EDM step
def edm_step(
    net, sigma, x_cur, k=0.1, 
    class_labels=None,# randn_like=torch.randn_like,
    sigma_min=0.002, sigma_max=80, rho=7,
    ddim=False,
    #S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    try:
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)
    except:
        sigma_min = max(sigma_min, net.module.sigma_min)
        sigma_max = min(sigma_max, net.module.sigma_max)

    # Set time step
    t_cur = sigma
    t_next = sample_noise_level(sigma, k=k, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
    
    # # Increase noise temporarily.
    # gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    # t_hat = net.round_sigma(t_cur + gamma * t_cur)
    # x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
    try:
        t_hat = net.round_sigma(t_cur)
    except:
        t_hat = net.module.round_sigma(t_cur)
    x_hat = x_cur

    # Euler step.
    denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur
    
    # Apply 2nd order correction if t_next > sigma_min.
    if not ddim:
        denoised = net(x_next, t_next, class_labels).to(torch.float64)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next > sigma_min) * (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime) \
                    + (t_next <= sigma_min) * (t_next - t_hat) * d_cur

    return x_next, t_next