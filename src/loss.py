### run a single update
import torch
from src.kernels import l2_kernel


def IMM_loss(model, data, FLAGS, labels=None):
    M = FLAGS.M
    total_batch = data.shape[0]
    batch_size = total_batch / M

    ### loop over num groups
    loss = 0
    for i in range(batch_size):
        group_data = data[i * M : (i + 1) * M,]
        assert group_data.shape[0] == M, "Each group must have exactly M samples!"
        loss += IMM_loss_per_group(group_data, labels, model, FLAGS)

    return loss


### compute IMM loss
def IMM_loss_per_group(model, data):
    batch_size = data.shape[0]

    ### sample from prior
    noise = torch.randn_like(data)
    print(noise.shape)
    ### x_{t} <- DDIM(noise, x, t, 1)
    one = torch.full((batch_size,), 1.0, device=data.device, dtype=torch.float32)
    t = torch.full((batch_size,), 0.6, device=data.device, dtype=torch.float32)
    x_t = model.ddim(noise, data, t, one)

    assert x_t.shape == data.shape, "DDIM output wrong shape"

    ### x_{r} <- DDIM(x_{t},x,r,t)
    r = torch.full((batch_size,), 0.3, device=data.device, dtype=torch.float32)
    x_r = model.ddim(x_t, data, r, t)

    loss = MMD_loss(model, x_r, x_t, r, t)

    weight_s_t = 1

    return weight_s_t * loss


def MMD_loss(model, x_r, x_t, s, r, t):
    ### terms evaluated with grad f_{s,t}( x_{t} )
    f_grad = model.cfg_forward(
        x=x_t,
        t=t,
        s=s,
    )

    ### terms with no grad f_{s,r}( x_{r} )
    with torch.no_grad():
        f_no_grad = model.cfg_forward(x=x_r, t=r, s=s)

    assert f_grad.shape == f_no_grad.shape, "Grad and No Grad terms different shapes"
    return torch.norm(f_grad - f_no_grad, p=2, dim=tuple(range(1, f_grad.ndim)))
