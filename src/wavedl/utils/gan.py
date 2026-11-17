"""OpenFWI WGAN-GP training primitives (BSD-3-Clause; see THIRD_PARTY_NOTICES).

The update order, interpolation and losses follow lanl/OpenFWI gan_train.py
and utils.py at 48754806b7b4c5877259c6b958a87f4513fcdc0b.
"""

import math

import torch
from torch import nn


class OpenFWIPixelLoss(nn.Module):
    """Weighted L1 + MSE loss used by OpenFWI's generator and validation."""

    def __init__(self, lambda_g1v=100.0, lambda_g2v=100.0):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v

    def forward(self, prediction, target):
        return self.lambda_g1v * nn.functional.l1_loss(
            prediction, target
        ) + self.lambda_g2v * nn.functional.mse_loss(prediction, target)


def validate_adversarial_options(args, model_class):
    """Reject combinations whose GAN update semantics are not supported."""
    if getattr(args, "adversarial", "none") == "none":
        return
    if not callable(getattr(model_class, "build_discriminator", None)):
        raise ValueError(
            "--adversarial wgan-gp requires a model exposing build_discriminator()."
        )
    if args.grad_accum_steps != 1 or args.compile or args.precision != "no":
        raise ValueError(
            "WGAN-GP requires --grad_accum_steps 1 --precision no and no --compile (second-order gradient penalty)."
        )
    if args.cv or args.constraint or args.constraint_file:
        raise ValueError(
            "WGAN-GP does not support cross-validation or physical constraints."
        )
    if args.scheduler == "onecycle":
        raise ValueError(
            "WGAN-GP does not support --scheduler onecycle; use an epoch scheduler."
        )
    if args.n_critic < 1:
        raise ValueError("--n_critic must be >= 1.")
    if len(args.gan_betas) != 2 or not all(0 <= b < 1 for b in args.gan_betas):
        raise ValueError("--gan_betas must contain two values in [0, 1).")
    for name in ("lambda_g1v", "lambda_g2v", "lambda_adv", "lambda_gp", "d_lr"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name == "d_lr" and value == 0):
            raise ValueError(
                f"--{name} must be finite and {'positive' if name == 'd_lr' else 'nonnegative'}."
            )


def gradient_penalty(critic, real, fake):
    """Upstream interpolation and per-sample L2 gradient-norm penalty."""
    alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def wgan_gp_step(
    model,
    critic,
    optimizer,
    critic_optimizer,
    criterion,
    x,
    y,
    batch_index,
    num_batches,
    args,
    accelerator,
):
    """Update critic every batch, then generator every n_critic and last batch.

    Like upstream, recompute G(x) for a generator update and do not clip gradients.
    The no-grad generator pass still updates train-mode batch-normalization buffers.
    Return pixel loss for reporting (also on critic-only batches).
    """
    critic_optimizer.zero_grad()
    with torch.no_grad():
        fake = model(x)
    if fake.shape != y.shape:
        raise ValueError(
            f"Model output shape {tuple(fake.shape)} does not match target shape {tuple(y.shape)}."
        )
    gp = gradient_penalty(critic, y, fake.detach())
    loss_real = critic(y).mean()
    loss_fake = critic(fake.detach()).mean()
    loss_d = loss_fake - loss_real + args.lambda_gp * gp
    if not torch.isfinite(loss_d.detach()).all():
        raise FloatingPointError("Critic loss is non-finite; training aborted.")
    accelerator.backward(loss_d)
    critic_optimizer.step()
    loss = criterion(fake, y)
    if (batch_index + 1) % args.n_critic == 0 or batch_index == num_batches - 1:
        optimizer.zero_grad()
        prediction = model(x)
        loss = criterion(prediction, y)
        loss_g = loss - args.lambda_adv * critic(prediction).mean()
        if not torch.isfinite(loss_g.detach()).all():
            raise FloatingPointError("Generator loss is non-finite; training aborted.")
        accelerator.backward(loss_g)
        optimizer.step()
    return loss.detach()
