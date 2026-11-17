"""CPU-synthetic GAN update, checkpoint, and CLI coverage."""

import pickle
import sys
from argparse import Namespace

import pandas as pd
import pytest
import torch
from torch import nn

from unit_tests.test_openfwi_data import TinyDense, manifest
from wavedl import test as inference, train
from wavedl.models.registry import MODEL_REGISTRY
from wavedl.utils.gan import (
    OpenFWIPixelLoss,
    gradient_penalty,
    validate_adversarial_options,
    wgan_gp_step,
)


class TinyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(24, 1)

    def forward(self, x):
        return self.head(x.flatten(1))


class TinyGAN(TinyDense):
    def build_discriminator(self):
        return TinyCritic()


class BackwardOnly:
    @staticmethod
    def backward(loss):
        loss.backward()


def options(**overrides):
    values = {
        "adversarial": "wgan-gp",
        "grad_accum_steps": 1,
        "compile": False,
        "precision": "no",
        "cv": 0,
        "constraint": [],
        "constraint_file": None,
        "scheduler": "linear_warmup",
        "n_critic": 3,
        "gan_betas": [0.0, 0.9],
        "lambda_g1v": 100,
        "lambda_g2v": 100,
        "lambda_adv": 1,
        "lambda_gp": 10,
        "d_lr": 1e-4,
    }
    values.update(overrides)
    return Namespace(**values)


def test_gradient_penalty_matches_direct_upstream_formula():
    critic = nn.Sequential(nn.Flatten(), nn.Linear(24, 8), nn.Tanh(), nn.Linear(8, 1))
    real, fake = torch.randn(3, 1, 4, 6), torch.randn(3, 1, 4, 6)
    torch.manual_seed(7)
    actual = gradient_penalty(critic, real, fake)
    torch.manual_seed(7)
    alpha = torch.rand(real.size(0), 1, 1, 1)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(interpolates)
    grads = torch.autograd.grad(
        scores,
        interpolates,
        torch.ones(3, 1),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    expected = ((grads.view(3, -1).norm(2, dim=1) - 1) ** 2).mean()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert critic[1].weight.grad is not None


def test_wgan_update_frequency_and_last_batch(monkeypatch):
    generator = TinyGAN()
    critic = generator.build_discriminator()
    optimizer_g = torch.optim.AdamW(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizer_d = torch.optim.AdamW(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
    updates_g, updates_d = [], []
    original_g, original_d = optimizer_g.step, optimizer_d.step
    batch = 0

    def step_g(*args, **kwargs):
        updates_g.append(batch)
        return original_g(*args, **kwargs)

    def step_d(*args, **kwargs):
        updates_d.append(batch)
        return original_d(*args, **kwargs)

    monkeypatch.setattr(optimizer_g, "step", step_g)
    monkeypatch.setattr(optimizer_d, "step", step_d)
    initial_g = generator.head.weight.detach().clone()
    initial_d = critic.head.weight.detach().clone()
    for batch in range(7):
        wgan_gp_step(
            generator,
            critic,
            optimizer_g,
            optimizer_d,
            OpenFWIPixelLoss(),
            torch.randn(2, 5, 8, 6),
            torch.randn(2, 1, 4, 6),
            batch,
            7,
            options(),
            BackwardOnly(),
        )
    assert updates_d == list(range(7))
    assert updates_g == [2, 5, 6]
    assert not torch.equal(initial_g, generator.head.weight)
    assert not torch.equal(initial_d, critic.head.weight)


def test_pixel_loss_is_weighted_l1_plus_mse():
    x, y = torch.randn(2, 1, 4, 6), torch.randn(2, 1, 4, 6)
    torch.testing.assert_close(
        OpenFWIPixelLoss(3, 4)(x, y),
        3 * (x - y).abs().mean() + 4 * (x - y).square().mean(),
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"grad_accum_steps": 2}, "grad_accum_steps"),
        ({"precision": "bf16"}, "precision no"),
        ({"compile": True}, "compile"),
        ({"cv": 2}, "cross-validation"),
        ({"scheduler": "onecycle"}, "onecycle"),
        ({"n_critic": 0}, "n_critic"),
        ({"gan_betas": [0, 1]}, "gan_betas"),
        ({"lambda_gp": -1}, "lambda_gp"),
        ({"d_lr": 0}, "d_lr"),
    ],
)
def test_reject_unsupported_gan_options(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_adversarial_options(options(**overrides), TinyGAN)


def test_cli_rejects_model_without_discriminator(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wavedl-train",
            "--model",
            "cnn",
            "--adversarial",
            "wgan-gp",
            "--precision",
            "no",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        train.main()
    assert "build_discriminator" in capsys.readouterr().err


def run_tiny_training(tmp_path, monkeypatch, *, resume=None, epochs=1):
    monkeypatch.setitem(MODEL_REGISTRY, "tiny_openfwi_gan", TinyGAN)
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    data = tmp_path / "samples/manifest.txt"
    validation = tmp_path / "val/manifest.txt"
    if not data.exists():
        data, _, _ = manifest(tmp_path, lengths=(8,))
        validation, _, _ = manifest(tmp_path, "val", lengths=(3,))
    argv = [
        "wavedl-train",
        "--model",
        "tiny_openfwi_gan",
        "--adversarial",
        "wgan-gp",
        "--data_path",
        str(data),
        "--val_data_path",
        str(validation),
        "--openfwi_dataset",
        "flatvel-a",
        "--epochs",
        str(epochs),
        "--batch_size",
        "2",
        "--workers",
        "0",
        "--precision",
        "no",
        "--no_tf32",
        "--n_critic",
        "3",
        "--lr",
        "1e-4",
        "--scheduler",
        "linear_warmup",
        "--warmup_epochs",
        "0",
        "--save_every",
        "1",
        "--output_dir",
        str(tmp_path / "run"),
    ]
    if resume:
        argv.extend(["--resume", str(resume)])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        train, "plot_training_progress", lambda *a, **k: None, raising=False
    )
    train.main()
    return validation


@pytest.mark.parametrize(
    "kind", ["best_checkpoint", "epoch_1_checkpoint", "interrupted_checkpoint"]
)
def test_gan_checkpoint_resume_and_inference(tmp_path, monkeypatch, capsys, kind):
    original = train.wgan_gp_step
    if kind == "interrupted_checkpoint":

        def interrupt(*args, **kwargs):
            if args[7] == 1:
                raise KeyboardInterrupt
            return original(*args, **kwargs)

        monkeypatch.setattr(train, "wgan_gp_step", interrupt)
        with pytest.raises(KeyboardInterrupt):
            run_tiny_training(tmp_path, monkeypatch)
    else:
        run_tiny_training(tmp_path, monkeypatch)
    checkpoint = tmp_path / "run" / kind
    state = torch.load(checkpoint / "critic.pt", weights_only=True)
    assert set(state) == {"model_d", "optimizer_d", "scheduler_d"}
    assert state["optimizer_d"]["state"]
    assert tuple(state["optimizer_d"]["param_groups"][0]["betas"]) == (0, 0.9)
    with (checkpoint / "training_meta.pkl").open("rb") as file:
        assert pickle.load(file)["adversarial"] == "wgan-gp"
    observed = []
    restore = train._restore_critic_checkpoint

    def check_scheduler(*args):
        restore(*args)
        assert args[4].state_dict() == state["scheduler_d"]

    monkeypatch.setattr(train, "_restore_critic_checkpoint", check_scheduler)

    def check_restored(*args, **kwargs):
        if not observed:
            _, critic, _, optimizer_d = args[:4]
            for name, tensor in critic.state_dict().items():
                torch.testing.assert_close(
                    tensor, state["model_d"][name], rtol=0, atol=0
                )
            restored = optimizer_d.state_dict()
            assert restored["param_groups"] == state["optimizer_d"]["param_groups"]
            for key, values in restored["state"].items():
                for name, value in values.items():
                    torch.testing.assert_close(
                        value, state["optimizer_d"]["state"][key][name], rtol=0, atol=0
                    )
            observed.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(train, "wgan_gp_step", check_restored)
    validation = run_tiny_training(tmp_path, monkeypatch, resume=checkpoint, epochs=2)
    assert observed
    history = pd.read_csv(tmp_path / "run/training_history.csv")
    assert history["epoch"].tolist() == [1, 2]
    assert {
        "train_loss",
        "val_loss",
        "val_normalized_mae",
        "val_normalized_ssim",
    } <= set(history.columns)
    assert (tmp_path / "run/epoch_2_checkpoint/critic.pt").is_file()
    # The inference loader must select only the generator, never pytorch_model_1.bin.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wavedl-test",
            "--checkpoint",
            str(tmp_path / "run/best_checkpoint"),
            "--data_path",
            str(validation),
            "--workers",
            "0",
            "--output_dir",
            str(tmp_path / "results"),
        ],
    )
    inference.main()
    assert "OPENFWI NORMALIZED METRICS" in capsys.readouterr().out


def test_missing_critic_state_fails_resume(tmp_path):
    with pytest.raises(ValueError, match="requires critic"):
        train._restore_critic_checkpoint(tmp_path, None, None, None, None)


@pytest.mark.slow
def test_real_velocitygan_single_step():
    from wavedl.models import build_model

    model = build_model(
        "velocitygan",
        in_shape=(1000, 70),
        out_size=1,
        in_channels=5,
        out_shape=(1, 70, 70),
    )
    critic = model.build_discriminator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizer_d = torch.optim.AdamW(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
    loss = wgan_gp_step(
        model,
        critic,
        optimizer,
        optimizer_d,
        OpenFWIPixelLoss(),
        torch.randn(2, 5, 1000, 70),
        torch.randn(2, 1, 70, 70),
        0,
        1,
        options(),
        BackwardOnly(),
    )
    assert torch.isfinite(loss)
    assert optimizer.state and optimizer_d.state
