import numpy as np
import torch

from vps.adapter import AdaptedEmbedder, ArcFace, embed, train_adapter


def test_ratio_zero_is_frozen_clip():
    f = torch.randn(8, 512)
    out = AdaptedEmbedder(ratio=0.0)(f)
    assert torch.allclose(out, torch.nn.functional.normalize(f, dim=-1), atol=1e-6)


def test_output_is_unit_norm_and_bottleneck_size():
    m = AdaptedEmbedder(dim=512, reduction=4, ratio=0.2)
    assert m.adapter.fc[0].out_features == 128
    out = embed(m, np.random.randn(10, 512).astype(np.float32))
    assert out.shape == (10, 512)
    assert np.allclose(np.linalg.norm(out, axis=1), 1, atol=1e-5)


def test_arcface_initial_loss_near_log_classes():
    torch.manual_seed(0)
    z = torch.randn(512, 64)
    y = torch.randint(0, 100, (512,))
    # With a small scale the logits are ~0 at init, so cross-entropy is ~ln(C).
    # (With s=30 random cosines are amplified and the initial loss sits above ln(C).)
    loss = ArcFace(64, 100, s=1.0, m=0.0)(z, y).item()
    assert abs(loss - np.log(100)) < 0.5


def test_training_reduces_loss_on_separable_data():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(20, 64))
    y = np.repeat(np.arange(20), 10)
    x = (centres[y] + 0.3 * rng.normal(size=(200, 64))).astype(np.float32)
    _, log = train_adapter(x, y, ratio=0.5, epochs=15, batch_size=32, seed=0)
    assert log[-1]["train_loss"] < 0.5 * log[0]["train_loss"]


def test_training_accepts_multiple_cached_views():
    rng = np.random.default_rng(1)
    centres = rng.normal(size=(10, 64))
    y = np.repeat(np.arange(10), 8)
    x = (centres[y][:, None] + 0.3 * rng.normal(size=(80, 3, 64))).astype(np.float32)
    _, log = train_adapter(x, y, ratio=0.5, epochs=10, batch_size=16, seed=0)
    assert log[-1]["train_loss"] < log[0]["train_loss"]
