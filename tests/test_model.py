"""
Unit tests for STELLA model components.

Run with: pytest tests/test_model.py -v

Tests verify shape correctness, forward pass stability, and
parameter counts within expected range.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.models.temporal_mamba import S3MBlock, TemporalMamba
from src.models.tokenizer import DualStreamTokenizer, TemporalPatchEmbed, SpectralBandEncoder
from src.models.spatial_transformer import ChannelSpatialTransformer
from src.models.fusion import GatedSpectralFusion
from src.models.stella import STELLA, STELLAConfig, build_stella
from src.models.baselines import EEGNet, ShallowConvNet, DeepConvNet, VanillaTransformerEEG
from src.models.heads import LinearProbe, MLPHead, ProjectionHead


@pytest.fixture(autouse=True)
def fix_seed():
    set_seed(42)


# -----------------------------------------------------------------------
# S3M / Mamba block
# -----------------------------------------------------------------------

class TestS3MBlock:
    def test_output_shape(self):
        block = S3MBlock(d_model=64, d_state=8, d_conv=4, expand=2)
        x = torch.randn(2, 20, 64)
        y = block(x)
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    def test_no_nan(self):
        block = S3MBlock(d_model=64, d_state=8)
        x = torch.randn(2, 50, 64)
        y = block(x)
        assert not torch.isnan(y).any(), "NaN in S3MBlock output"

    def test_gradient_flow(self):
        block = S3MBlock(d_model=32, d_state=4)
        x = torch.randn(1, 10, 32, requires_grad=True)
        y = block(x).mean()
        y.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_temporal_mamba_stack(self):
        mamba = TemporalMamba(d_model=64, n_layers=2, d_state=8)
        x = torch.randn(2, 30, 64)
        y = mamba(x)
        assert y.shape == x.shape


# -----------------------------------------------------------------------
# Tokenizer
# -----------------------------------------------------------------------

class TestTokenizer:
    def test_temporal_patch_embed_shape(self):
        embed = TemporalPatchEmbed(n_channels=64, patch_size=40, patch_stride=20, d_model=128)
        x = torch.randn(2, 64, 640)
        tokens = embed(x)
        assert tokens.shape[-1] == 128
        assert tokens.ndim == 3   # (B, P, D)

    def test_spectral_encoder_shape(self):
        enc = SpectralBandEncoder(n_channels=64, sfreq=160.0, d_spectral=128, n_fft=256)
        x = torch.randn(2, 64, 640)
        tokens = enc(x)
        assert tokens.shape == (2, 5, 128)   # 5 EEG bands

    def test_dual_stream_tokenizer(self):
        tok = DualStreamTokenizer(n_channels=64, sfreq=160.0, d_model=128)
        x = torch.randn(2, 64, 640)
        t_tokens, s_tokens = tok(x)
        assert t_tokens.shape[-1] == 128
        assert s_tokens.shape == (2, 5, 128)

    def test_no_nan_in_tokens(self):
        tok = DualStreamTokenizer(n_channels=32, sfreq=160.0, d_model=64)
        x = torch.randn(1, 32, 640)
        t, s = tok(x)
        assert not torch.isnan(t).any()
        assert not torch.isnan(s).any()


# -----------------------------------------------------------------------
# Spatial Transformer
# -----------------------------------------------------------------------

class TestSpatialTransformer:
    def test_output_shape(self):
        st = ChannelSpatialTransformer(d_model=64, n_heads=4, n_layers=2, n_channels=30)
        x = torch.randn(4, 30, 64)   # (N, C, D)
        y = st(x)
        assert y.shape == x.shape

    def test_no_nan(self):
        st = ChannelSpatialTransformer(d_model=64, n_heads=4, n_layers=1, n_channels=22)
        x = torch.randn(2, 22, 64)
        y = st(x)
        assert not torch.isnan(y).any()


# -----------------------------------------------------------------------
# Fusion
# -----------------------------------------------------------------------

class TestFusion:
    def test_gated_fusion_shape(self):
        fusion = GatedSpectralFusion(d_model=64, n_heads=4)
        temporal = torch.randn(2, 20, 64)
        spectral = torch.randn(2, 5, 64)
        out = fusion(temporal, spectral)
        assert out.shape == temporal.shape

    def test_gated_fusion_no_nan(self):
        fusion = GatedSpectralFusion(d_model=64, n_heads=4)
        t = torch.randn(2, 10, 64)
        s = torch.randn(2, 5, 64)
        out = fusion(t, s)
        assert not torch.isnan(out).any()


# -----------------------------------------------------------------------
# Full STELLA model
# -----------------------------------------------------------------------

class TestSTELLA:
    @pytest.fixture
    def small_cfg(self):
        return STELLAConfig(
            n_channels=22, sfreq=160.0, segment_len=320,
            patch_size=20, patch_stride=10, d_model=64,
            spatial_layers=1, spatial_heads=4,
            mamba_layers=1, mamba_d_state=4,
        )

    def test_forward_shape(self, small_cfg):
        model = build_stella(small_cfg)
        x = torch.randn(2, 22, 320)
        repr = model(x)
        assert repr.shape == (2, 64)

    def test_pretrain_forward(self, small_cfg):
        model = build_stella(small_cfg)
        x1 = torch.randn(3, 22, 320)
        x2 = torch.randn(3, 22, 320)
        labels = torch.randint(0, 4, (3,))
        domain_ids = torch.randint(0, 10, (3,))
        out = model.pretrain_forward(x1, x2, labels, domain_ids)
        assert "z1" in out
        assert out["z1"].shape == (3, small_cfg.proj_dim)
        assert not torch.isnan(out["z1"]).any()

    def test_no_nan_on_forward(self, small_cfg):
        model = build_stella(small_cfg)
        x = torch.randn(2, 22, 320)
        y = model(x)
        assert not torch.isnan(y).any()

    def test_freeze_encoder(self, small_cfg):
        model = build_stella(small_cfg)
        model.freeze_encoder()
        for p in model.encoder.parameters():
            assert not p.requires_grad

    def test_parameter_count_range(self, small_cfg):
        model = build_stella(small_cfg)
        counts = model.count_parameters()
        total = counts["encoder_total"]
        assert total > 100_000, f"Encoder too small: {total}"
        # Full config should be < 12M
        full_cfg = STELLAConfig()
        full_model = build_stella(full_cfg)
        full_total = full_model.count_parameters()["encoder_total"]
        assert full_total < 12_000_000, f"Encoder too large: {full_total:,}"

    def test_downstream_head(self, small_cfg):
        model = build_stella(small_cfg)
        head = LinearProbe(64, 4)
        model.attach_downstream_head(head)
        x = torch.randn(2, 22, 320)
        logits = model(x)
        assert logits.shape == (2, 4)

    def test_backward_pass(self, small_cfg):
        model = build_stella(small_cfg)
        head = LinearProbe(64, 4)
        model.attach_downstream_head(head)
        x = torch.randn(2, 22, 320)
        logits = model(x)
        loss = logits.mean()
        loss.backward()
        # Check that gradients exist
        has_grad = any(
            p.grad is not None for p in model.encoder.parameters()
        )
        assert has_grad, "No gradients in encoder during backward pass"


# -----------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------

class TestBaselines:
    @pytest.fixture
    def common_kwargs(self):
        return dict(n_channels=22, n_classes=4, T=320)

    def test_eegnet_forward(self, common_kwargs):
        model = EEGNet(**common_kwargs, sfreq=160.0)
        x = torch.randn(2, 22, 320)
        out = model(x)
        assert out.shape == (2, 4)

    def test_shallow_convnet(self, common_kwargs):
        model = ShallowConvNet(**common_kwargs)
        x = torch.randn(2, 22, 320)
        assert model(x).shape == (2, 4)

    def test_deep_convnet(self, common_kwargs):
        model = DeepConvNet(**common_kwargs)
        x = torch.randn(2, 22, 320)
        assert model(x).shape == (2, 4)

    def test_vanilla_transformer(self, common_kwargs):
        model = VanillaTransformerEEG(**common_kwargs, d_model=64, n_heads=4, n_layers=2)
        x = torch.randn(2, 22, 320)
        assert model(x).shape == (2, 4)


# -----------------------------------------------------------------------
# Loss functions
# -----------------------------------------------------------------------

class TestLosses:
    def test_ntxent(self):
        from src.losses.contrastive import SubjectAwareNTXentLoss
        loss_fn = SubjectAwareNTXentLoss(temperature=0.1)
        z1 = torch.randn(4, 32)
        z2 = torch.randn(4, 32)
        import torch.nn.functional as F
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        loss = loss_fn(z1, z2)
        assert loss.item() >= 0

    def test_vicreg(self):
        from src.losses.consistency import VICRegLoss
        loss_fn = VICRegLoss()
        z1 = torch.randn(8, 32)
        z2 = torch.randn(8, 32)
        loss = loss_fn(z1, z2)
        assert loss.item() >= 0

    def test_mmd(self):
        from src.losses.alignment import MMDLoss
        loss_fn = MMDLoss()
        src = torch.randn(8, 32)
        tgt = torch.randn(8, 32)
        loss = loss_fn(src, tgt)
        assert loss.item() >= 0

    def test_combined_loss(self):
        from src.losses.combined import STELLAPretrainLoss
        loss_fn = STELLAPretrainLoss()
        model_output = {
            "z1": torch.randn(4, 32), "z2": torch.randn(4, 32),
            "aux_logits": torch.randn(4, 4),
            "stc_temporal": torch.randn(4, 16),
            "stc_spectral": torch.randn(4, 16),
            "repr1": torch.randn(4, 64),
        }
        import torch.nn.functional as F
        model_output["z1"] = F.normalize(model_output["z1"], dim=-1)
        model_output["z2"] = F.normalize(model_output["z2"], dim=-1)

        labels = torch.randint(0, 4, (4,))
        domain_ids = torch.randint(0, 10, (4,))
        total, comps = loss_fn(model_output, labels, domain_ids)
        assert total.item() >= 0
        assert "contrastive" in comps
