# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""DASO-Det fusion modules for SAR--optical object-level change detection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .conv import Conv

__all__ = (
    "DASOSelect",
    "GateHead",
    "build_hermitian_low_frequency_mask",
    "SCIMV2",
    "HybridSpatialBlock",
    "CRCMLiteV2",
    "SMDMLiteV2",
)


def _unpack_pair(first, second, module_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accept either two tensors or one two-element list/tuple."""
    if second is None:
        if not isinstance(first, (list, tuple)) or len(first) != 2:
            raise ValueError(f"{module_name} expects exactly two inputs ordered as [SAR, Optical]")
        first, second = first
    return first, second






class DASOSelect(nn.Module):
    """Expose one calibrated SMDM output as an explicit YAML graph branch."""

    def __init__(self, index: int):
        """Select index 0 (SAR) or 1 (Optical) from the SMDM output pair."""
        super().__init__()
        if index not in {0, 1}:
            raise ValueError(f"DASOSelect index must be 0 (SAR) or 1 (Optical), but got {index}")
        self.index = index

    def forward(self, x: Union[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Return one tensor from a two-element calibrated feature pair."""
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("DASOSelect expects the two outputs returned by SMDM")
        return x[self.index]


class GateHead(nn.Module):
    """Full-channel gate head whose terminal layer emits unconstrained linear logits."""

    def __init__(self, in_channels: int, out_channels: int):
        """Build a spatial feature extractor followed by a bias-enabled linear projection."""
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(f"gate channels must be positive, but got {(in_channels, out_channels)}")
        self.features = Conv(in_channels, out_channels, 3)
        self.logits = nn.Conv2d(out_channels, out_channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return pre-sigmoid logits without terminal normalization or activation."""
        return self.logits(self.features(x))


def build_hermitian_low_frequency_mask(
    height: int,
    width: int,
    beta: float,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build an unshifted rectangular low-frequency mask closed under frequency conjugation.

    ``beta`` denotes the normalized full width of the pass band along each axis, so the inclusive
    cutoff is ``beta / 2``. Discrete symmetry requires an odd number of low-frequency bins around
    DC; consequently the exact covered area depends on the feature-map size.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, but got {(height, width)}")
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in (0, 1), but got {beta}")
    if height == 1 and width == 1:
        raise ValueError("a 1x1 spectrum cannot have a non-full low-frequency mask")

    cutoff = float(beta) / 2.0
    frequency_y = torch.fft.fftfreq(height, device=device)
    frequency_x = torch.fft.fftfreq(width, device=device)
    mask = (frequency_y.abs()[:, None] <= cutoff) & (frequency_x.abs()[None, :] <= cutoff)
    return mask[None, None].to(dtype=dtype)






class SCIMV2(nn.Module):
    """Ordered interaction with common-support-conditioned directional routing.

    SCIMV2 applies one shared projection to ``[SAR, Optical]`` and its reversal, decomposing the two embeddings into a
    common component and two non-negative order directions, and reconstructs the base embedding with
    two independent full-channel gates. Compact common-support evidence conditions those gates but is
    never appended to the output. Zero terminal gate logits give unit scales, so a new module exactly
    recovers its own base-order Concat projection up to floating-point roundoff. This is the
    repository's only SCIM implementation.
    """

    BASE_ORDER = ("SAR", "Optical")
    REVERSE_ORDER = ("Optical", "SAR")
    DEFAULT_CONFIG = {"state_ratio": 0.25}

    def __init__(
        self,
        channels: int,
        config: Optional[Dict[str, Any]] = None,
        state_ratio: float = 0.25,
    ):
        """Build the shared ordered projection and common-support-conditioned direction gates."""
        super().__init__()
        if config is not None:
            config = dict(config)
            unknown = set(config) - set(self.DEFAULT_CONFIG)
            if unknown:
                raise ValueError(f"unknown SCIMV2 config keys: {sorted(unknown)}")
            state_ratio = {**self.DEFAULT_CONFIG, **config}["state_ratio"]
        if channels <= 0:
            raise ValueError(f"SCIMV2 channels must be positive, got {channels}")
        if not 0.0 < state_ratio <= 1.0:
            raise ValueError(f"SCIMV2 state_ratio must be in (0, 1], got {state_ratio}")
        self.channels = channels
        self.state_ratio = float(state_ratio)
        self.state_channels = max(1, int(round(channels * self.state_ratio)))
        state_channels = self.state_channels

        # Evaluate the shared 2C->C projection on a batch stack containing both input orders, guaranteeing
        # identical projection parameters and normalization statistics for H_base and H_reverse.
        self.ordered_projection = Conv(2 * channels, channels, 1)

        # Initialize routing context without shifting the RNG stream for the neck and OBB head.
        with torch.random.fork_rng(devices=[]):
            # In the full DASO path these inputs are the calibrated SMDMLiteV2 outputs. The SCIM-only
            # ablation intentionally applies the same operation to uncalibrated backbone features.
            self.common_support = nn.Sequential(
                Conv(channels, state_channels, 1),
                Conv(state_channels, state_channels, 3, g=state_channels),
            )
            self.common_context = Conv(channels, state_channels, 1)
            # One shared low-dimensional encoder is batch-applied to both full-channel directions.
            self.direction_context = Conv(channels, state_channels, 1)

            # X_g contains Q_common, Q_reverse, Q_base, and E_shared, each with r channels. Only the
            # terminal r->2C convolution is zero-initialized; the two C-channel gates do not compete.
            self.gate_features = nn.Sequential(
                Conv(4 * state_channels, state_channels, 1),
                Conv(state_channels, state_channels, 3, g=state_channels),
            )
            self.gate_logits = nn.Conv2d(state_channels, 2 * channels, 1, bias=True)
            nn.init.zeros_(self.gate_logits.weight)
            nn.init.zeros_(self.gate_logits.bias)

    @staticmethod
    def ordered_components(
        h_base: torch.Tensor, h_reverse: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return common, reverse-enhanced, and base-enhanced components of two ordered embeddings."""
        common = 0.5 * (h_base + h_reverse)
        delta = 0.5 * (h_reverse - h_base)
        d_reverse = torch.relu(delta)
        d_base = torch.relu(-delta)
        return common, d_reverse, d_base

    def direction_scales(self, state_logits: torch.Tensor) -> torch.Tensor:
        """Return identity-centered spatial scales over the complete open interval (0, 2)."""
        return 2.0 * torch.sigmoid(state_logits)

    def load_concat_projection_(self, concat_projection: nn.Module) -> "SCIMV2":
        """Copy a trained Ultralytics Concat-compression ``Conv`` into the shared ordered projection.

        This copies only the fusion projection. Calling code remains responsible for loading every
        other corresponding backbone, neck, and head tensor before claiming recovery of a trained
        Concat model.
        """
        if not isinstance(concat_projection, Conv):
            raise TypeError(
                "concat_projection must be the Ultralytics Conv immediately after Concat, "
                f"but got {type(concat_projection).__name__}"
            )
        source_state = concat_projection.state_dict()
        target_state = self.ordered_projection.state_dict()
        if source_state.keys() != target_state.keys() or any(
            source_state[key].shape != target_state[key].shape for key in source_state
        ):
            raise ValueError("Concat projection and SCIM ordered projection state shapes do not match")
        self.ordered_projection.load_state_dict(source_state, strict=True)
        return self

    def forward(self, fs, fo: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fuse an ordered ``[SAR, Optical]`` pair into one state-conditioned feature map."""
        fs, fo = _unpack_pair(fs, fo, "SCIMV2")
        if fs.shape != fo.shape:
            raise ValueError(f"SCIMV2 inputs must have identical shapes, got {fs.shape} and {fo.shape}")
        if fs.ndim != 4 or fs.shape[1] != self.channels:
            raise ValueError(f"SCIMV2 expects BCHW tensors with {self.channels} channels, got {fs.shape}")

        # Audited current control: yaml indices [16, 7]/[18, 9]/[20, 11] concatenate [SAR, Optical].
        x_base = torch.cat((fs, fo), dim=1)
        x_reverse = torch.cat((fo, fs), dim=1)
        ordered = self.ordered_projection(torch.cat((x_base, x_reverse), dim=0))
        h_base, h_reverse = torch.chunk(ordered, 2, dim=0)
        common, d_reverse, d_base = self.ordered_components(h_base, h_reverse)

        shared_evidence = self.common_support(torch.relu(fs) * torch.relu(fo))
        q_common = self.common_context(common)
        direction_context = self.direction_context(torch.cat((d_reverse, d_base), dim=0))
        q_reverse, q_base = torch.chunk(direction_context, 2, dim=0)
        gate_features = self.gate_features(torch.cat((q_common, q_reverse, q_base, shared_evidence), dim=1))
        gates = self.direction_scales(self.gate_logits(gate_features))
        g_reverse, g_base = torch.chunk(gates, 2, dim=1)

        return common - g_reverse * d_reverse + g_base * d_base


class HybridSpatialBlock(nn.Module):
    """Mix all inputs, split spatial processing into dense/depthwise paths, then fuse."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        dense_ratio: float = 0.5,
        terminal_act: bool = True,
    ):
        """Build an efficient spatial block without discarding any raw input channel group."""
        super().__init__()
        if min(in_channels, out_channels, hidden_channels) <= 0:
            raise ValueError(
                "HybridSpatialBlock channels must be positive, "
                f"but got {(in_channels, out_channels, hidden_channels)}"
            )
        if not 0.0 < dense_ratio < 1.0:
            raise ValueError(f"dense_ratio must be in (0, 1), but got {dense_ratio}")
        if hidden_channels < 2:
            raise ValueError("HybridSpatialBlock hidden_channels must be at least 2")

        dense_channels = max(1, min(hidden_channels - 1, int(round(hidden_channels * dense_ratio))))
        light_channels = hidden_channels - dense_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.dense_channels = dense_channels
        self.light_channels = light_channels
        self.dense_ratio = float(dense_ratio)

        # The full-channel 1x1 mix precedes the split, so both spatial paths can use
        # information from every input channel rather than a fixed raw-channel subset.
        self.reduce = Conv(in_channels, hidden_channels, 1)
        self.dense_spatial = Conv(dense_channels, dense_channels, 3)
        self.light_spatial = Conv(light_channels, light_channels, 3, g=light_channels)
        self.fuse = Conv(hidden_channels, out_channels, 1, act=terminal_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply mixed dense/depthwise spatial modeling."""
        mixed = self.reduce(x)
        dense, light = torch.split(mixed, (self.dense_channels, self.light_channels), dim=1)
        return self.fuse(torch.cat((self.dense_spatial(dense), self.light_spatial(light)), dim=1))


class CRCMLiteV2(nn.Module):
    """Efficient CRCM that retains independent manifestation and discrepancy gates."""

    def __init__(self, channels: int, diagnostic_dim: int):
        """Build diagnostic-space spatial trunks and decode full-channel gate logits."""
        super().__init__()
        if channels <= 0 or diagnostic_dim <= 0:
            raise ValueError(f"CRCM-Lite channels must be positive, but got {(channels, diagnostic_dim)}")
        self.channels = channels
        self.diagnostic_dim = diagnostic_dim
        self.prior = HybridSpatialBlock(2 * diagnostic_dim, diagnostic_dim, diagnostic_dim, dense_ratio=0.5)
        self.manifestation_features = HybridSpatialBlock(
            2 * diagnostic_dim, diagnostic_dim, diagnostic_dim, dense_ratio=0.5
        )
        self.discrepancy_features = HybridSpatialBlock(
            2 * diagnostic_dim, diagnostic_dim, diagnostic_dim, dense_ratio=0.5
        )
        self.manifestation_logits = nn.Conv2d(diagnostic_dim, channels, 1, bias=True)
        self.discrepancy_logits = nn.Conv2d(diagnostic_dim, channels, 1, bias=True)

    def forward(
        self, us: torch.Tensor, uo: torch.Tensor, rd: torch.Tensor, rm: torch.Tensor
    ) -> torch.Tensor:
        """Return the full-channel complementary gate ``Gm * (1 - Gd)``."""
        response_prior = self.prior(torch.cat((us + uo, us * uo), dim=1))
        g_m = torch.sigmoid(
            self.manifestation_logits(self.manifestation_features(torch.cat((response_prior, rm), dim=1)))
        )
        g_d = torch.sigmoid(
            self.discrepancy_logits(self.discrepancy_features(torch.cat((response_prior, rd), dim=1)))
        )
        return g_m * (1.0 - g_d)


class SMDMLiteV2(nn.Module):
    """SMDM with full feature bases and an efficient configurable diagnostic path."""

    DEFAULT_CONFIG = {
        "diagnostic_dim": 64,
        "beta": 0.125,
        "eps": 1e-6,
        "fft_norm": "ortho",
        "use_rd": True,
        "use_rm": True,
        "residual_scale": 0.1,
    }

    def __init__(
        self,
        channels: int,
        config: Optional[Dict[str, Any]] = None,
        diagnostic_dim: int = 64,
        beta: float = 0.125,
        eps: float = 1e-6,
        fft_norm: str = "ortho",
        use_rd: bool = True,
        use_rm: bool = True,
        residual_scale: float = 0.1,
    ):
        """Initialize preserved RD/RM/CRCM semantics with efficient spatial operators."""
        super().__init__()
        if config is not None:
            config = dict(config)
            unknown = set(config) - set(self.DEFAULT_CONFIG)
            if unknown:
                raise ValueError(f"unknown SMDMLiteV2 config keys: {sorted(unknown)}")
            values = {**self.DEFAULT_CONFIG, **config}
            diagnostic_dim = values["diagnostic_dim"]
            beta = values["beta"]
            eps = values["eps"]
            fft_norm = values["fft_norm"]
            use_rd = values["use_rd"]
            use_rm = values["use_rm"]
            residual_scale = values["residual_scale"]
        if channels <= 0 or diagnostic_dim <= 0:
            raise ValueError(f"SMDM-Lite channels must be positive, but got {(channels, diagnostic_dim)}")
        if diagnostic_dim > channels:
            raise ValueError(f"diagnostic_dim cannot exceed channels, got {(diagnostic_dim, channels)}")
        if not 0.0 < beta < 1.0:
            raise ValueError(f"beta must be in (0, 1), but got {beta}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, but got {eps}")
        if fft_norm not in {"forward", "backward", "ortho"}:
            raise ValueError(f"unsupported FFT normalization {fft_norm!r}")
        if residual_scale <= 0.0:
            raise ValueError(f"residual_scale must be positive, but got {residual_scale}")

        self.channels = channels
        self.diagnostic_dim = int(diagnostic_dim)
        self.beta = float(beta)
        self.eps = float(eps)
        self.fft_norm = fft_norm
        self.use_rd = bool(use_rd)
        self.use_rm = bool(use_rm)
        self.residual_scale = float(residual_scale)
        self.register_buffer("_cached_lf_mask", torch.empty(0, dtype=torch.bool), persistent=False)
        self._cached_lf_mask_meta = None

        self.proj_s = Conv(channels, channels, 1)
        self.proj_o = Conv(channels, channels, 1)
        # Linear encoders retain signed responses for spectral decomposition.
        self.enc_s = nn.Conv2d(channels, self.diagnostic_dim, 1, bias=True)
        self.enc_o = nn.Conv2d(channels, self.diagnostic_dim, 1, bias=True)
        self.rd_branch = (
            HybridSpatialBlock(2 * self.diagnostic_dim, self.diagnostic_dim, self.diagnostic_dim, dense_ratio=0.5)
            if self.use_rd
            else None
        )
        self.rm_branch = (
            HybridSpatialBlock(2 * self.diagnostic_dim, self.diagnostic_dim, self.diagnostic_dim, dense_ratio=0.5)
            if self.use_rm
            else None
        )
        self.crcm = CRCMLiteV2(channels, self.diagnostic_dim)
        self.delta_features = HybridSpatialBlock(
            3 * self.diagnostic_dim, self.diagnostic_dim, self.diagnostic_dim, dense_ratio=0.5
        )
        self.delta_s = nn.Conv2d(self.diagnostic_dim, channels, 1, bias=True)
        self.delta_o = nn.Conv2d(self.diagnostic_dim, channels, 1, bias=True)
        for decoder in (self.delta_s, self.delta_o):
            nn.init.zeros_(decoder.weight)
            nn.init.zeros_(decoder.bias)

    @staticmethod
    def low_frequency_mask(
        height: int,
        width: int,
        beta: float,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return an unshifted Hermitian low-frequency mask."""
        return build_hermitian_low_frequency_mask(height, width, beta, device=device, dtype=dtype)

    def _get_low_frequency_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Return a cached boolean low-frequency mask for one scale and device."""
        meta = (height, width, device.type, device.index, self.beta)
        if self._cached_lf_mask_meta != meta or self._cached_lf_mask.numel() == 0:
            self._cached_lf_mask = self.low_frequency_mask(
                height, width, self.beta, device=device, dtype=torch.float32
            ).bool()
            self._cached_lf_mask_meta = meta
        return self._cached_lf_mask

    def _spectra(self, us: torch.Tensor, uo: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute the two diagnostic spectra separately in FP32."""
        zs = torch.fft.fft2(us.float(), dim=(-2, -1), norm=self.fft_norm)
        zo = torch.fft.fft2(uo.float(), dim=(-2, -1), norm=self.fft_norm)
        amplitude_s, amplitude_o = torch.abs(zs), torch.abs(zo)
        phase_s = zs / (amplitude_s + self.eps)
        phase_o = zo / (amplitude_o + self.eps)
        return amplitude_s, amplitude_o, phase_s, phase_o

    def _ifft2_real(self, amplitude: torch.Tensor, phase: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Reconstruct one diagnostic feature in FP32 and restore the feature dtype."""
        return torch.fft.ifft2(amplitude * phase, dim=(-2, -1), norm=self.fft_norm).real.to(dtype=dtype)

    def _state_diagnostic(
        self,
        us: torch.Tensor,
        uo: torch.Tensor,
        amplitude_s: torch.Tensor,
        amplitude_o: torch.Tensor,
        phase_s: torch.Tensor,
        phase_o: torch.Tensor,
    ) -> torch.Tensor:
        """Compute state discrepancy via cross-modal phase recombination."""
        ref_s = self._ifft2_real(amplitude_s, phase_o, us.dtype)
        ref_o = self._ifft2_real(amplitude_o, phase_s, uo.dtype)
        return self.rd_branch(torch.cat((torch.abs(us - ref_s), torch.abs(uo - ref_o)), dim=1))

    def _manifestation_diagnostic(
        self,
        us: torch.Tensor,
        uo: torch.Tensor,
        amplitude_s: torch.Tensor,
        amplitude_o: torch.Tensor,
        phase_s: torch.Tensor,
        phase_o: torch.Tensor,
    ) -> torch.Tensor:
        """Compute manifestation interference via Hermitian low-frequency amplitude exchange."""
        mask = self._get_low_frequency_mask(*us.shape[-2:], us.device)
        mixed_s = torch.where(mask, amplitude_o, amplitude_s)
        mixed_o = torch.where(mask, amplitude_s, amplitude_o)
        ref_s = self._ifft2_real(mixed_s, phase_s, us.dtype)
        ref_o = self._ifft2_real(mixed_o, phase_o, uo.dtype)
        return self.rm_branch(torch.cat((torch.abs(us - ref_s), torch.abs(uo - ref_o)), dim=1))

    def forward(self, fs, fo: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calibrate full-dimensional modality bases using efficient diagnostic controls."""
        fs, fo = _unpack_pair(fs, fo, "SMDMLiteV2")
        if fs.shape != fo.shape:
            raise ValueError(f"SMDMLiteV2 inputs must have identical shapes, but got {fs.shape} and {fo.shape}")
        if fs.ndim != 4 or fs.shape[1] != self.channels:
            raise ValueError(f"SMDMLiteV2 expects BCHW tensors with {self.channels} channels, but got {fs.shape}")

        fs, fo = self.proj_s(fs), self.proj_o(fo)
        us, uo = self.enc_s(fs), self.enc_o(fo)
        if self.use_rd or self.use_rm:
            amplitude_s, amplitude_o, phase_s, phase_o = self._spectra(us, uo)
        rd = (
            self._state_diagnostic(us, uo, amplitude_s, amplitude_o, phase_s, phase_o)
            if self.use_rd
            else torch.zeros_like(us)
        )
        rm = (
            self._manifestation_diagnostic(us, uo, amplitude_s, amplitude_o, phase_s, phase_o)
            if self.use_rm
            else torch.zeros_like(us)
        )

        gate = self.crcm(us, uo, rd, rm)
        delta_latent = self.delta_features(torch.cat((us, uo, rm), dim=1))
        delta_s, delta_o = self.delta_s(delta_latent), self.delta_o(delta_latent)
        return fs + self.residual_scale * gate * delta_s, fo + self.residual_scale * gate * delta_o
