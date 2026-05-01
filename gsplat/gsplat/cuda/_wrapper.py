# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
from dataclasses import dataclass
from enum import IntEnum
from abc import ABC
from typing import Any, Callable, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal
from gsplat._helper import assert_shape

# Lidar bases used to come from gsplat.cuda._lidar, which has been removed
# along with the rest of the CUDA backend. Stage 1 of the MPS port wires the
# 3DGS / 2DGS / 3DGUT public functions to their existing torch references and
# is expected to delete the (now-dead) lidar-only API surface as part of that
# work. Until then, these placeholder bases keep the module importable.
class _LidarStubBase:  # noqa: D401
    """Inert base class kept alive while lidar paths are being removed."""


SpinningDirection = _LidarStubBase
LidarModelParameters = _LidarStubBase
RowOffsetStructuredSpinningLidarModelParameters = _LidarStubBase
RowOffsetStructuredSpinningLidarModelParametersExtBase = _LidarStubBase
FOVBase = _LidarStubBase

ExternalDistortionModelMeta = Literal["bivariate-windshield"]
CameraModel = Literal["pinhole", "ortho", "fisheye", "ftheta", "lidar"]


def _not_implemented(name: str) -> Callable:
    """Return a callable that raises when invoked.

    Used in place of the deleted CUDA-backed dispatch helpers. Stage 1 of the
    MPS port replaces every call site with a torch fallback or Metal kernel.
    """

    def _stub(*args, **kwargs):
        raise NotImplementedError(
            f"gsplat op '{name}' is not yet wired to the MPS / torch backend. "
            "It will be implemented in Stage 1 of the MPS port."
        )

    return _stub


def _make_lazy_cuda_func(name: str) -> Callable:
    return _not_implemented(name)


def _make_lazy_cuda_cls(name: str) -> Any:
    class _Stub:
        __name__ = name

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError(
                f"gsplat class '{name}' is not yet wired to the MPS / torch "
                "backend. It will be implemented in Stage 1 of the MPS port."
            )

    return _Stub


def _make_lazy_cuda_obj(name: str) -> Any:
    raise NotImplementedError(
        f"gsplat constant '{name}' is not yet wired to the MPS / torch "
        "backend. It will be implemented in Stage 1 of the MPS port."
    )


class RollingShutterType(IntEnum):
    ROLLING_TOP_TO_BOTTOM = 0
    ROLLING_LEFT_TO_RIGHT = 1
    ROLLING_BOTTOM_TO_TOP = 2
    ROLLING_RIGHT_TO_LEFT = 3
    GLOBAL = 4


class FThetaPolynomialType(IntEnum):
    PIXELDIST_TO_ANGLE = 0
    ANGLE_TO_PIXELDIST = 1


# Python dataclasses replacing the deleted C++ torch::class_<> bindings. The
# original CUDA classes carried extra accelerator-side state; the MPS torch
# fallback only needs the public knobs callers actually set.
@dataclass
class UnscentedTransformParameters:
    """Tunables for the Unscented Transform projection used by 3DGUT."""

    alpha: float = 1.0
    beta: float = 2.0
    kappa: float = 0.0
    in_image_margin_factor: float = 0.1
    require_all_sigma_points_valid: bool = False


@dataclass
class FThetaCameraDistortionParameters:
    """F-theta lens distortion polynomials."""

    reference_poly: int = 1  # ExternalDistortionReferencePolynomial.FORWARD
    pixeldist_to_angle_poly: Optional[Tensor] = None
    angle_to_pixeldist_poly: Optional[Tensor] = None
    max_angle: float = 0.0
    linear_cde: Optional[Tensor] = None


class ExternalDistortionModelParameters(ABC):
    """Base class for external distortion model parameters.

    All concrete external distortion models (e.g. BivariateWindshieldModelParameters)
    should inherit from this class so that the rendering API can accept any
    distortion model through a single type-erased parameter.
    """


class ExternalDistortionReferencePolynomial(IntEnum):
    FORWARD = 1
    BACKWARD = 2


class BivariateWindshieldModelParameters(ExternalDistortionModelParameters):
    """Thin wrapper around the CUDA BivariateWindshieldModelParameters class.

    torch::Library bindings does not allow standalone constants. This
    wrapper fetches MAX_ORDER and MAX_COEFFS from the C++ static getters
    and exposes them as class-level attributes, preserving the existing
    attribute-access calling convention.
    """

    _cuda_cls = None
    MAX_ORDER: int = 5  # default, overriden by C++ value
    MAX_COEFFS: int = 21  # default, overriden by C++ value

    @classmethod
    def _ensure_cuda_cls(cls):
        if cls._cuda_cls is None:
            cls._cuda_cls = _make_lazy_cuda_cls("BivariateWindshieldModelParameters")
            cls.MAX_ORDER = cls._cuda_cls.get_max_order()
            cls.MAX_COEFFS = cls._cuda_cls.get_max_coeffs()

    def __new__(cls):
        cls._ensure_cuda_cls()
        return cls._cuda_cls()


def has_camera_wrappers():
    from ._backend import _C

    # PyTorch will throw a RuntimeError if the class is not registered
    # but that's okay in this case because we're just checking if it exists
    try:
        return hasattr(torch.classes.gsplat, "BaseCameraModel")
    except RuntimeError:
        return False


def has_2dgs():
    from ._backend import _C

    return hasattr(torch.ops.gsplat, "projection_2dgs_fused_fwd")


def has_3dgs():
    from ._backend import _C

    return hasattr(torch.ops.gsplat, "projection_ewa_simple_fwd")


def has_3dgut():
    from ._backend import _C

    return hasattr(torch.ops.gsplat, "projection_ut_3dgs_fused")


def has_adam():
    from ._backend import _C

    return hasattr(torch.ops.gsplat, "adam")


def has_reloc():
    from ._backend import _C

    return hasattr(torch.ops.gsplat, "relocation")


def create_camera_model(
    camera_model: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    principal_points: Optional[Tensor] = None,
    focal_lengths: Optional[Tensor] = None,
    radial_coeffs: Optional[Tensor] = None,
    tangential_coeffs: Optional[Tensor] = None,
    thin_prism_coeffs: Optional[Tensor] = None,
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    external_distortion_coeffs: Optional[BivariateWindshieldModelParameters] = None,
    rs_type: RollingShutterType = RollingShutterType.GLOBAL,
    lidar_coeffs: Optional["RowOffsetStructuredSpinningLidarModelParametersExt"] = None,
):
    if camera_model == "lidar":
        assert (
            lidar_coeffs is not None
        ), "lidar_coeffs is required for lidar camera model"
        RowOffsetStructuredSpinningLidarModelCUDA = _make_lazy_cuda_cls(
            "RowOffsetStructuredSpinningLidarModel"
        )
        return RowOffsetStructuredSpinningLidarModelCUDA(lidar_coeffs.to_cpp())
    else:
        assert width is not None, "width is required for non-lidar camera models"
        assert height is not None, "height is required for non-lidar camera models"
        assert (
            principal_points is not None
        ), "principal_points is required for non-lidar camera models"
        BaseCameraModelCUDA = _make_lazy_cuda_cls("BaseCameraModel")
        return BaseCameraModelCUDA.create(
            width,
            height,
            camera_model,
            principal_points,
            focal_lengths,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            ftheta_coeffs,
            external_distortion_coeffs,
            rs_type,
        )


class FOV(FOVBase):
    @classmethod
    def from_base(cls, base: FOVBase) -> "FOV":
        return cls(start=base.start, span=base.span, direction=base.direction)

    def to_cpp(self):
        FOVCUDA = _make_lazy_cuda_cls("FOV")
        return FOVCUDA(start=self.start, span=self.span)


class RowOffsetStructuredSpinningLidarModelParametersExt(
    RowOffsetStructuredSpinningLidarModelParametersExtBase
):
    """Lidar camera parameters extended with acceleration structures"""

    def to_cpp(self) -> Any:
        """Convert to C++ custom class instance."""
        LidarParamsCUDA = _make_lazy_cuda_cls(
            "RowOffsetStructuredSpinningLidarModelParametersExt"
        )
        return LidarParamsCUDA(
            row_elevations_rad=self.row_elevations_rad.contiguous(),
            column_azimuths_rad=self.column_azimuths_rad.contiguous(),
            row_azimuth_offsets_rad=self.row_azimuth_offsets_rad.contiguous(),
            spinning_direction=self.spinning_direction.value,
            spinning_frequency_hz=self.spinning_frequency_hz,
            fov_vert_rad=FOV.from_base(self.fov_vert_rad).to_cpp(),
            fov_horiz_rad=FOV.from_base(self.fov_horiz_rad).to_cpp(),
            fov_eps_rad=self.fov_eps_rad,
            angles_to_columns_map=self.angles_to_columns_map,
            n_bins_azimuth=self.tiling.n_bins_azimuth,
            n_bins_elevation=self.tiling.n_bins_elevation,
            cdf_elevation=self.tiling.cdf_elevation.contiguous(),
            cdf_dense_ray_mask=self.tiling.cdf_dense_ray_mask.contiguous(),
            tiles_to_elements_map=self.tiling.tiles_to_elements_map.contiguous(),
            tiles_pack_info=self.tiling.tiles_pack_info.contiguous(),
        )


def world_to_cam(
    means: Tensor,  # [..., N, 3]
    covars: Tensor,  # [..., N, 3, 3]
    viewmats: Tensor,  # [..., C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """Transforms Gaussians from world to camera coordinate system.

    Args:
        means: Gaussian means. [..., N, 3]
        covars: Gaussian covariances. [..., N, 3, 3]
        viewmats: World-to-camera transformation matrices. [..., C, 4, 4]

    Returns:
        A tuple:

        - **Gaussian means in camera coordinate system**. [..., C, N, 3]
        - **Gaussian covariances in camera coordinate system**. [..., C, N, 3, 3]
    """
    from ._torch_impl import _world_to_cam

    warnings.warn(
        "world_to_cam() is removed from the CUDA backend as it's relatively easy to "
        "implement in PyTorch. Currently use the PyTorch implementation instead. "
        "This function will be completely removed in a future release.",
        DeprecationWarning,
    )
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == batch_dims + (N, 3), means.shape
    assert covars.shape == batch_dims + (N, 3, 3), covars.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    means = means.contiguous()
    covars = covars.contiguous()
    viewmats = viewmats.contiguous()
    return _world_to_cam(means, covars, viewmats)


def adam(
    param: Tensor,
    param_grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    valid: Tensor,
    lr: float,
    b1: float,
    b2: float,
    eps: float,
) -> None:
    _make_lazy_cuda_func("adam")(
        param, param_grad, exp_avg, exp_avg_sq, valid, lr, b1, b2, eps
    )


def spherical_harmonics(
    degrees_to_use: int,
    dirs: Tensor,  # [..., 3]
    coeffs: Tensor,  # [..., K, 3]
    masks: Optional[Tensor] = None,  # [...,]
) -> Tensor:
    """Computes spherical harmonics.

    Args:
        degrees_to_use: The degree to be used.
        dirs: Directions. [..., 3]
        coeffs: Coefficients. [..., K, 3]
        masks: Optional boolen masks to skip some computation. [...,] Default: None.

    Returns:
        Spherical harmonics. [..., 3]
    """
    assert (degrees_to_use + 1) ** 2 <= coeffs.shape[-2], coeffs.shape
    batch_dims = dirs.shape[:-1]
    assert dirs.shape == batch_dims + (3,), dirs.shape
    assert (
        (len(coeffs.shape) == len(batch_dims) + 2)
        and coeffs.shape[:-2] == batch_dims
        and coeffs.shape[-1] == 3
    ), coeffs.shape
    if masks is not None:
        assert masks.shape == batch_dims, masks.shape
        masks = masks.contiguous()

    from ._dispatch import select_backend, Backend

    if (
        select_backend(dirs, "spherical_harmonics") is Backend.METAL
        and coeffs.shape[-2] <= 25  # Metal kernel supports up to deg=4
    ):
        from ..mps._sh import spherical_harmonics_metal

        return spherical_harmonics_metal(
            degrees_to_use, dirs.contiguous(), coeffs.contiguous(), masks
        )

    from ._torch_impl import _spherical_harmonics

    out = _spherical_harmonics(degrees_to_use, dirs.contiguous(), coeffs.contiguous())
    if masks is not None:
        # CUDA kernel zeroed out rows where masks==False; preserve that
        # contract on the torch path so callers downstream see [..., 3]==0.
        out = torch.where(masks[..., None].to(torch.bool), out, torch.zeros_like(out))
    return out


def quat_scale_to_covar_preci(
    quats: Tensor,  # [..., 4],
    scales: Tensor,  # [..., 3],
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Converts quaternions and scales to covariance and precision matrices.

    Args:
        quats: Quaternions (No need to be normalized). [..., 4]
        scales: Scales. [..., 3]
        compute_covar: Whether to compute covariance matrices. Default: True. If False,
            the returned covariance matrices will be None.
        compute_preci: Whether to compute precision matrices. Default: True. If False,
            the returned precision matrices will be None.
        triu: If True, the return matrices will be upper triangular. Default: False.

    Returns:
        A tuple:

        - **Covariance matrices**. If `triu` is True the returned shape is [..., 6], otherwise [..., 3, 3].
        - **Precision matrices**. If `triu` is True the returned shape is [..., 6], otherwise [..., 3, 3].
    """
    batch_dims = quats.shape[:-1]
    assert quats.shape == batch_dims + (4,), quats.shape
    assert scales.shape == batch_dims + (3,), scales.shape
    quats = quats.contiguous()
    scales = scales.contiguous()
    from ._math import _quat_scale_to_covar_preci

    covars, precis = _quat_scale_to_covar_preci(
        quats,
        scales,
        compute_covar=compute_covar,
        compute_preci=compute_preci,
        triu=triu,
    )
    return covars if compute_covar else None, precis if compute_preci else None


def persp_proj(
    means: Tensor,  # [..., C, N, 3]
    covars: Tensor,  # [..., C, N, 3, 3]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """Perspective projection on Gaussians.
    DEPRECATED: please use `proj` with `ortho=False` instead.

    Args:
        means: Gaussian means. [..., C, N, 3]
        covars: Gaussian covariances. [..., C, N, 3, 3]
        Ks: Camera intrinsics. [..., C, 3, 3]
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **Projected means**. [..., C, N, 2]
        - **Projected covariances**. [..., C, N, 2, 2]
    """
    warnings.warn(
        "persp_proj is deprecated and will be removed in a future release. "
        "Use proj with ortho=False instead.",
        DeprecationWarning,
    )
    return proj(means, covars, Ks, width, height, ortho=False)


def proj(
    means: Tensor,  # [..., C, N, 3]
    covars: Tensor,  # [..., C, N, 3, 3]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    camera_model: CameraModel = "pinhole",
) -> Tuple[Tensor, Tensor]:
    """Projection of Gaussians (perspective or orthographic).

    Args:
        means: Gaussian means. [..., C, N, 3]
        covars: Gaussian covariances. [..., C, N, 3, 3]
        Ks: Camera intrinsics. [..., C, 3, 3]
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **Projected means**. [..., C, N, 2]
        - **Projected covariances**. [..., C, N, 2, 2]
    """
    assert (
        camera_model != "ftheta"
    ), "ftheta camera is only supported via UT, please set with_ut=True in the rasterization()"

    batch_dims = means.shape[:-3]
    C, N = means.shape[-3:-1]
    assert means.shape == batch_dims + (C, N, 3), means.shape
    assert covars.shape == batch_dims + (C, N, 3, 3), covars.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    means = means.contiguous()
    covars = covars.contiguous()
    Ks = Ks.contiguous()
    return _Proj.apply(means, covars, Ks, width, height, camera_model)


def fully_fused_projection(
    means: Tensor,  # [..., N, 3]
    covars: Optional[Tensor],  # [..., N, 6] or None
    quats: Optional[Tensor],  # [..., N, 4] or None
    scales: Optional[Tensor],  # [..., N, 3] or None
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
    calc_compensations: bool = False,
    camera_model: CameraModel = "pinhole",
    opacities: Optional[Tensor] = None,  # [..., N] or None
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Projects Gaussians to 2D.

    This function fuse the process of computing covariances
    (:func:`quat_scale_to_covar_preci()`), transforming to camera space (:func:`world_to_cam()`),
    and projection (:func:`proj()`).

    .. note::

        During projection, we ignore the Gaussians that are outside of the camera frustum.
        So not all the elements in the output tensors are valid. The output `radii` could serve as
        an indicator, in which zero radii means the corresponding elements are invalid in
        the output tensors and will be ignored in the next rasterization process. If `packed=True`,
        the output tensors will be packed into a flattened tensor, in which all elements are valid.
        In this case, a ``batch_ids` tensor and `camera_ids` tensor will be returned to indicate the
        batch, camera and gaussian indices of the packed flattened tensor, which is essentially following the
        COO sparse tensor format.

    .. note::

        This functions supports projecting Gaussians with either covariances or {quaternions, scales},
        which will be converted to covariances internally in a fused CUDA kernel. Either `covars` or
        {`quats`, `scales`} should be provided.

    Args:
        means: Gaussian means. [..., N, 3]
        covars: Gaussian covariances (flattened upper triangle). [..., N, 6] Optional.
        quats: Quaternions (No need to be normalized). [..., N, 4] Optional.
        scales: Scales. [..., N, 3] Optional.
        viewmats: World-to-camera matrices. [..., C, 4, 4]
        Ks: Camera intrinsics. [..., C, 3, 3]
        width: Image width.
        height: Image height.
        eps2d: A epsilon added to the 2D covariance for numerical stability. Default: 0.3.
        near_plane: Near plane distance. Default: 0.01.
        far_plane: Far plane distance. Default: 1e10.
        radius_clip: Gaussians with projected radii smaller than this value will be ignored. Default: 0.0.
        packed: If True, the output tensors will be packed into a flattened tensor. Default: False.
        sparse_grad: This is only effective when `packed` is True. If True, during backward the gradients
          of {`means`, `covars`, `quats`, `scales`} will be a sparse Tensor in COO layout. Default: False.
        calc_compensations: If True, a view-dependent opacity compensation factor will be computed, which
          is useful for anti-aliasing. Default: False.
        opacities: Gaussian opacities in range [0, 1]. If provided, will use it to compute a tighter bounds.
            [..., N] or None. Default: None.

    Returns:
        A tuple:

        If `packed` is True:

        - **batch_ids**. The batch indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **camera_ids**. The camera indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **gaussian_ids**. The column indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **indptr**. CSR-style index pointer into gaussian_ids for batch-camera pairs. Int32 tensor of shape [B*C+1].
        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [nnz, 2].
        - **means**. Projected Gaussian means in 2D. [nnz, 2]
        - **depths**. The z-depth of the projected Gaussians. [nnz]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [nnz, 3]
        - **compensations**. The view-dependent opacity compensation factor. [nnz]

        If `packed` is False:

        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [..., C, N, 2].
        - **means**. Projected Gaussian means in 2D. [..., C, N, 2]
        - **depths**. The z-depth of the projected Gaussians. [..., C, N]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [..., C, N, 3]
        - **compensations**. The view-dependent opacity compensation factor. [..., C, N]
    """
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == batch_dims + (N, 3), means.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    means = means.contiguous()
    if covars is not None:
        assert covars.shape == batch_dims + (N, 6), covars.shape
        covars = covars.contiguous()
    else:
        assert quats is not None, "covars or quats is required"
        assert scales is not None, "covars or scales is required"
        assert quats.shape == batch_dims + (N, 4), quats.shape
        assert scales.shape == batch_dims + (N, 3), scales.shape
        quats = quats.contiguous()
        scales = scales.contiguous()
    if sparse_grad:
        assert packed, "sparse_grad is only supported when packed is True"
        assert batch_dims == (), "sparse_grad does not support batch dimensions"
    if opacities is not None:
        assert opacities.shape == batch_dims + (N,), opacities.shape
        opacities = opacities.contiguous()

    assert (
        camera_model != "ftheta"
    ), "ftheta camera is only supported via UT, please set with_ut=True in the rasterization()"

    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()

    # Phase B: route the common 3DGS path (pinhole + quats/scales + no
    # compensations + no packed + batch-shape (B,)) through the Metal kernel.
    from ._dispatch import select_backend, Backend

    if (
        select_backend(means, "projection_ewa_3dgs_fused") is Backend.METAL
        and covars is None
        and not packed
        and not calc_compensations
        and camera_model == "pinhole"
        and len(batch_dims) <= 1
    ):
        from ..mps._projection import fully_fused_projection_3dgs_metal

        # Promote to (B, N, ...) shapes.
        if len(batch_dims) == 0:
            means_b = means.unsqueeze(0)
            quats_b = quats.unsqueeze(0) if quats is not None else None
            scales_b = scales.unsqueeze(0) if scales is not None else None
            viewmats_b = viewmats.unsqueeze(0)
            Ks_b = Ks.unsqueeze(0)
            opacities_b = opacities.unsqueeze(0) if opacities is not None else None
        else:
            means_b, quats_b, scales_b = means, quats, scales
            viewmats_b, Ks_b, opacities_b = viewmats, Ks, opacities

        radii_b, m2d_b, depths_b, conics_b, comp_b = (
            fully_fused_projection_3dgs_metal(
                means_b,
                quats_b,
                scales_b,
                viewmats_b,
                Ks_b,
                width,
                height,
                eps2d=eps2d,
                near_plane=near_plane,
                far_plane=far_plane,
                radius_clip=radius_clip,
                opacities=opacities_b,
            )
        )
        if len(batch_dims) == 0:
            return (
                radii_b.squeeze(0),
                m2d_b.squeeze(0),
                depths_b.squeeze(0),
                conics_b.squeeze(0),
                comp_b,
            )
        return radii_b, m2d_b, depths_b, conics_b, comp_b

    # Compute full 3x3 covariances if only quats/scales were provided.
    if covars is None:
        from ._math import _quat_scale_to_covar_preci

        covars3x3, _ = _quat_scale_to_covar_preci(
            quats,
            scales,
            compute_covar=True,
            compute_preci=False,
            triu=False,
        )
    else:
        # Input covars are flattened upper-triangle [..., N, 6].
        covars3x3 = _triu6_to_full3x3(covars)

    from ._torch_impl import _fully_fused_projection

    radii, means2d, depths, conics, compensations = _fully_fused_projection(
        means,
        covars3x3,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        calc_compensations=calc_compensations,
        camera_model=camera_model,
    )
    if compensations is None:
        # Match the CUDA-era contract that always returned a tensor for the
        # 5th slot; rendering.py treats `None` as "no compensation requested".
        # When `calc_compensations=False`, the CUDA backend returns a zero-
        # length tensor; we stay compatible by returning `None` and letting
        # callers branch on it.
        pass

    if not packed:
        return radii, means2d, depths, conics, compensations

    # Pack the dense result into COO-style buffers. The torch reference does
    # not produce packed output natively, so we filter on `radii > 0` and emit
    # the same eight-tuple as the CUDA packed path.
    valid = (radii > 0).all(dim=-1)  # [..., C, N]
    batch_dims = valid.shape[:-2]
    B = math.prod(batch_dims) if batch_dims else 1
    C, N = valid.shape[-2:]
    valid_f = valid.reshape(B, C, N)
    nz = torch.nonzero(valid_f, as_tuple=False)  # [nnz, 3]
    batch_ids = nz[:, 0].to(torch.int32)
    camera_ids = nz[:, 1].to(torch.int32)
    gaussian_ids = nz[:, 2].to(torch.int32)
    # CSR-style indptr over the (B*C) image groups.
    image_keys = (batch_ids.long() * C + camera_ids.long())
    indptr = torch.zeros(B * C + 1, dtype=torch.int32, device=means.device)
    counts = torch.bincount(image_keys, minlength=B * C).to(torch.int32)
    indptr[1:] = torch.cumsum(counts, dim=0).to(torch.int32)

    def _gather(t):
        flat = t.reshape((B, C, N) + t.shape[len(batch_dims) + 2 :])
        return flat[batch_ids.long(), camera_ids.long(), gaussian_ids.long()]

    radii_p = _gather(radii)
    means2d_p = _gather(means2d)
    depths_p = _gather(depths)
    conics_p = _gather(conics)
    comp_p = _gather(compensations) if compensations is not None else None
    return (
        batch_ids,
        camera_ids,
        gaussian_ids,
        indptr,
        radii_p,
        means2d_p,
        depths_p,
        conics_p,
        comp_p,
    )


def _triu6_to_full3x3(triu: Tensor) -> Tensor:
    """Expand `[..., N, 6]` upper-triangle covariances to `[..., N, 3, 3]`."""
    a, b, c, d, e, f = torch.unbind(triu, dim=-1)
    row0 = torch.stack([a, b, c], dim=-1)
    row1 = torch.stack([b, d, e], dim=-1)
    row2 = torch.stack([c, e, f], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


@torch.no_grad()
def isect_tiles(
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    radii: Tensor,  # [..., N, 2] or [nnz, 2]
    depths: Tensor,  # [..., N] or [nnz]
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
    segmented: bool = False,
    packed: bool = False,
    n_images: Optional[int] = None,
    image_ids: Optional[Tensor] = None,
    gaussian_ids: Optional[Tensor] = None,
    conics: Optional[
        Tensor
    ] = None,  # [..., N, 3] or [nnz, 3], enables AccuTile when provided
    opacities: Optional[
        Tensor
    ] = None,  # [..., N] or [nnz], enables AccuTile when provided
) -> Tuple[Tensor, Tensor, Tensor]:
    """Maps projected Gaussians to intersecting tiles.

    When `conics` and `opacities` are provided the kernel uses conservative ellipse intersection (AccuTile/SNUGBOX),
    skipping tiles that the opacity-thresholded ellipse does not touch. When either is `None` the kernel falls back to the original axis-aligned bounding box.

    Args:
        means2d: Projected Gaussian means. [..., N, 2] if packed is False, [nnz, 2] if packed is True.
        radii: Maximum radii of the projected Gaussians. [..., N, 2] if packed is False, [nnz, 2] if packed is True.
        depths: Z-depth of the projected Gaussians. [..., N] if packed is False, [nnz] if packed is True.
        tile_size: Tile size.
        tile_width: Tile width.
        tile_height: Tile height.
        sort: If True, the returned intersections will be sorted by the intersection ids. Default: True.
        segmented: If True, segmented radix sort will be used to sort the intersections. Default: False.
        packed: If True, the input tensors are packed. Default: False.
        n_images: Number of images. Required if packed is True.
        image_ids: The image indices of the projected Gaussians. Required if packed is True.
        gaussian_ids: The column indices of the projected Gaussians. Required if packed is True.
        conics: Inverse of projected covariances (upper triangle). [..., N, 3] if packed is False, [nnz, 3] if packed is True. Enables AccuTile when provided together with opacities.
        opacities: Gaussian opacities. [..., N] if packed is False, [nnz] if packed is True. Enables AccuTile when provided together with conics.

    Returns:
        A tuple:

        - **Tiles per Gaussian**. The number of tiles intersected by each Gaussian.
          Int32 [..., N] if packed is False, Int32 [nnz] if packed is True.
        - **Intersection ids**. Each id is an 64-bit integer with the following
          information: image_id (Xc bits) | tile_id (Xt bits) | depth (32 bits).
          Xc and Xt are the maximum number of bits required to represent the image and
          tile ids, respectively. Int64 [n_isects]
        - **Flatten ids**. The global flatten indices in [I * N] or [nnz] (packed). [n_isects]
    """
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert radii.shape == (nnz, 2), radii.shape
        assert depths.shape == (nnz,), depths.shape
        if conics is not None:
            assert conics.shape == (nnz, 3), conics.shape
        if opacities is not None:
            assert opacities.shape == (nnz,), opacities.shape
        assert image_ids is not None, "image_ids is required if packed is True"
        assert gaussian_ids is not None, "gaussian_ids is required if packed is True"
        assert n_images is not None, "n_images is required if packed is True"
        image_ids = image_ids.contiguous()
        gaussian_ids = gaussian_ids.contiguous()
        I = n_images

    else:
        image_dims = means2d.shape[:-2]
        I = math.prod(image_dims)
        N = means2d.shape[-2]
        assert means2d.shape == image_dims + (N, 2), means2d.shape
        assert radii.shape == image_dims + (N, 2), radii.shape
        assert depths.shape == image_dims + (N,), depths.shape
        if conics is not None:
            assert conics.shape == image_dims + (N, 3), conics.shape
        if opacities is not None:
            assert opacities.shape == image_dims + (N,), opacities.shape

    if packed:
        raise NotImplementedError(
            "isect_tiles(packed=True) is not yet wired in the MPS port. "
            "Use packed=False (the default for the public rasterization() API)."
        )

    from ._dispatch import select_backend, Backend

    if select_backend(means2d, "intersect_tile") is Backend.METAL:
        from ..mps._intersect import _intersect_tile_metal

        return _intersect_tile_metal(
            means2d.contiguous(),
            radii.contiguous(),
            depths.contiguous(),
            tile_size,
            tile_width,
            tile_height,
            sort=sort,
        )

    from ._torch_impl import _isect_tiles

    tiles_per_gauss, isect_ids, flatten_ids = _isect_tiles(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        tile_size,
        tile_width,
        tile_height,
        sort=sort,
    )
    return tiles_per_gauss, isect_ids, flatten_ids


@torch.no_grad()
def isect_tiles_lidar(
    lidar: RowOffsetStructuredSpinningLidarModelParametersExt,
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    radii: Tensor,  # [..., N, 2] or [nnz, 2]
    depths: Tensor,  # [..., N] or [nnz]
    sort: bool = True,
    segmented: bool = False,
    packed: bool = False,
    n_images: Optional[int] = None,
    image_ids: Optional[Tensor] = None,
    gaussian_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Maps projected Gaussians to intersecting tiles.

    Args:
        means2d: Projected Gaussian means. [..., N, 2] if packed is False, [nnz, 2] if packed is True.
        radii: Maximum radii of the projected Gaussians. [..., N, 2] if packed is False, [nnz, 2] if packed is True.
        depths: Z-depth of the projected Gaussians. [..., N] if packed is False, [nnz] if packed is True.
        sort: If True, the returned intersections will be sorted by the intersection ids. Default: True.
        segmented: If True, segmented radix sort will be used to sort the intersections. Default: False.
        packed: If True, the input tensors are packed. Default: False.
        n_images: Number of images. Required if packed is True.
        image_ids: The image indices of the projected Gaussians. Required if packed is True.
        gaussian_ids: The column indices of the projected Gaussians. Required if packed is True.

    Returns:
        A tuple:

        - **Tiles per Gaussian**. The number of tiles intersected by each Gaussian.
          Int32 [..., N] if packed is False, Int32 [nnz] if packed is True.
        - **Intersection ids**. Each id is an 64-bit integer with the following
          information: image_id (Xc bits) | tile_id (Xt bits) | depth (32 bits).
          Xc and Xt are the maximum number of bits required to represent the image and
          tile ids, respectively. Int64 [n_isects]
        - **Flatten ids**. The global flatten indices in [I * N] or [nnz] (packed). [n_isects]
    """
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert radii.shape == (nnz, 2), radii.shape
        assert depths.shape == (nnz,), depths.shape
        assert image_ids is not None, "image_ids is required if packed is True"
        assert gaussian_ids is not None, "gaussian_ids is required if packed is True"
        assert n_images is not None, "n_images is required if packed is True"
        image_ids = image_ids.contiguous()
        gaussian_ids = gaussian_ids.contiguous()
        I = n_images

    else:
        image_dims = means2d.shape[:-2]
        I = math.prod(image_dims)
        N = means2d.shape[-2]
        assert means2d.shape == (*image_dims, N, 2), means2d.shape
        assert radii.shape == (*image_dims, N, 2), radii.shape
        assert depths.shape == (*image_dims, N), depths.shape

    tiles_per_gauss, isect_ids, flatten_ids = _make_lazy_cuda_func(
        "intersect_tile_lidar"
    )(
        lidar.to_cpp(),
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        image_ids,
        gaussian_ids,
        I,
        sort,
        segmented,
    )
    return tiles_per_gauss, isect_ids, flatten_ids


@torch.no_grad()
def isect_offset_encode(
    isect_ids: Tensor,
    n_images: int,
    tile_width: int,
    tile_height: int,
) -> Tensor:
    """Encodes intersection ids to offsets.

    Args:
        isect_ids: Intersection ids. [n_isects]
        n_images: Number of images.
        tile_width: Tile width.
        tile_height: Tile height.

    Returns:
        Offsets. [I, tile_height, tile_width]
    """
    from ._torch_impl import _isect_offset_encode

    return _isect_offset_encode(
        isect_ids.contiguous(), n_images, tile_width, tile_height
    )


def rasterize_to_pixels(
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    conics: Tensor,  # [..., N, 3] or [nnz, 3]
    colors: Tensor,  # [..., N, channels] or [nnz, channels]
    opacities: Tensor,  # [..., N] or [nnz]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., channels]
    masks: Optional[Tensor] = None,  # [..., tile_height, tile_width]
    packed: bool = False,
    absgrad: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Rasterizes Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [..., N, 2] if packed is False, [nnz, 2] if packed is True.
        conics: Inverse of the projected covariances with only upper triangle values. [..., N, 3] if packed is False, [nnz, 3] if packed is True.
        colors: Gaussian colors or ND features. [..., N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [..., N] if packed is False, [nnz] if packed is True.
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [..., tile_height, tile_width]
        flatten_ids: The global flatten indices in [I * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [..., channels]. Default: None.
        masks: Optional tile mask to skip rendering GS to masked tiles. [..., tile_height, tile_width]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

    Returns:
        A tuple:

        - **Rendered colors**. [..., image_height, image_width, channels]
        - **Rendered alphas**. [..., image_height, image_width, 1]
    """

    image_dims = means2d.shape[:-2]
    channels = colors.shape[-1]
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert conics.shape == (nnz, 3), conics.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(-2)
        assert means2d.shape == image_dims + (N, 2), means2d.shape
        assert conics.shape == image_dims + (N, 3), conics.shape
        assert colors.shape == image_dims + (N, channels), colors.shape
        assert opacities.shape == image_dims + (N,), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == image_dims + (channels,), backgrounds.shape
        backgrounds = backgrounds.contiguous()
    if masks is not None:
        assert masks.shape == isect_offsets.shape, masks.shape
        masks = masks.contiguous()

    # Pad the channels to the nearest supported number if necessary
    if channels > 513 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        16,
        17,
        32,
        33,
        64,
        65,
        128,
        129,
        256,
        257,
        512,
        513,
    ):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [
                colors,
                torch.zeros(*colors.shape[:-1], padded_channels, device=device),
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[-2:]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    if packed:
        raise NotImplementedError(
            "rasterize_to_pixels(packed=True) is not yet wired in the MPS port. "
            "Use packed=False (the default for the public rasterization() API)."
        )

    # Phase B: route MPS tensors through the native Metal kernel when the
    # input shape matches what the kernel supports (CDIM=3, tile_size=16,
    # no backgrounds / no masks). Anything else falls back to the torch
    # rasterizer below.
    from ._dispatch import select_backend, Backend

    if (
        select_backend(means2d, "rasterize_to_pixels_3dgs_fwd") is Backend.METAL
        and tile_size == 16
        and channels == 3
        and not packed
        and backgrounds is None
        and masks is None
    ):
        from ..mps._rasterize import rasterize_to_pixels_3dgs_metal

        render_colors, render_alphas = rasterize_to_pixels_3dgs_metal(
            means2d.contiguous(),
            conics.contiguous(),
            colors.contiguous(),
            opacities.contiguous(),
            image_width,
            image_height,
            tile_size,
            isect_offsets.contiguous(),
            flatten_ids.contiguous(),
            backgrounds=None,
            masks=None,
        )
        if padded_channels > 0:
            render_colors = render_colors[..., :-padded_channels]
        return render_colors, render_alphas

    from ._torch_rasterize import _rasterize_to_pixels_torch

    render_colors, render_alphas = _rasterize_to_pixels_torch(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        backgrounds=backgrounds,
        masks=masks,
    )

    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]
    return render_colors, render_alphas


def rasterize_to_pixels_eval3d(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    colors: Tensor,  # [..., C, N, channels] or [nnz, channels]
    opacities: Tensor,  # [..., C, N] or [nnz]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., C, channels]
    masks: Optional[Tensor] = None,  # [..., C, tile_height, tile_width]
    camera_model: CameraModel = "pinhole",
    ut_params: Optional[UnscentedTransformParameters] = None,
    rays: Optional[Tensor] = None,  # [..., C, H, W, 6]
    # distortion
    radial_coeffs: Optional[Tensor] = None,  # [..., C, 6] or [..., C, 4]
    tangential_coeffs: Optional[Tensor] = None,  # [..., C, 2]
    thin_prism_coeffs: Optional[Tensor] = None,  # [..., C, 4]
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    lidar_coeffs: Optional[RowOffsetStructuredSpinningLidarModelParametersExt] = None,
    external_distortion_coeffs: Optional[BivariateWindshieldModelParameters] = None,
    # rolling shutter
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[Tensor] = None,  # [..., C, 4, 4]
    use_hit_distance: bool = False,
    return_normals: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Rasterizes Gaussians to pixels.

    Similar to `rasterize_to_pixels()`, but compute the Gaussian responses in the
    3D world space instead of the 2D image space. Supports rolling shutter and
    camera distortion.

    Returns:
        A tuple:

        - **Rendered colors**. [..., C, image_height, image_width, channels]
        - **Rendered alphas**. [..., C, image_height, image_width, 1]
    """
    if ut_params is None:
        ut_params = UnscentedTransformParameters()

    colors, alphas, *_ = rasterize_to_pixels_eval3d_extra(
        means=means,
        quats=quats,
        scales=scales,
        colors=colors,
        opacities=opacities,
        viewmats=viewmats,
        Ks=Ks,
        rays=rays,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        backgrounds=backgrounds,
        masks=masks,
        camera_model=camera_model,
        ut_params=ut_params,
        radial_coeffs=radial_coeffs,
        tangential_coeffs=tangential_coeffs,
        thin_prism_coeffs=thin_prism_coeffs,
        ftheta_coeffs=ftheta_coeffs,
        lidar_coeffs=lidar_coeffs,
        external_distortion_coeffs=external_distortion_coeffs,
        rolling_shutter=rolling_shutter,
        viewmats_rs=viewmats_rs,
        return_sample_counts=False,
        use_hit_distance=use_hit_distance,
        return_normals=return_normals,
    )
    return colors, alphas


def rasterize_to_pixels_eval3d_extra(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    colors: Tensor,  # [..., C, N, channels] or [nnz, channels]
    opacities: Tensor,  # [..., C, N] or [nnz]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., C, channels]
    masks: Optional[Tensor] = None,  # [..., C, tile_height, tile_width]
    camera_model: CameraModel = "pinhole",
    ut_params: Optional[UnscentedTransformParameters] = None,
    rays: Optional[Tensor] = None,  # [..., C, P, 6]
    # distortion
    radial_coeffs: Optional[Tensor] = None,  # [..., C, 6] or [..., C, 4]
    tangential_coeffs: Optional[Tensor] = None,  # [..., C, 2]
    thin_prism_coeffs: Optional[Tensor] = None,  # [..., C, 4]
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    lidar_coeffs: Optional[RowOffsetStructuredSpinningLidarModelParametersExt] = None,
    external_distortion_coeffs: Optional[BivariateWindshieldModelParameters] = None,
    # rolling shutter
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[Tensor] = None,  # [..., C, 4, 4]
    return_sample_counts: bool = False,
    use_hit_distance: bool = False,
    return_normals: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    """Rasterizes Gaussians to pixels, returning extra information for debugging.

    Similar to `rasterize_to_pixels_eval3d()`, but returns turns the last gaussian id
    accumulated in a pixel, and optionally the number of accumulated samples per pixel.

    Args:
        return_last_ids: If True, also return last flatten_idx per pixel. Default: False.
        return_sample_counts: If True, also return number of accumulated samples per pixel. Default: False.
        return_normals: If True, compute and return accumulated normals per pixel.
            Normals are computed from Gaussian quaternions (canonical normal = (0,0,1)
            transformed by rotation, flipped if facing away from ray). Default: False.

    Returns:
        A tuple (contents depend on return flags):

        - **Rendered colors**. [..., C, image_height, image_width, channels]
        - **Rendered alphas**. [..., C, image_height, image_width, 1]
        - **Last flatten_idx**. [..., C, image_height, image_width]
        - **Sample counts** (optional). [..., C, image_height, image_width]. If return_sample_counts=True.
        - **Rendered normals** (optional). [..., C, image_height, image_width, 3]. If return_normals=True.
    """
    if ut_params is None:
        ut_params = UnscentedTransformParameters()

    batch_dims = means.shape[:-2]
    num_batch_dims = len(batch_dims)
    N = means.size(-2)
    C = viewmats.size(-3)
    P = rays.shape[-2] if rays is not None else 0
    channels = colors.shape[-1]
    device = means.device

    assert means.shape == batch_dims + (N, 3), means.shape
    assert quats.shape == batch_dims + (N, 4), quats.shape
    assert scales.shape == batch_dims + (N, 3), scales.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    if rays is not None:
        assert_shape("rays", rays, batch_dims + (C, P, 6))
        assert (
            rays.dtype == torch.float32
        ), f"rays must be torch.float32, got {rays.dtype}"

    assert colors.ndim in (num_batch_dims + 2, num_batch_dims + 3), colors.shape
    if colors.ndim == num_batch_dims + 2:
        raise NotImplementedError("packed mode is not supported yet")
        assert (
            colors.shape[:-2] == batch_dims and colors.shape[-1] == channels
        ), colors.shape
    else:
        assert colors.shape == batch_dims + (C, N, channels), colors.shape
    assert opacities.shape == colors.shape[:-1], opacities.shape

    if backgrounds is not None:
        assert backgrounds.shape == batch_dims + (C, channels), backgrounds.shape
        backgrounds = backgrounds.contiguous()

    if masks is not None:
        assert masks.shape == isect_offsets.shape, masks.shape
        masks = masks.contiguous()

    if radial_coeffs is not None:
        assert radial_coeffs.shape[:-1] == batch_dims + (C,) and radial_coeffs.shape[
            -1
        ] in (6, 4), radial_coeffs.shape
        radial_coeffs = radial_coeffs.contiguous()

    if tangential_coeffs is not None:
        assert tangential_coeffs.shape == batch_dims + (C, 2), tangential_coeffs.shape
        tangential_coeffs = tangential_coeffs.contiguous()

    if thin_prism_coeffs is not None:
        assert thin_prism_coeffs.shape == batch_dims + (C, 4), thin_prism_coeffs.shape
        thin_prism_coeffs = thin_prism_coeffs.contiguous()

    if viewmats_rs is not None:
        assert viewmats_rs.shape == batch_dims + (C, 4, 4), viewmats_rs.shape
        viewmats_rs = viewmats_rs.contiguous()

    # Pad the channels to the nearest supported number if necessary
    channels = colors.shape[-1]
    if channels > 513 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        16,
        17,
        32,
        33,
        64,
        65,
        128,
        129,
        256,
        257,
        512,
        513,
    ):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        # Insert padding before the last channel so that it stays at
        # CDIM-1.  When depth is present it is always the last channel,
        # so this keeps it where the CUDA kernel writes hit_distance.
        # When depth is absent the last channel is preserved
        # through the round-trip.
        # This matches the approach used in rasterize_to_pixels_2dgs.
        colors = torch.cat(
            [
                colors[..., :-1],
                torch.zeros(*colors.shape[:-1], padded_channels, device=device),
                colors[..., -1:],
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[-2:]
    if camera_model == "lidar":
        assert tile_width == lidar_coeffs.tiling.n_bins_azimuth
        assert tile_height == lidar_coeffs.tiling.n_bins_elevation
        # TODO: improve checks. Right now we don't have access to max_pts_per_tile used,
        # hence this assert needs to be commented out.
        # assert tile_width*tile_height*lidar_coeffs.tiling.max_pts_per_tile >= lidar_coeffs.n_rows*lidar_coeffs.n_columns
    else:
        assert (
            tile_height * tile_size >= image_height
        ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
        assert (
            tile_width * tile_size >= image_width
        ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    # Phase B: route the simplest eval3d path (pinhole, global shutter, no
    # distortion, CDIM=3, no backgrounds/masks, no normals/sample-counts/hit-
    # distance, default ut_params) through the Metal kernel. Anything else
    # falls back to the torch reference.
    from ._dispatch import select_backend, Backend

    # Camera-model dispatch:
    #   pinhole + no distortion → cm_id=0
    #   fisheye + radial[4]    → cm_id=2
    # Pinhole + radial/tangential/thin-prism distortion (cm_id=1) requires
    # Newton-iteration undistortion in MSL with a non-trivial residual chain
    # (radial division + delta polynomials); still falls back to torch.
    if camera_model == "pinhole" and radial_coeffs is None and tangential_coeffs is None and thin_prism_coeffs is None:
        _eval3d_cm_id = 0
    elif (camera_model == "fisheye"
          and tangential_coeffs is None and thin_prism_coeffs is None):
        _eval3d_cm_id = 2
    else:
        _eval3d_cm_id = -1

    eval3d_metal_eligible = (
        select_backend(means, "rasterize_to_pixels_eval3d_fwd") is Backend.METAL
        and _eval3d_cm_id >= 0
        and rays is None
        and ftheta_coeffs is None
        and lidar_coeffs is None
        and external_distortion_coeffs is None
        and rolling_shutter == RollingShutterType.GLOBAL
        and viewmats_rs is None
        and backgrounds is None
        and masks is None
        and channels == 3
        and tile_size == 16
        and not return_sample_counts
        and not use_hit_distance
        and not return_normals
    )
    if eval3d_metal_eligible:
        from ..mps._eval3d import rasterize_to_pixels_eval3d_metal

        import math as _math
        Bm = _math.prod(batch_dims) if batch_dims else 1
        Im = Bm * C
        means_m = means.reshape(Bm, N, 3).contiguous()
        quats_m = quats.reshape(Bm, N, 4).contiguous()
        scales_m = scales.reshape(Bm, N, 3).contiguous()
        colors_m = colors.reshape(Im, N, channels).contiguous()
        op_m = opacities.reshape(Im, N).contiguous()
        vm_m = viewmats.reshape(Im, 4, 4).contiguous()
        Ks_m = Ks.reshape(Im, 3, 3).contiguous()
        iso_m = isect_offsets.reshape(Im, tile_height, tile_width).contiguous()

        # Per-camera radial buffer (fisheye uses 4 coeffs).
        if radial_coeffs is not None:
            rad_for_metal = radial_coeffs.reshape(Im, -1).contiguous()
        else:
            rad_for_metal = None

        rc_m, ra_m = rasterize_to_pixels_eval3d_metal(
            means_m, quats_m, scales_m, colors_m, op_m, vm_m, Ks_m,
            image_width, image_height, tile_size,
            iso_m, flatten_ids.contiguous(), C,
            cm_id=_eval3d_cm_id, radial_coeffs=rad_for_metal,
        )
        render_colors = rc_m.reshape(batch_dims + (C, image_height, image_width, channels))
        render_alphas = ra_m.reshape(batch_dims + (C, image_height, image_width, 1))
        last_ids = torch.zeros(
            batch_dims + (C, image_height, image_width),
            device=device, dtype=torch.int32,
        )
        sample_counts = torch.zeros_like(last_ids)
        render_normals = None
        if padded_channels > 0:
            render_colors = torch.cat(
                [render_colors[..., : -padded_channels - 1], render_colors[..., -1:]],
                dim=-1,
            )
        return (
            render_colors, render_alphas, last_ids, sample_counts, render_normals,
        )

    (
        render_colors,
        render_alphas,
        last_ids,
        sample_counts,
        render_normals,
    ) = _RasterizeToPixelsEval3D.apply(
        means.contiguous(),
        quats.contiguous(),
        scales.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        backgrounds.contiguous() if backgrounds is not None else None,
        masks.contiguous() if masks is not None else None,
        viewmats.contiguous(),
        Ks.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        camera_model,
        ut_params,
        rays.contiguous() if rays is not None else None,
        # distortion
        radial_coeffs.contiguous() if radial_coeffs is not None else None,
        tangential_coeffs.contiguous() if tangential_coeffs is not None else None,
        thin_prism_coeffs.contiguous() if thin_prism_coeffs is not None else None,
        ftheta_coeffs,
        lidar_coeffs,
        external_distortion_coeffs,
        # rolling shutter
        rolling_shutter,
        viewmats_rs.contiguous() if viewmats_rs is not None else None,
        # Forward is always collecting the last_ids for the backward pass,
        # no need to tell it to do it.
        return_sample_counts,  # Pass flag to forward
        use_hit_distance,
        return_normals,  # Pass return_normals flag to forward
    )

    if padded_channels > 0:
        render_colors = torch.cat(
            [render_colors[..., : -padded_channels - 1], render_colors[..., -1:]],
            dim=-1,
        )

    return render_colors, render_alphas, last_ids, sample_counts, render_normals


@torch.no_grad()
def rasterize_to_indices_in_range(
    range_start: int,
    range_end: int,
    transmittances: Tensor,  # [..., image_height, image_width]
    means2d: Tensor,  # [..., N, 2]
    conics: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Rasterizes a batch of Gaussians to images but only returns the indices.

    .. note::

        This function supports iterative rasterization, in which each call of this function
        will rasterize a batch of Gaussians from near to far, defined by `[range_start, range_end)`.
        If a one-step full rasterization is desired, set `range_start` to 0 and `range_end` to a really
        large number, e.g, 1e10.

    Args:
        range_start: The start batch of Gaussians to be rasterized (inclusive).
        range_end: The end batch of Gaussians to be rasterized (exclusive).
        transmittances: Currently transmittances. [..., image_height, image_width]
        means2d: Projected Gaussian means. [..., N, 2]
        conics: Inverse of the projected covariances with only upper triangle values. [..., N, 3]
        opacities: Gaussian opacities that support per-view values. [..., N]
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [..., tile_height, tile_width]
        flatten_ids: The global flatten indices in [I * N] from  `isect_tiles()`. [n_isects]

    Returns:
        A tuple:

        - **Gaussian ids**. Gaussian ids for the pixel intersection. A flattened list of shape [M].
        - **Pixel ids**. pixel indices (row-major). A flattened list of shape [M].
        - **Image ids**. image indices. A flattened list of shape [M].
    """

    image_dims = means2d.shape[:-2]
    tile_height, tile_width = isect_offsets.shape[-2:]
    N = means2d.shape[-2]
    assert transmittances.shape == image_dims + (
        image_height,
        image_width,
    ), transmittances.shape
    assert means2d.shape == image_dims + (N, 2), means2d.shape
    assert conics.shape == image_dims + (N, 3), conics.shape
    assert opacities.shape == image_dims + (N,), opacities.shape
    assert isect_offsets.shape == image_dims + (
        tile_height,
        tile_width,
    ), isect_offsets.shape
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    out_gauss_ids, out_indices = _make_lazy_cuda_func("rasterize_to_indices_3dgs")(
        range_start,
        range_end,
        transmittances.contiguous(),
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
    )
    out_pixel_ids = out_indices % (image_width * image_height)
    out_image_ids = out_indices // (image_width * image_height)
    return out_gauss_ids, out_pixel_ids, out_image_ids


class _QuatScaleToCovarPreci(torch.autograd.Function):
    """Converts quaternions and scales to covariance and precision matrices."""

    @staticmethod
    def forward(
        ctx,
        quats: Tensor,  # [..., 4],
        scales: Tensor,  # [..., 3],
        compute_covar: bool = True,
        compute_preci: bool = True,
        triu: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        covars, precis = _make_lazy_cuda_func("quat_scale_to_covar_preci_fwd")(
            quats, scales, compute_covar, compute_preci, triu
        )
        ctx.save_for_backward(quats, scales)
        ctx.compute_covar = compute_covar
        ctx.compute_preci = compute_preci
        ctx.triu = triu
        return covars, precis

    @staticmethod
    def backward(ctx, v_covars: Tensor, v_precis: Tensor):
        quats, scales = ctx.saved_tensors
        compute_covar = ctx.compute_covar
        compute_preci = ctx.compute_preci
        triu = ctx.triu
        if compute_covar and v_covars.is_sparse:
            v_covars = v_covars.to_dense()
        if compute_preci and v_precis.is_sparse:
            v_precis = v_precis.to_dense()
        v_quats, v_scales = _make_lazy_cuda_func("quat_scale_to_covar_preci_bwd")(
            quats,
            scales,
            triu,
            v_covars.contiguous() if compute_covar else None,
            v_precis.contiguous() if compute_preci else None,
        )
        return (
            v_quats,
            v_scales,
            None,  # compute_covar
            None,  # compute_preci
            None,  # triu
        )


class _Proj(torch.autograd.Function):
    """Perspective fully_fused_projection on Gaussians."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [..., C, N, 3]
        covars: Tensor,  # [..., C, N, 3, 3]
        Ks: Tensor,  # [..., C, 3, 3]
        width: int,
        height: int,
        camera_model: CameraModel = "pinhole",
    ) -> Tuple[Tensor, Tensor]:
        assert (
            camera_model != "ftheta"
        ), "ftheta camera is only supported via UT, please set with_ut=True in the rasterization()"

        camera_model_type = _make_lazy_cuda_obj(
            f"CameraModelType.{camera_model.upper()}"
        )

        means2d, covars2d = _make_lazy_cuda_func("projection_ewa_simple_fwd")(
            means,
            covars,
            Ks,
            width,
            height,
            camera_model_type,
        )
        ctx.save_for_backward(means, covars, Ks)
        ctx.width = width
        ctx.height = height
        ctx.camera_model_type = camera_model_type
        return means2d, covars2d

    @staticmethod
    def backward(ctx, v_means2d: Tensor, v_covars2d: Tensor):
        means, covars, Ks = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        camera_model_type = ctx.camera_model_type
        v_means, v_covars = _make_lazy_cuda_func("projection_ewa_simple_bwd")(
            means,
            covars,
            Ks,
            width,
            height,
            camera_model_type,
            v_means2d.contiguous(),
            v_covars2d.contiguous(),
        )
        return (
            v_means,
            v_covars,
            None,  # Ks
            None,  # width
            None,  # height
            None,  # camera_model
        )


class _FullyFusedProjection(torch.autograd.Function):
    """Projects Gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [..., N, 3]
        covars: Tensor,  # [..., N, 6] or None
        quats: Tensor,  # [..., N, 4] or None
        scales: Tensor,  # [..., N, 3] or None
        viewmats: Tensor,  # [..., C, 4, 4]
        Ks: Tensor,  # [..., C, 3, 3]
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        calc_compensations: bool,
        camera_model: CameraModel = "pinhole",
        opacities: Optional[Tensor] = None,  # [..., N] or None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        assert (
            camera_model != "ftheta"
        ), "ftheta camera is only supported via UT, please set with_ut=True in the rasterization()"

        camera_model_type = _make_lazy_cuda_obj(
            f"CameraModelType.{camera_model.upper()}"
        )

        # "covars" and {"quats", "scales"} are mutually exclusive
        radii, means2d, depths, conics, compensations = _make_lazy_cuda_func(
            "projection_ewa_3dgs_fused_fwd"
        )(
            means,
            covars,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
            camera_model_type,
        )
        if not calc_compensations:
            compensations = None
        ctx.save_for_backward(
            means, covars, quats, scales, viewmats, Ks, radii, conics, compensations
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d
        ctx.camera_model_type = camera_model_type

        return radii, means2d, depths, conics, compensations

    @staticmethod
    def backward(ctx, v_radii, v_means2d, v_depths, v_conics, v_compensations):
        (
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            radii,
            conics,
            compensations,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        eps2d = ctx.eps2d
        camera_model_type = ctx.camera_model_type
        if v_compensations is not None:
            v_compensations = v_compensations.contiguous()
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "projection_ewa_3dgs_fused_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            camera_model_type,
            radii,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            ctx.needs_input_grad[4],  # viewmats_requires_grad
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_covars = None
        if not ctx.needs_input_grad[2]:
            v_quats = None
        if not ctx.needs_input_grad[3]:
            v_scales = None
        if not ctx.needs_input_grad[4]:
            v_viewmats = None
        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            v_viewmats,
            None,  # Ks
            None,  # width
            None,  # height
            None,  # eps2d
            None,  # near_plane
            None,  # far_plane
            None,  # radius_clip
            None,  # calc_compensations
            None,  # camera_model
            None,  # ut_params
            None,  # radial_coeffs
        )


def fully_fused_projection_with_ut(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    opacities: Optional[Tensor],  # [..., N]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    calc_compensations: bool = False,
    camera_model: CameraModel = "pinhole",
    ut_params: Optional[UnscentedTransformParameters] = None,
    # distortion
    radial_coeffs: Optional[Tensor] = None,  # [..., C, 6] or [..., C, 4]
    tangential_coeffs: Optional[Tensor] = None,  # [..., C, 2]
    thin_prism_coeffs: Optional[Tensor] = None,  # [..., C, 4]
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    lidar_coeffs: Optional[RowOffsetStructuredSpinningLidarModelParametersExt] = None,
    external_distortion_coeffs: Optional[BivariateWindshieldModelParameters] = None,
    # rolling shutter
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[Tensor] = None,  # [..., C, 4, 4]
    global_z_order: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Projects Gaussians to 2D using Unscented Transform (UT).

    similar to `fully_fused_projection()`, but supports camera distortion and
    rolling shutter.

    .. warning::
        This function is not differentiable to any input.

    Args:
        global_z_order: Defines how Gaussians are sorted for depth ordering. If True (default),
            Gaussians are sorted by their z-coordinate in camera space. If False, they are sorted
            by their Euclidean distance from the camera origin.             The z-coordinate sorting is typically
            faster and sufficient for most cases, while Euclidean distance can be useful for scenes
            with wide field-of-view or non-standard camera models. Default: True.
    """
    if ut_params is None:
        ut_params = UnscentedTransformParameters()

    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == batch_dims + (N, 3), means.shape
    assert quats.shape == batch_dims + (N, 4), quats.shape
    assert scales.shape == batch_dims + (N, 3), scales.shape
    if opacities is not None:
        assert opacities.shape == batch_dims + (N,), opacities.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    if radial_coeffs is not None:
        assert radial_coeffs.shape[:-1] == batch_dims + (C,) and radial_coeffs.shape[
            -1
        ] in [6, 4], radial_coeffs.shape
    if tangential_coeffs is not None:
        assert tangential_coeffs.shape == batch_dims + (C, 2), tangential_coeffs.shape
    if thin_prism_coeffs is not None:
        assert thin_prism_coeffs.shape == batch_dims + (C, 4), thin_prism_coeffs.shape
    if viewmats_rs is not None:
        assert viewmats_rs.shape == batch_dims + (C, 4, 4), viewmats_rs.shape

    if lidar_coeffs is not None:
        assert isinstance(
            lidar_coeffs, RowOffsetStructuredSpinningLidarModelParametersExt
        )

    if lidar_coeffs is not None or external_distortion_coeffs is not None:
        raise NotImplementedError(
            "lidar / external-distortion paths are out of scope in the MPS port. "
            "Pass lidar_coeffs=None and external_distortion_coeffs=None."
        )

    # Phase B: Metal forward for the simplest UT path (pinhole + global
    # shutter + no distortion + global_z_order=True + non-strict sigma
    # validity). Anything more exotic falls back to the torch reference.
    from ._dispatch import select_backend, Backend

    # Camera-model dispatch into the Metal kernel:
    #   pinhole + no distortion        → cm_id=0
    #   pinhole + any radial/tang/prism → cm_id=1
    #   fisheye (4 radial coeffs)      → cm_id=2
    has_pinhole_distortion = (
        camera_model == "pinhole"
        and (radial_coeffs is not None
             or tangential_coeffs is not None
             or thin_prism_coeffs is not None)
    )
    if camera_model == "pinhole" and not has_pinhole_distortion:
        _ut_cm_id = 0
    elif camera_model == "pinhole":
        _ut_cm_id = 1
    elif camera_model == "fisheye":
        _ut_cm_id = 2
    else:
        _ut_cm_id = -1

    metal_eligible = (
        select_backend(means, "projection_ut_3dgs_fused_fwd") is Backend.METAL
        and _ut_cm_id >= 0
        and ftheta_coeffs is None
        and rolling_shutter == RollingShutterType.GLOBAL
        and viewmats_rs is None
        and global_z_order
        and not ut_params.require_all_sigma_points_valid
    )

    if metal_eligible:
        from ..mps._projection_ut import _projection_ut_3dgs_fused_fwd_metal

        # The Metal kernel expects [B, N, ...] (means/quats/scales) and
        # [B, C, ...] (viewmats/Ks). Collapse leading batch_dims to a single
        # B for the kernel and reshape back after.
        import math as _math
        B = _math.prod(batch_dims) if batch_dims else 1
        means_flat_b = means.reshape(B, N, 3).contiguous()
        quats_flat_b = quats.reshape(B, N, 4).contiguous()
        scales_flat_b = scales.reshape(B, N, 3).contiguous()
        op_flat_b = (
            opacities.reshape(B, N).contiguous() if opacities is not None else None
        )
        viewmats_flat_b = viewmats.reshape(B, C, 4, 4).contiguous()
        Ks_flat_b = Ks.reshape(B, C, 3, 3).contiguous()
        # Reshape distortion coeffs to [B, C, ...] (or None).
        def _reshape_coeffs(t, last_dim):
            if t is None:
                return None
            return t.reshape(B, C, last_dim).contiguous()

        rad_b = _reshape_coeffs(
            radial_coeffs, radial_coeffs.shape[-1] if radial_coeffs is not None else 0
        )
        tan_b = _reshape_coeffs(tangential_coeffs, 2)
        prism_b = _reshape_coeffs(thin_prism_coeffs, 4)

        radii, means2d, depths, conics, compensations = (
            _projection_ut_3dgs_fused_fwd_metal(
                means_flat_b, quats_flat_b, scales_flat_b, op_flat_b,
                viewmats_flat_b, Ks_flat_b,
                width, height,
                eps2d, near_plane, far_plane, radius_clip,
                ut_params.alpha, ut_params.beta, ut_params.kappa,
                ut_params.in_image_margin_factor,
                calc_compensations,
                cm_id=_ut_cm_id,
                radial_coeffs=rad_b,
                tangential_coeffs=tan_b,
                thin_prism_coeffs=prism_b,
            )
        )
        radii = radii.reshape(batch_dims + (C, N, 2))
        means2d = means2d.reshape(batch_dims + (C, N, 2))
        depths = depths.reshape(batch_dims + (C, N))
        conics = conics.reshape(batch_dims + (C, N, 3))
        if compensations is not None:
            compensations = compensations.reshape(batch_dims + (C, N))
    else:
        from ._torch_impl_ut import _fully_fused_projection_with_ut

        radii, means2d, depths, conics, compensations = _fully_fused_projection_with_ut(
            means.contiguous(),
            quats.contiguous(),
            scales.contiguous(),
            opacities.contiguous() if opacities is not None else None,
            viewmats.contiguous(),
            Ks.contiguous(),
            width,
            height,
            eps2d=eps2d,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
            ut_params=ut_params,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            ftheta_coeffs=ftheta_coeffs,
            rolling_shutter=rolling_shutter,
            viewmats_rs=viewmats_rs,
            global_z_order=global_z_order,
        )
    if not calc_compensations:
        compensations = None
    return radii, means2d, depths, conics, compensations


class _RasterizeToPixels(torch.autograd.Function):
    """Rasterize gaussians"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,  # [..., N, 2] or [nnz, 2]
        conics: Tensor,  # [..., N, 3] or [nnz, 3]
        colors: Tensor,  # [..., N, channels] or [nnz, channels]
        opacities: Tensor,  # [..., N] or [nnz]
        backgrounds: Tensor,  # [..., channels], Optional
        masks: Tensor,  # [..., tile_height, tile_width], Optional
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [..., tile_height, tile_width]
        flatten_ids: Tensor,  # [n_isects]
        absgrad: bool,
    ) -> Tuple[Tensor, Tensor]:
        render_colors, render_alphas, last_ids = _make_lazy_cuda_func(
            "rasterize_to_pixels_3dgs_fwd"
        )(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )

        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad

        # double to float
        render_alphas = render_alphas.float()
        return render_colors, render_alphas

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,  # [..., H, W, 3]
        v_render_alphas: Tensor,  # [..., H, W, 1]
    ):
        (
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad

        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_bwd")(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            absgrad,
        )

        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[4]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(-3, -2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_backgrounds,
            None,  # masks
            None,  # width
            None,  # height
            None,  # tile_size
            None,  # isect_offsets
            None,  # flatten_ids
            None,  # absgrad
        )


class _RasterizeToPixelsEval3D(torch.autograd.Function):
    """Rasterize gaussians"""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [..., N, 3]
        quats: Tensor,  # [..., N, 4]
        scales: Tensor,  # [..., N, 3]
        colors: Tensor,  # [..., C, N, D] or [nnz, D]
        opacities: Tensor,  # [..., C, N] or [nnz]
        backgrounds: Tensor,  # [..., C, D], Optional
        masks: Tensor,  # [..., C, tile_height, tile_width], Optional
        viewmats: Tensor,  # [..., C, 4, 4]
        Ks: Tensor,  # [..., C, 3, 3]
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [..., C, tile_height, tile_width]
        flatten_ids: Tensor,  # [..., n_isects]
        camera_model: CameraModel = "pinhole",
        ut_params: Optional[UnscentedTransformParameters] = None,
        rays: Optional[Tensor] = None,  # [..., C, P, 6]
        # distortion
        radial_coeffs: Optional[Tensor] = None,  # [..., C, 6] or [..., C, 4]
        tangential_coeffs: Optional[Tensor] = None,  # [..., C, 2]
        thin_prism_coeffs: Optional[Tensor] = None,  # [..., C, 4]
        ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
        lidar_coeffs: Optional[
            RowOffsetStructuredSpinningLidarModelParametersExt
        ] = None,
        external_distortion_coeffs: Optional[BivariateWindshieldModelParameters] = None,
        # rolling shutter
        rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
        viewmats_rs: Optional[Tensor] = None,  # [..., C, 4, 4]
        return_sample_counts: bool = False,
        use_hit_distance: bool = False,
        return_normals: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        if ut_params is None:
            ut_params = UnscentedTransformParameters()

        camera_model_type = _make_lazy_cuda_obj(
            f"CameraModelType.{camera_model.upper()}"
        )
        ftheta_coeffs = (
            ftheta_coeffs
            if ftheta_coeffs is not None
            else FThetaCameraDistortionParameters()
        )

        lidar_coeffs = lidar_coeffs.to_cpp() if lidar_coeffs is not None else None

        # Extract batch_dims for sample_counts allocation
        batch_dims = means.shape[:-2]
        C = viewmats.size(-3)

        # Conditionally allocate sample_counts based on flag
        if return_sample_counts:
            # Allocate with correct final shape (batch_dims, C, H, W)
            sample_counts = torch.empty(
                batch_dims + (C, height, width), dtype=torch.int32, device=means.device
            )
        else:
            sample_counts = None

        # Conditionally allocate normals based on flag
        if return_normals:
            render_normals = torch.empty(
                batch_dims + (C, height, width, 3),
                dtype=torch.float32,
                device=means.device,
            )
        else:
            render_normals = None

        render_colors, render_alphas, last_ids = _make_lazy_cuda_func(
            "rasterize_to_pixels_from_world_3dgs_fwd"
        )(
            means,
            quats,
            scales,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            viewmats,
            viewmats_rs,
            Ks,
            camera_model_type,
            ut_params,
            rolling_shutter,
            rays,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            ftheta_coeffs,
            lidar_coeffs,
            external_distortion_coeffs,
            isect_offsets,
            flatten_ids,
            use_hit_distance,
            sample_counts,
            render_normals,
        )

        ctx.save_for_backward(
            means,
            quats,
            scales,
            colors,
            opacities,
            backgrounds,
            masks,
            viewmats,
            viewmats_rs,
            Ks,
            rays,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.ut_params = ut_params
        ctx.rs_type = rolling_shutter
        ctx.camera_model_type = camera_model_type
        ctx.tile_size = tile_size
        ctx.ftheta_coeffs = ftheta_coeffs
        ctx.lidar_coeffs = lidar_coeffs
        ctx.external_distortion_coeffs = external_distortion_coeffs
        ctx.use_hit_distance = use_hit_distance

        return render_colors, render_alphas, last_ids, sample_counts, render_normals

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,  # [..., C, H, W, 3]
        v_render_alphas: Tensor,  # [..., C, H, W, 1]
        v_last_ids: Optional[Tensor],  # None - last_ids is integer (non-differentiable)
        v_sample_counts: Optional[
            Tensor
        ],  # None - sample_counts is integer (non-differentiable)
        v_render_normals: Optional[Tensor],  # [..., C, H, W, 3]
    ):
        (
            means,
            quats,
            scales,
            colors,
            opacities,
            backgrounds,
            masks,
            viewmats,
            viewmats_rs,
            Ks,
            rays,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        ut_params = ctx.ut_params
        rs_type = ctx.rs_type
        camera_model_type = ctx.camera_model_type
        tile_size = ctx.tile_size
        ftheta_coeffs = ctx.ftheta_coeffs
        lidar_coeffs = ctx.lidar_coeffs
        external_distortion_coeffs = ctx.external_distortion_coeffs
        use_hit_distance = ctx.use_hit_distance

        (
            v_means,
            v_quats,
            v_scales,
            v_colors,
            v_opacities,
            v_rays,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_from_world_3dgs_bwd")(
            means,
            quats,
            scales,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            viewmats,
            viewmats_rs,
            Ks,
            camera_model_type,
            ut_params,
            rs_type,
            rays,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            ftheta_coeffs,
            lidar_coeffs,  # already converted to C++ in forward
            external_distortion_coeffs,
            isect_offsets,
            flatten_ids,
            use_hit_distance,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_render_normals.contiguous() if v_render_normals is not None else None,
        )

        if ctx.needs_input_grad[5]:  # backgrounds
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(-3, -2)
            )
        else:
            v_backgrounds = None

        # Check not needed anymore because we return v_rays directly
        # if ctx.needs_input_grad[7]:  # viewmats
        #    raise NotImplementedError

        return (
            v_means,
            v_quats,
            v_scales,
            v_colors,
            v_opacities,
            v_backgrounds,
            None,  # masks
            None,  # viewmats
            None,  # Ks
            None,  # width
            None,  # height
            None,  # tile_size
            None,  # isect_offsets
            None,  # flatten_ids
            None,  # camera_model
            None,  # ut_params
            v_rays,  # rays
            None,  # radial_coeffs
            None,  # tangential_coeffs
            None,  # thin_prism_coeffs
            None,  # ftheta_coeffs
            None,  # lidar_coeffs
            None,  # external_distortion_coeffs
            None,  # rolling_shutter
            None,  # viewmats_rs
            None,  # return_sample_counts (flag, no gradient)
            None,  # use_hit_distance
            None,  # return_normals (flag, no gradient)
        )


class _FullyFusedProjectionPacked(torch.autograd.Function):
    """Projects Gaussians to 2D. Return packed tensors."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [..., N, 3]
        covars: Tensor,  # [..., N, 6] or None
        quats: Tensor,  # [..., N, 4] or None
        scales: Tensor,  # [..., N, 3] or None
        viewmats: Tensor,  # [..., C, 4, 4]
        Ks: Tensor,  # [..., C, 3, 3]
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        sparse_grad: bool,
        calc_compensations: bool,
        camera_model: CameraModel = "pinhole",
        opacities: Optional[Tensor] = None,  # [..., N] or None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        assert (
            camera_model != "ftheta"
        ), "ftheta camera is only supported via UT, please set with_ut=True in the rasterization()"

        camera_model_type = _make_lazy_cuda_obj(
            f"CameraModelType.{camera_model.upper()}"
        )

        (
            indptr,
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = _make_lazy_cuda_func("projection_ewa_3dgs_packed_fwd")(
            means,
            covars,  # optional
            quats,  # optional
            scales,  # optional
            opacities,  # optional
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
            camera_model_type,
        )
        if not calc_compensations:
            compensations = None
        ctx.save_for_backward(
            batch_ids,
            camera_ids,
            gaussian_ids,
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            conics,
            compensations,
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d
        ctx.sparse_grad = sparse_grad
        ctx.camera_model_type = camera_model_type

        return (
            batch_ids,
            camera_ids,
            gaussian_ids,
            indptr,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        )

    @staticmethod
    def backward(
        ctx,
        v_batch_ids,
        v_camera_ids,
        v_gaussian_ids,
        v_indptr,
        v_radii,
        v_means2d,
        v_depths,
        v_conics,
        v_compensations,
    ):
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            conics,
            compensations,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        eps2d = ctx.eps2d
        sparse_grad = ctx.sparse_grad
        camera_model_type = ctx.camera_model_type

        if v_compensations is not None:
            v_compensations = v_compensations.contiguous()
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "projection_ewa_3dgs_packed_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            camera_model_type,
            batch_ids,
            camera_ids,
            gaussian_ids,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            ctx.needs_input_grad[4],  # viewmats_requires_grad
            sparse_grad,
        )

        if sparse_grad:
            batch_dims = means.shape[:-2]
            B = math.prod(batch_dims)
            N = means.shape[-2]
        if not ctx.needs_input_grad[0]:
            v_means = None
        else:
            if sparse_grad:
                # TODO: gaussian_ids is duplicated so not ideal.
                # An idea is to directly set the attribute (e.g., .sparse_grad) of
                # the tensor but this requires the tensor to be leaf node only. And
                # a customized optimizer would be needed in this case.
                v_means = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],
                    values=v_means,  # [nnz, 3]
                    size=means.shape,
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[1]:
            v_covars = None
        else:
            if sparse_grad:
                v_covars = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],
                    values=v_covars,  # [nnz, 6]
                    size=covars.shape,
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[2]:
            v_quats = None
        else:
            if sparse_grad:
                v_quats = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],
                    values=v_quats,  # [nnz, 4]
                    size=quats.shape,
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[3]:
            v_scales = None
        else:
            if sparse_grad:
                v_scales = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],
                    values=v_scales,  # [nnz, 3]
                    size=scales.shape,
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[4]:
            v_viewmats = None

        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            v_viewmats,
            None,  # Ks
            None,  # width
            None,  # height
            None,  # eps2d
            None,  # near_plane
            None,  # far_plane
            None,  # radius_clip
            None,  # calc_compensations
            None,  # sparse_grad
            None,  # camera_model
            None,  # ut_params
        )


class _SphericalHarmonics(torch.autograd.Function):
    """Spherical Harmonics"""

    @staticmethod
    def forward(
        ctx, sh_degree: int, dirs: Tensor, coeffs: Tensor, masks: Tensor
    ) -> Tensor:
        colors = _make_lazy_cuda_func("spherical_harmonics_fwd")(
            sh_degree, dirs, coeffs, masks
        )
        ctx.save_for_backward(dirs, coeffs, masks)
        ctx.sh_degree = sh_degree
        return colors

    @staticmethod
    def backward(ctx, v_colors: Tensor):
        dirs, coeffs, masks = ctx.saved_tensors
        sh_degree = ctx.sh_degree
        compute_v_dirs = ctx.needs_input_grad[1]
        v_coeffs, v_dirs = _make_lazy_cuda_func("spherical_harmonics_bwd")(
            sh_degree,
            dirs,
            coeffs,
            masks,
            v_colors.contiguous(),
            compute_v_dirs,
        )
        if not compute_v_dirs:
            v_dirs = None
        return (
            None,  # sh_degree
            v_dirs,
            v_coeffs,
            None,  # masks
        )


###### 2DGS ######
def fully_fused_projection_2dgs(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Prepare Gaussians for rasterization

    This function prepares ray-splat intersection matrices, computes
    per splat bounding box and 2D means in image space.

    Args:
        means: Gaussian means. [..., N, 3]
        quats: Quaternions (No need to be normalized). [..., N, 4].
        scales: Scales. [..., N, 3].
        viewmats: World-to-camera matrices. [..., C, 4, 4]
        Ks: Camera intrinsics. [..., C, 3, 3]
        width: Image width.
        height: Image height.
        near_plane: Near plane distance. Default: 0.01.
        far_plane: Far plane distance. Default: 200.
        radius_clip: Gaussians with projected radii smaller than this value will be ignored. Default: 0.0.
        packed: If True, the output tensors will be packed into a flattened tensor. Default: False.
        sparse_grad (Experimental): This is only effective when `packed` is True. If True, during backward the gradients
          of {`means`, `covars`, `quats`, `scales`} will be a sparse Tensor in COO layout. Default: False.

    Returns:
        A tuple:

        If `packed` is True:

        - **batch_ids**. The batch indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **camera_ids**. The camera indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **gaussian_ids**. The column indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [nnz, 2].
        - **means**. Projected Gaussian means in 2D. [nnz, 2]
        - **depths**. The z-depth of the projected Gaussians. [nnz]
        - **ray_transforms**. transformation matrices that transforms xy-planes in pixel spaces into splat coordinates (WH)^T in equation (9) in paper [nnz, 3, 3]
        - **normals**. The normals in camera spaces. [nnz, 3]

        If `packed` is False:

        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [..., C, N, 2].
        - **means**. Projected Gaussian means in 2D. [..., C, N, 2]
        - **depths**. The z-depth of the projected Gaussians. [..., C, N]
        - **ray_transforms**. transformation matrices that transforms xy-planes in pixel spaces into splat coordinates [..., C, N, 3, 3]
        - **normals**. The normals in camera spaces. [..., C, N, 3]

    """
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == batch_dims + (N, 3), means.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    means = means.contiguous()
    assert quats is not None, "quats is required"
    assert scales is not None, "scales is required"
    assert quats.shape == batch_dims + (N, 4), quats.shape
    assert scales.shape == batch_dims + (N, 3), scales.shape
    quats = quats.contiguous()
    scales = scales.contiguous()
    if sparse_grad:
        assert packed, "sparse_grad is only supported when packed is True"

    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()

    from ._dispatch import select_backend, Backend

    if (
        select_backend(means, "projection_2dgs_fused") is Backend.METAL
        and not packed
        and len(batch_dims) <= 1
    ):
        from ..mps._projection_2dgs import fully_fused_projection_2dgs_metal

        if len(batch_dims) == 0:
            means_b = means.unsqueeze(0)
            quats_b = quats.unsqueeze(0)
            scales_b = scales.unsqueeze(0)
            viewmats_b = viewmats.unsqueeze(0)
            Ks_b = Ks.unsqueeze(0)
        else:
            means_b, quats_b, scales_b, viewmats_b, Ks_b = (
                means, quats, scales, viewmats, Ks,
            )
        radii_b, m2d_b, d_b, rt_b, n_b = fully_fused_projection_2dgs_metal(
            means_b, quats_b, scales_b, viewmats_b, Ks_b,
            width, height, near_plane, far_plane, radius_clip,
        )
        if len(batch_dims) == 0:
            return (
                radii_b.squeeze(0), m2d_b.squeeze(0), d_b.squeeze(0),
                rt_b.squeeze(0), n_b.squeeze(0),
            )
        return radii_b, m2d_b, d_b, rt_b, n_b

    from ._torch_impl_2dgs import _fully_fused_projection_2dgs

    radii, means2d, depths, ray_transforms, normals = _fully_fused_projection_2dgs(
        means,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        near_plane=near_plane,
        far_plane=far_plane,
    )

    if not packed:
        return radii, means2d, depths, ray_transforms, normals

    valid = (radii > 0).all(dim=-1)
    batch_dims_v = valid.shape[:-2]
    B = math.prod(batch_dims_v) if batch_dims_v else 1
    C, N_ = valid.shape[-2:]
    valid_f = valid.reshape(B, C, N_)
    nz = torch.nonzero(valid_f, as_tuple=False)
    batch_ids = nz[:, 0].to(torch.int32)
    camera_ids = nz[:, 1].to(torch.int32)
    gaussian_ids = nz[:, 2].to(torch.int32)

    def _gather(t):
        flat = t.reshape((B, C, N_) + t.shape[len(batch_dims_v) + 2 :])
        return flat[batch_ids.long(), camera_ids.long(), gaussian_ids.long()]

    return (
        batch_ids,
        camera_ids,
        gaussian_ids,
        _gather(radii),
        _gather(means2d),
        _gather(depths),
        _gather(ray_transforms),
        _gather(normals),
    )


class _FullyFusedProjection2DGS(torch.autograd.Function):
    """Projects Gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [..., N, 3]
        quats: Tensor,  # [..., N, 4]
        scales: Tensor,  # [..., N, 3]
        viewmats: Tensor,  # [..., C, 4, 4]
        Ks: Tensor,  # [..., C, 3, 3]
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        radii, means2d, depths, ray_transforms, normals = _make_lazy_cuda_func(
            "projection_2dgs_fused_fwd"
        )(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
        )
        ctx.save_for_backward(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            radii,
            ray_transforms,
            normals,
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d

        return radii, means2d, depths, ray_transforms, normals

    @staticmethod
    def backward(ctx, v_radii, v_means2d, v_depths, v_ray_transforms, v_normals):
        (
            means,
            quats,
            scales,
            viewmats,
            Ks,
            radii,
            ray_transforms,
            normals,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        eps2d = ctx.eps2d
        v_means, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "projection_2dgs_fused_bwd"
        )(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            radii,
            ray_transforms,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_normals.contiguous(),
            v_ray_transforms.contiguous(),
            ctx.needs_input_grad[3],  # viewmats_requires_grad
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_quats = None
        if not ctx.needs_input_grad[2]:
            v_scales = None
        if not ctx.needs_input_grad[3]:
            v_viewmats = None

        return (
            v_means,
            v_quats,
            v_scales,
            v_viewmats,
            None,  # Ks
            None,  # width
            None,  # height
            None,  # eps2d
            None,  # near_plane
            None,  # far_plane
            None,  # radius_clip
            None,  # camera_model
        )


class _FullyFusedProjectionPacked2DGS(torch.autograd.Function):
    """Projects Gaussians to 2D. Return packed tensors."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [..., N, 3]
        quats: Tensor,  # [..., N, 4]
        scales: Tensor,  # [..., N, 3]
        viewmats: Tensor,  # [..., C, 4, 4]
        Ks: Tensor,  # [..., C, 3, 3]
        width: int,
        height: int,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        sparse_grad: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        (
            indptr,
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            ray_transforms,
            normals,
        ) = _make_lazy_cuda_func("projection_2dgs_packed_fwd")(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            near_plane,
            far_plane,
            radius_clip,
        )
        ctx.save_for_backward(
            batch_ids,
            camera_ids,
            gaussian_ids,
            means,
            quats,
            scales,
            viewmats,
            Ks,
            ray_transforms,
        )
        ctx.width = width
        ctx.height = height
        ctx.sparse_grad = sparse_grad

        return (
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            ray_transforms,
            normals,
        )

    @staticmethod
    def backward(
        ctx,
        v_batch_ids,
        v_camera_ids,
        v_gaussian_ids,
        v_radii,
        v_means2d,
        v_depths,
        v_ray_transforms,
        v_normals,
    ):
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            means,
            quats,
            scales,
            viewmats,
            Ks,
            ray_transforms,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        sparse_grad = ctx.sparse_grad

        v_means, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "projection_2dgs_packed_bwd"
        )(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            batch_ids,
            camera_ids,
            gaussian_ids,
            ray_transforms,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_ray_transforms.contiguous(),
            v_normals.contiguous(),
            ctx.needs_input_grad[3],  # viewmats_requires_grad
            sparse_grad,
        )

        if sparse_grad:
            batch_dims = means.shape[:-2]
            B = math.prod(batch_dims)
            N = means.shape[-2]

        if not ctx.needs_input_grad[0]:
            v_means = None
        else:
            if sparse_grad:
                # TODO: gaussian_ids is duplicated so not ideal.
                # An idea is to directly set the attribute (e.g., .sparse_grad) of
                # the tensor but this requires the tensor to be leaf node only. And
                # a customized optimizer would be needed in this case.
                v_means = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],
                    values=v_means,  # [nnz, 3]
                    size=means.shape,
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[1]:
            v_quats = None
        else:
            if sparse_grad:
                v_quats = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],
                    values=v_quats,  # [nnz, 4]
                    size=quats.shape,
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[2]:
            v_scales = None
        else:
            if sparse_grad:
                v_scales = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],
                    values=v_scales,  # [nnz, 3]
                    size=scales.shape,
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[3]:
            v_viewmats = None

        return (
            v_means,
            v_quats,
            v_scales,
            v_viewmats,
            None,  # Ks
            None,  # width
            None,  # height
            None,  # eps2d
            None,  # near_plane
            None,  # far_plane
            None,  # radius_clip
            None,  # sparse_grad
            None,  # camera_model
        )


def rasterize_to_pixels_2dgs(
    means2d: Tensor,  # [..., N, 2]
    ray_transforms: Tensor,  # [..., N, 3, 3]
    colors: Tensor,  # [..., N, channels]
    opacities: Tensor,  # [..., N]
    normals: Tensor,  # [..., N, 3]
    densify: Tensor,  # [..., N, 2]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., channels]
    masks: Optional[Tensor] = None,  # [..., tile_height, tile_width]
    packed: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Rasterize Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [..., N, 2] if packed is False, [nnz, 2] if packed is True.
        ray_transforms: transformation matrices that transforms xy-planes in pixel spaces into splat coordinates. [..., N, 3, 3] if packed is False, [nnz, channels] if packed is True.
        colors: Gaussian colors or ND features. [..., N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [..., N] if packed is False, [nnz] if packed is True.
        normals: The normals in camera space. [..., N, 3] if packed is False, [nnz, 3] if packed is True.
        densify: Dummy variable to keep track of gradient for densification. [..., N, 2] if packed, [nnz, 3] if packed is True.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [..., tile_height, tile_width]
        flatten_ids: The global flatten indices in [I * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [..., channels]. Default: None.
        masks: Optional tile mask to skip rendering GS to masked tiles. [..., tile_height, tile_width]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

    Returns:
        A tuple:

        - **Rendered colors**.      [..., image_height, image_width, channels]
        - **Rendered alphas**.      [..., image_height, image_width, 1]
        - **Rendered normals**.     [..., image_height, image_width, 3]
        - **Rendered distortion**.  [..., image_height, image_width, 1]
        - **Rendered median depth**.[..., image_height, image_width, 1]


    """
    image_dims = means2d.shape[:-2]
    channels = colors.shape[-1]
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert ray_transforms.shape == (nnz, 3, 3), ray_transforms.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(-2)
        assert means2d.shape == image_dims + (N, 2), means2d.shape
        assert ray_transforms.shape == image_dims + (N, 3, 3), ray_transforms.shape
        # Colors may arrive as `[..., N, channels]` (without the per-camera
        # dim); broadcast across `image_dims` so the inner rasterizer always
        # sees `image_dims + (N, channels)`.
        target_color_shape = image_dims + (N, channels)
        if colors.shape != target_color_shape:
            # insert leading dims of size 1 then expand
            extra = len(target_color_shape) - colors.dim()
            if extra > 0:
                colors = colors[(None,) * extra]
            colors = colors.expand(target_color_shape).contiguous()
        assert colors.shape == target_color_shape, colors.shape
        assert opacities.shape == image_dims + (N,), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == image_dims + (channels,), backgrounds.shape
        backgrounds = backgrounds.contiguous()

    # Pad the channels to the nearest supported number if necessary
    if channels > 512 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        # Make sure the depth (last channel if present) remains in the last channel after padding (for depth distortion and median depth in CUDA kernel)
        colors = torch.cat(
            [
                colors[..., :-1],
                torch.empty(*colors.shape[:-1], padded_channels, device=device),
                colors[..., -1:],
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0
    tile_height, tile_width = isect_offsets.shape[-2:]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    if packed:
        raise NotImplementedError(
            "rasterize_to_pixels_2dgs(packed=True) is not yet wired in the MPS port. "
            "Use packed=False for the time being."
        )

    # Phase B: route MPS tensors through the Metal forward kernel when the
    # input shape matches what the kernel supports (CDIM=3, tile_size=16,
    # no backgrounds / no masks). Anything else falls back to the torch
    # rasterizer below. The Metal forward returns colors / alphas / normals
    # only; distortion and median are zeros (out-of-scope for now).
    from ._dispatch import select_backend, Backend

    if (
        select_backend(means2d, "rasterize_to_pixels_2dgs_fwd") is Backend.METAL
        and tile_size == 16
        and channels == 3
        and not packed
        and backgrounds is None
        and masks is None
    ):
        # Match torch ref's dim collapsing: collapse image_dims to a single I.
        import math as _math
        I = _math.prod(image_dims) if image_dims else 1
        N = means2d.shape[-2]
        tile_h, tile_w = isect_offsets.shape[-2:]

        m2d_f = means2d.contiguous().reshape(I, N, 2)
        rt_f = ray_transforms.contiguous().reshape(I, N, 3, 3)
        col_f = colors.contiguous().reshape(I, N, channels)
        op_f = opacities.contiguous().reshape(I, N)
        nrm_f = normals.contiguous().reshape(I, N, 3)
        iso_f = isect_offsets.contiguous().reshape(I, tile_h, tile_w)

        from ..mps._rasterize_2dgs import rasterize_to_pixels_2dgs_metal

        render_colors_f, render_alphas_f, render_normals_f = (
            rasterize_to_pixels_2dgs_metal(
                m2d_f, rt_f, col_f, op_f, nrm_f,
                image_width, image_height, tile_size,
                iso_f, flatten_ids.contiguous(),
            )
        )
        render_colors = render_colors_f.reshape(image_dims + (image_height, image_width, channels))
        render_alphas = render_alphas_f.reshape(image_dims + (image_height, image_width, 1))
        render_normals = render_normals_f.reshape(image_dims + (image_height, image_width, 3))
        zero_dm_shape = image_dims + (image_height, image_width, 1)
        render_distort = torch.zeros(zero_dm_shape, device=device, dtype=means2d.dtype)
        render_median = torch.zeros(zero_dm_shape, device=device, dtype=means2d.dtype)

        if padded_channels > 0:
            render_colors = torch.cat(
                [render_colors[..., : -padded_channels - 1], render_colors[..., -1:]],
                dim=-1,
            )
        return render_colors, render_alphas, render_normals, render_distort, render_median

    from ._torch_rasterize import _rasterize_to_pixels_2dgs_torch

    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
    ) = _rasterize_to_pixels_2dgs_torch(
        means2d.contiguous(),
        ray_transforms.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        normals.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        backgrounds=backgrounds,
        masks=masks,
    )

    if padded_channels > 0:
        render_colors = torch.cat(
            [render_colors[..., : -padded_channels - 1], render_colors[..., -1:]],
            dim=-1,
        )

    return render_colors, render_alphas, render_normals, render_distort, render_median


@torch.no_grad()
def rasterize_to_indices_in_range_2dgs(
    range_start: int,
    range_end: int,
    transmittances: Tensor,  # [..., image_height, image_width]
    means2d: Tensor,  # [..., N, 2]
    ray_transforms: Tensor,  # [..., N, 3, 3]
    opacities: Tensor,  # [..., N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Rasterizes a batch of Gaussians to images but only returns the indices.

    .. note::

        This function supports iterative rasterization, in which each call of this function
        will rasterize a batch of Gaussians from near to far, defined by `[range_start, range_end)`.
        If a one-step full rasterization is desired, set `range_start` to 0 and `range_end` to a really
        large number, e.g, 1e10.

    Args:
        range_start: The start batch of Gaussians to be rasterized (inclusive).
        range_end: The end batch of Gaussians to be rasterized (exclusive).
        transmittances: Currently transmittances. [..., image_height, image_width]
        means2d: Projected Gaussian means. [..., N, 2]
        ray_transforms: transformation matrices that transforms xy-planes in pixel spaces into splat coordinates. [..., N, 3, 3]
        opacities: Gaussian opacities that support per-view values. [..., N]
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [..., tile_height, tile_width]
        flatten_ids: The global flatten indices in [I * N] from  `isect_tiles()`. [n_isects]

    Returns:
        A tuple:

        - **Gaussian ids**. Gaussian ids for the pixel intersection. A flattened list of shape [M].
        - **Pixel ids**. pixel indices (row-major). A flattened list of shape [M].
        - **Camera ids**. Camera indices. A flattened list of shape [M].
        - **Batch ids**. Batch indices. A flattened list of shape [M].
    """

    image_dims = means2d.shape[:-2]
    tile_height, tile_width = isect_offsets.shape[-2:]
    N = means2d.shape[-2]
    assert transmittances.shape == image_dims + (
        image_height,
        image_width,
    ), transmittances.shape
    assert means2d.shape == image_dims + (N, 2), means2d.shape
    assert ray_transforms.shape == image_dims + (N, 3, 3), ray_transforms.shape
    assert opacities.shape == image_dims + (N,), opacities.shape
    assert isect_offsets.shape == image_dims + (
        tile_height,
        tile_width,
    ), isect_offsets.shape
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    out_gauss_ids, out_indices = _make_lazy_cuda_func("rasterize_to_indices_2dgs")(
        range_start,
        range_end,
        transmittances.contiguous(),
        means2d.contiguous(),
        ray_transforms.contiguous(),
        opacities.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
    )
    out_pixel_ids = out_indices % (image_width * image_height)
    out_image_ids = out_indices // (image_width * image_height)
    return out_gauss_ids, out_pixel_ids, out_image_ids


class _RasterizeToPixels2DGS(torch.autograd.Function):
    """Rasterize gaussians 2DGS"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,
        ray_transforms: Tensor,
        colors: Tensor,
        opacities: Tensor,
        normals: Tensor,
        densify: Tensor,
        backgrounds: Tensor,
        masks: Tensor,
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,
        flatten_ids: Tensor,
        absgrad: bool,
        distloss: bool,
    ) -> Tuple[Tensor, Tensor]:
        (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            last_ids,
            median_ids,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_2dgs_fwd")(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )

        ctx.save_for_backward(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.distloss = distloss

        # double to float
        render_alphas = render_alphas.float()
        return (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
        )

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,
        v_render_alphas: Tensor,
        v_render_normals: Tensor,
        v_render_distort: Tensor,
        v_render_median: Tensor,
    ):

        (
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad

        (
            v_means2d_abs,
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_2dgs_bwd")(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_render_normals.contiguous(),
            v_render_distort.contiguous(),
            v_render_median.contiguous(),
            absgrad,
        )
        from ._backend import _sync

        _sync(means2d.device)
        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[6]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(-3, -2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify,
            v_backgrounds,
            None,  # masks
            None,  # width
            None,  # height
            None,  # tile_size
            None,  # isect_offsets
            None,  # flatten_ids
            None,  # absgrad
            None,  # distloss
        )
