// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused 3DGUT projection (forward) — supports global-shutter pinhole
// (with optional OpenCV radial / tangential / thin-prism distortion) and
// OpenCV-fisheye (4 radial coefficients). ftheta and rolling-shutter fall
// back to the torch reference at the host wrapper layer.
//
// camera_model_id selector:
//   0  pinhole, no distortion          (matches `_PerfectPinholeCameraModel`)
//   1  OpenCV-pinhole + radial[6] + tangential[2] + thin-prism[4]
//                                       (matches `_OpenCVPinholeCameraModel`)
//   2  OpenCV-fisheye + radial[4]       (matches `_OpenCVFisheyeCameraModel`)
//
// Algorithm (per gauss):
//   D = 3, λ = α²(D+κ) - D
//   wm_c = λ/(D+λ)                  (center weight, mean)
//   wc_c = λ/(D+λ) + (1 - α² + β)   (center weight, cov)
//   w_o  = 1/(2(D+λ))               (offset weight)
//
//   1. R_q = quat_to_rotmat(quats / |quats|)
//   2. deltas[j] = sqrt(D+λ) * R_q[:, j] * scales[j]   for j in 0..2
//   3. sigma points = [mean] ++ [mean + deltas[j]] ++ [mean - deltas[j]]   (7 total)
//   4. For each sigma point: transform to camera space (R_view·point + t),
//      check (depth > 0), project (u = fx*x/z + cx, v = fy*y/z + cy),
//      check image bounds with margin_factor.
//   5. valid_gaussian = ANY(valid_points)  for require_all_sigma_points_valid=false
//   6. mean_2d = sum_i w_mean[i] * points_2d[i]
//   7. cov_2d = sum_i w_cov[i] * (points_2d[i] - mean_2d) * (points_2d[i] - mean_2d)^T
//   8. cov_2d_blur = cov_2d + eps2d * I
//   9. conics = inv(cov_2d_blur)
//   10. radii from cov_2d diagonals, image-bounds cull.

#include <metal_stdlib>
using namespace metal;

constant constexpr float DEFAULT_EXTEND   = 3.33f;
constant constexpr float ALPHA_THRESHOLD  = 1.0f / 255.0f;
constant constexpr float MIN_COMPENSATION = 0.1f;

inline float3x3 quat_to_rotmat_norm(float w, float x, float y, float z) {
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    return float3x3(
        float3(1.0f - 2.0f * (y2 + z2), 2.0f * (xy + wz),       2.0f * (xz - wy)),
        float3(2.0f * (xy - wz),        1.0f - 2.0f * (x2 + z2), 2.0f * (yz + wx)),
        float3(2.0f * (xz + wy),        2.0f * (yz - wx),       1.0f - 2.0f * (x2 + y2))
    );
}

// Project a camera-space point to image coordinates with optional OpenCV
// pinhole / fisheye distortion. Sets `valid` based on (a) depth>0 and
// (b) per-model validity heuristics (icD>0.8 for pinhole-distorted, delta>0
// for fisheye).
inline float2 project_to_image(
    uint   cm_id,
    float3 cam,
    float fx, float fy, float cx, float cy,
    device const float* radial_per_cam,      // 6 floats (only first 4 used for fisheye)
    device const float* tangential_per_cam,  // 2 floats
    device const float* thin_prism_per_cam,  // 4 floats
    thread bool& valid
) {
    if (cam.z <= 0.0f) {
        valid = false;
        return float2(0.0f, 0.0f);
    }
    if (cm_id == 2u) {
        // OpenCV fisheye.
        const float xy_norm = sqrt(cam.x * cam.x + cam.y * cam.y);
        if (xy_norm <= 0.0f) {
            valid = true;  // center point projects to (cx, cy)
            return float2(cx, cy);
        }
        const float theta = atan2(xy_norm, cam.z);
        const float k1 = radial_per_cam[0];
        const float k2 = radial_per_cam[1];
        const float k3 = radial_per_cam[2];
        const float k4 = radial_per_cam[3];
        const float t2 = theta * theta;
        const float t3 = theta * t2;
        const float t5 = t3 * t2;
        const float t7 = t5 * t2;
        const float t9 = t7 * t2;
        const float poly = theta + k1 * t3 + k2 * t5 + k3 * t7 + k4 * t9;
        const float delta = poly / xy_norm;
        valid = (delta > 0.0f);
        const float u = delta * cam.x;
        const float v = delta * cam.y;
        return float2(fx * u + cx, fy * v + cy);
    }
    if (cm_id == 1u) {
        // OpenCV pinhole + distortion.
        const float u = cam.x / cam.z;
        const float v = cam.y / cam.z;
        const float r2 = u * u + v * v;
        const float a1 = 2.0f * u * v;
        const float a2 = r2 + 2.0f * u * u;
        const float a3 = r2 + 2.0f * v * v;
        const float k1 = radial_per_cam[0];
        const float k2 = radial_per_cam[1];
        const float k3 = radial_per_cam[2];
        const float k4 = radial_per_cam[3];
        const float k5 = radial_per_cam[4];
        const float k6 = radial_per_cam[5];
        const float p1 = tangential_per_cam[0];
        const float p2 = tangential_per_cam[1];
        const float s1 = thin_prism_per_cam[0];
        const float s2 = thin_prism_per_cam[1];
        const float s3 = thin_prism_per_cam[2];
        const float s4 = thin_prism_per_cam[3];
        const float icD_num = 1.0f + r2 * (k1 + r2 * (k2 + r2 * k3));
        const float icD_den = 1.0f + r2 * (k4 + r2 * (k5 + r2 * k6));
        const float icD = icD_num / icD_den;
        valid = (icD > 0.8f);
        const float dx = p1 * a1 + p2 * a2 + r2 * (s1 + r2 * s2);
        const float dy = p1 * a3 + p2 * a1 + r2 * (s3 + r2 * s4);
        const float u_d = icD * u + dx;
        const float v_d = icD * v + dy;
        return float2(fx * u_d + cx, fy * v_d + cy);
    }
    // cm_id == 0: pure pinhole.
    valid = true;
    const float rz = 1.0f / cam.z;
    return float2(fx * cam.x * rz + cx, fy * cam.y * rz + cy);
}

kernel void projection_ut_3dgs_fused_fwd(
    // outputs
    device int*                 radii          [[buffer(0)]],   // [B*C*N, 2]
    device float*               means2d        [[buffer(1)]],   // [B*C*N, 2]
    device float*               depths         [[buffer(2)]],   // [B*C*N]
    device float*               conics         [[buffer(3)]],   // [B*C*N, 3]
    device float*               compensations  [[buffer(4)]],   // [B*C*N]   (only written if has_compensations==1)
    // inputs
    device const packed_float3* means          [[buffer(5)]],   // [B*N, 3]
    device const float*         quats          [[buffer(6)]],   // [B*N, 4]
    device const packed_float3* scales         [[buffer(7)]],   // [B*N, 3]
    device const float*         opacities      [[buffer(8)]],   // [B*N] (only read if has_opacities==1)
    device const float*         viewmats       [[buffer(9)]],   // [B*C, 16]
    device const float*         Ks             [[buffer(10)]],  // [B*C, 9]
    // shape / constants
    constant uint&              B              [[buffer(11)]],
    constant uint&              C              [[buffer(12)]],
    constant uint&              N              [[buffer(13)]],
    constant uint&              image_width    [[buffer(14)]],
    constant uint&              image_height   [[buffer(15)]],
    constant float&             eps2d          [[buffer(16)]],
    constant float&             near_plane     [[buffer(17)]],
    constant float&             far_plane      [[buffer(18)]],
    constant float&             radius_clip    [[buffer(19)]],
    constant float&             ut_alpha       [[buffer(20)]],
    constant float&             ut_beta        [[buffer(21)]],
    constant float&             ut_kappa       [[buffer(22)]],
    constant float&             margin_factor  [[buffer(23)]],
    constant uint&              has_opacities  [[buffer(24)]],
    constant uint&              has_compensations [[buffer(25)]],
    // Camera-model dispatch + distortion coefficients. Buffers are always
    // bound (with zeros for unused models) so MSL doesn't need conditional
    // attribute decls. cm_id=0 ignores all three coefficient buffers.
    constant uint&              cm_id          [[buffer(26)]],
    device const float*         radial_coeffs  [[buffer(27)]],   // [B*C, 6]
    device const float*         tangential_coeffs [[buffer(28)]],// [B*C, 2]
    device const float*         thin_prism_coeffs [[buffer(29)]],// [B*C, 4]
    uint                        idx            [[thread_position_in_grid]]
) {
    if (idx >= B * C * N) {
        return;
    }
    const uint bid = idx / (C * N);
    const uint cid = (idx / N) % C;
    const uint gid = idx % N;
    const uint mn_off = bid * N + gid;

    // Default-initialize outputs.
    radii[idx * 2 + 0] = 0;
    radii[idx * 2 + 1] = 0;
    means2d[idx * 2 + 0] = 0.0f;
    means2d[idx * 2 + 1] = 0.0f;
    depths[idx] = 0.0f;
    conics[idx * 3 + 0] = 0.0f;
    conics[idx * 3 + 1] = 0.0f;
    conics[idx * 3 + 2] = 0.0f;
    if (has_compensations != 0u) {
        compensations[idx] = 0.0f;
    }

    // UT params.
    constexpr float Df = 3.0f;
    const float lambda_ut = ut_alpha * ut_alpha * (Df + ut_kappa) - Df;
    const float D_plus_lambda = Df + lambda_ut;
    if (D_plus_lambda <= 0.0f) {
        return;  // invalid UT params
    }
    const float wm_center = lambda_ut / D_plus_lambda;
    const float wc_center = lambda_ut / D_plus_lambda + (1.0f - ut_alpha * ut_alpha + ut_beta);
    const float w_offset  = 1.0f / (2.0f * D_plus_lambda);
    const float sqrt_dpl = sqrt(D_plus_lambda);

    // Reconstruct R_view, t_view, K.
    const uint vm_off = (bid * C + cid) * 16u;
    const float v0 = viewmats[vm_off + 0],  v1 = viewmats[vm_off + 1];
    const float v2 = viewmats[vm_off + 2],  v3 = viewmats[vm_off + 3];
    const float v4 = viewmats[vm_off + 4],  v5 = viewmats[vm_off + 5];
    const float v6 = viewmats[vm_off + 6],  v7 = viewmats[vm_off + 7];
    const float v8 = viewmats[vm_off + 8],  v9 = viewmats[vm_off + 9];
    const float v10 = viewmats[vm_off + 10], v11 = viewmats[vm_off + 11];
    const float3x3 R_view = float3x3(
        float3(v0, v4, v8), float3(v1, v5, v9), float3(v2, v6, v10)
    );
    const float3 t_view = float3(v3, v7, v11);

    const uint k_off = (bid * C + cid) * 9u;
    const float fx = Ks[k_off + 0];
    const float fy = Ks[k_off + 4];
    const float cx = Ks[k_off + 2];
    const float cy = Ks[k_off + 5];

    // Center frustum-cull: based on the gauss center only (matches torch).
    const float3 mean_w = float3(means[mn_off]);
    const float3 mean_c = R_view * mean_w + t_view;
    const float center_z = mean_c.z;
    const bool in_frustum = (center_z >= near_plane) && (center_z <= far_plane);

    // Cull degenerate quaternions / scales.
    const float qw_raw = quats[mn_off * 4 + 0];
    const float qx_raw = quats[mn_off * 4 + 1];
    const float qy_raw = quats[mn_off * 4 + 2];
    const float qz_raw = quats[mn_off * 4 + 3];
    const float qnsq = qw_raw * qw_raw + qx_raw * qx_raw
                     + qy_raw * qy_raw + qz_raw * qz_raw;
    const float3 s_raw = float3(scales[mn_off]);
    const bool valid_quat = qnsq > 1e-30f;
    const bool valid_scale = (s_raw.x > 1e-30f) && (s_raw.y > 1e-30f) && (s_raw.z > 1e-30f);
    if (!in_frustum || !valid_quat || !valid_scale) {
        depths[idx] = center_z;
        return;
    }

    const float inv_qn = rsqrt(qnsq);
    const float qw = qw_raw * inv_qn;
    const float qx = qx_raw * inv_qn;
    const float qy = qy_raw * inv_qn;
    const float qz = qz_raw * inv_qn;
    const float3x3 R_q = quat_to_rotmat_norm(qw, qx, qy, qz);
    const float3 delta_x = sqrt_dpl * R_q[0] * s_raw.x;
    const float3 delta_y = sqrt_dpl * R_q[1] * s_raw.y;
    const float3 delta_z = sqrt_dpl * R_q[2] * s_raw.z;

    // Sigma points (world space).
    float3 sp[7];
    sp[0] = mean_w;
    sp[1] = mean_w + delta_x;
    sp[2] = mean_w + delta_y;
    sp[3] = mean_w + delta_z;
    sp[4] = mean_w - delta_x;
    sp[5] = mean_w - delta_y;
    sp[6] = mean_w - delta_z;

    // Project each sigma point. Validity = (depth>0) & (within margin-extended image bounds).
    // Image bounds margin: torch's `check_image_bounds` uses
    //   x in [-margin*W, (1+margin)*W) ; same for y. (Confirmed from
    //   `_BaseCameraModel.check_image_bounds`.)
    const float xmin = -margin_factor * (float)image_width;
    const float xmax = (1.0f + margin_factor) * (float)image_width;
    const float ymin = -margin_factor * (float)image_height;
    const float ymax = (1.0f + margin_factor) * (float)image_height;

    // Per-camera distortion coefficient pointers.
    const uint cam_lin = bid * C + cid;
    device const float* rad_per_cam   = radial_coeffs    + cam_lin * 6u;
    device const float* tan_per_cam   = tangential_coeffs + cam_lin * 2u;
    device const float* prism_per_cam = thin_prism_coeffs + cam_lin * 4u;

    float2 pts2d[7];
    bool   valid_pts[7];
    bool   any_valid = false;
    for (uint i = 0; i < 7; ++i) {
        const float3 cam = R_view * sp[i] + t_view;
        bool ok_proj = false;
        const float2 p = project_to_image(
            cm_id, cam, fx, fy, cx, cy,
            rad_per_cam, tan_per_cam, prism_per_cam, ok_proj
        );
        pts2d[i] = ok_proj ? p : float2(0.0f, 0.0f);
        const bool ok_xy = (p.x >= xmin) && (p.x < xmax) && (p.y >= ymin) && (p.y < ymax);
        valid_pts[i] = ok_proj && ok_xy;
        any_valid = any_valid || valid_pts[i];
    }
    if (!any_valid) {
        depths[idx] = center_z;
        return;
    }

    // Weighted mean (use ALL sigma points; matches `require_all_sigma_points_valid=false`).
    float2 mean_2d = wm_center * pts2d[0]
                  + w_offset * (pts2d[1] + pts2d[2] + pts2d[3] + pts2d[4] + pts2d[5] + pts2d[6]);

    // Weighted covariance (2x2 sym).
    float c00 = 0.0f, c01 = 0.0f, c11 = 0.0f;
    for (uint i = 0; i < 7; ++i) {
        const float w_i = (i == 0) ? wc_center : w_offset;
        const float dx = pts2d[i].x - mean_2d.x;
        const float dy = pts2d[i].y - mean_2d.y;
        c00 += w_i * dx * dx;
        c01 += w_i * dx * dy;
        c11 += w_i * dy * dy;
    }

    // Compensation: sqrt(det_orig / det_blur) clamped.
    const float det_orig = c00 * c11 - c01 * c01;
    c00 += eps2d;
    c11 += eps2d;
    const float det_blur = c00 * c11 - c01 * c01;
    if (det_blur <= 0.0f) {
        depths[idx] = center_z;
        return;
    }

    // The UT center cov weight can be very negative (e.g. ≈ -96 with α=0.1),
    // so the UT covariance is not guaranteed PSD even after blur. Cull
    // gauss with non-positive diagonals.
    if (c00 <= 0.0f || c11 <= 0.0f) {
        depths[idx] = center_z;
        return;
    }

    const float inv_det = 1.0f / det_blur;
    const float i00 =  c11 * inv_det;
    const float i01 = -c01 * inv_det;
    const float i11 =  c00 * inv_det;

    // Opacity-aware extent (matches the torch reference).
    float extend = DEFAULT_EXTEND;
    if (has_opacities != 0u) {
        const float opac = opacities[mn_off];
        // Compensation factor (det_orig / det_blur clamped).
        const float comp = sqrt(max(det_orig / det_blur, MIN_COMPENSATION * MIN_COMPENSATION));
        const float opacity = opac * comp;
        if (opacity < ALPHA_THRESHOLD) {
            depths[idx] = center_z;
            return;
        }
        extend = min(DEFAULT_EXTEND, sqrt(2.0f * log(max(opacity / ALPHA_THRESHOLD, 1.0f))));
    }

    // Eigenvalue-based tight bounding radius:
    //   trace = c00 + c11 ; b = trace/2 ; v1 = b + sqrt(max(b² - det, 0.01))
    //   r1 = extend * sqrt(max(v1, 0))
    //   r_x = min(extend*sqrt(c00), r1)  ;  r_y = min(extend*sqrt(c11), r1)
    const float trace2 = 0.5f * (c00 + c11);
    const float disc = max(trace2 * trace2 - det_blur, 0.01f);
    const float lambda1 = trace2 + sqrt(disc);
    const float r1 = extend * sqrt(max(lambda1, 0.0f));
    const float r_x = ceil(min(extend * sqrt(c00), r1));
    const float r_y = ceil(min(extend * sqrt(c11), r1));

    // Radius clip + image-bounds cull (only zero radii; means2d/conics/depths
    // stay valid, matching torch behavior so the bwd chain works).
    means2d[idx * 2 + 0] = mean_2d.x;
    means2d[idx * 2 + 1] = mean_2d.y;
    depths[idx] = center_z;
    conics[idx * 3 + 0] = i00;
    conics[idx * 3 + 1] = i01;
    conics[idx * 3 + 2] = i11;
    if (has_compensations != 0u) {
        compensations[idx] = sqrt(max(det_orig / det_blur,
                                       MIN_COMPENSATION * MIN_COMPENSATION));
    }

    if (max(r_x, r_y) <= radius_clip) {
        return;
    }
    if (mean_2d.x + r_x <= 0.0f
     || mean_2d.x - r_x >= (float)image_width
     || mean_2d.y + r_y <= 0.0f
     || mean_2d.y - r_y >= (float)image_height) {
        return;
    }
    radii[idx * 2 + 0] = (int)r_x;
    radii[idx * 2 + 1] = (int)r_y;
}
