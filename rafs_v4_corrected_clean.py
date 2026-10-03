"""Corrected RAFS-v4 core: log-field SR with fractional B-spline prefilter.

Optimized version:
- Caches Fourier frequency response H_alpha(fx, fy)
- Reuses forward 2D rfft2 transforms for both scalar fields
- Vectorized alpha-anchor stream blending
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy.ndimage import zoom, sobel
import scipy.fft as sfft

ALPHA_ANCHORS = (1.2, 2.0, 2.8, 3.6)
ALPHA_MIN, ALPHA_MAX = ALPHA_ANCHORS[0], ALPHA_ANCHORS[-1]
EDGE_SCALE = 0.05
M_BOUNDS = (0.25, 15.0)
OMEGA_BOUNDS = (1e-5, float(np.exp(20.0)))

def to_log_coordinates(m, omega, eps=OMEGA_BOUNDS[0]):
    """Convert positive, same-shaped 2-D fields to bounded log coordinates."""
    m, omega = np.asarray(m, dtype=float), np.asarray(omega, dtype=float)
    if m.ndim != 2 or m.shape != omega.shape:
        raise ValueError("m and omega must be same-shaped 2-D arrays")
    if not np.isfinite(m).all() or not np.isfinite(omega).all():
        raise ValueError("m and omega must contain only finite values")
    return (np.log(np.clip(m, *M_BOUNDS)),
            np.log(np.clip(omega, eps, OMEGA_BOUNDS[1])))

def from_log_coordinates(log_m, log_omega):
    """Map logs back to positive physical fields with explicit bounds."""
    lm = np.clip(np.asarray(log_m, float), np.log(M_BOUNDS[0]), np.log(M_BOUNDS[1]))
    lo = np.clip(np.asarray(log_omega, float), np.log(OMEGA_BOUNDS[0]), np.log(OMEGA_BOUNDS[1]))
    return np.exp(lm), np.exp(lo)

def _resize(a, scale, order=1):
    if not isinstance(scale, (int, float, np.integer, np.floating)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a finite positive number")
    a = np.asarray(a, float)
    out = zoom(a, (float(scale), float(scale)), order=order,
               mode="nearest", prefilter=(order > 1))
    expected = (int(round(a.shape[0] * scale)), int(round(a.shape[1] * scale)))
    if out.shape != expected:
        raise RuntimeError(f"unexpected resized shape {out.shape}; expected {expected}")
    return out

@lru_cache(maxsize=16)
def _get_spline_filter(shape: tuple[int, int], alpha: float):
    fy = sfft.fftfreq(shape[0])[:, None]
    fx = sfft.rfftfreq(shape[1])[None, :]
    return np.sinc(fx) ** (alpha + 1.0) * np.sinc(fy) ** (alpha + 1.0)

def edge_fraction(log_m, log_omega, scale=EDGE_SCALE):
    """Bounded joint Sobel gradient magnitude in LR log-coordinate units."""
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("edge scale must be finite and positive")
    def grad(a):
        return sobel(a, axis=1, mode="reflect") / 8.0, sobel(a, axis=0, mode="reflect") / 8.0
    mx, my = grad(log_m)
    ox, oy = grad(log_omega)
    g = np.sqrt(mx * mx + my * my + ox * ox + oy * oy)
    return g / (g + scale)

def alpha_map(edge):
    """Reset mapping: edge=0 -> max alpha; edge=1 -> min alpha."""
    e = np.clip(np.asarray(edge, dtype=float), 0.0, 1.0)
    return ALPHA_MAX - (ALPHA_MAX - ALPHA_MIN) * e

def _blend_weights(target):
    """Convex weights for the two alpha anchors surrounding each target."""
    t = np.clip(np.asarray(target, float), ALPHA_MIN, ALPHA_MAX)
    aa = np.asarray(ALPHA_ANCHORS)
    w = np.zeros((len(aa),) + t.shape, dtype=float)
    for i in range(len(aa) - 1):
        mask = (t >= aa[i]) & (t <= aa[i + 1])
        q = np.clip((t - aa[i]) / (aa[i + 1] - aa[i]), 0, 1)
        w[i][mask], w[i + 1][mask] = (1 - q)[mask], q[mask]
    w[0][t <= aa[0]] = 1.0
    w[-1][t >= aa[-1]] = 1.0
    return w

def spline_prefilter(a, alpha, pad=24):
    """Legacy compatibility prefilter (cached single field)."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("a must be a finite 2-D array")
    if not any(np.isclose(alpha, x) for x in ALPHA_ANCHORS):
        raise ValueError(f"alpha must be one of {ALPHA_ANCHORS}")
    alpha = float(alpha)
    p = min(max(int(pad), 0), max(0, min(a.shape) - 1))
    ap = np.pad(a, ((p, p), (p, p)), mode="reflect") if p else a
    H = _get_spline_filter(ap.shape, alpha)
    z = sfft.irfft2(sfft.rfft2(ap, workers=-1) * H, s=ap.shape, workers=-1)
    return z[p:p + a.shape[0], p:p + a.shape[1]] if p else z

def frac_riemann_sr(m, omega, scale=2, *, adaptive=True, fixed_alpha=2.0, pad=24):
    """Return ``(m_hr, omega_hr, alpha_hr)`` using optimized RAFS-v4."""
    lm, lo = to_log_coordinates(m, omega)
    lm_hr, lo_hr = _resize(lm, scale), _resize(lo, scale)
    
    p = min(max(int(pad), 0), max(0, min(lm_hr.shape) - 1))
    orig_shape = lm_hr.shape

    # Reflect padding
    if p:
        ap_m = np.pad(lm_hr, ((p, p), (p, p)), mode="reflect")
        ap_o = np.pad(lo_hr, ((p, p), (p, p)), mode="reflect")
    else:
        ap_m, ap_o = lm_hr, lo_hr

    pad_shape = ap_m.shape
    # اشتراک‌گذاری تبدیل فوریه ۲ بعدی رفت
    rfft_m = sfft.rfft2(ap_m, workers=-1)
    rfft_o = sfft.rfft2(ap_o, workers=-1)

    if adaptive:
        alpha = alpha_map(_resize(edge_fraction(lm, lo), scale))
        weights = _blend_weights(alpha)
        fm = np.zeros_like(lm_hr)
        fo = np.zeros_like(lo_hr)

        for i, a in enumerate(ALPHA_ANCHORS):
            H = _get_spline_filter(pad_shape, a)
            zm = sfft.irfft2(rfft_m * H, s=pad_shape, workers=-1)
            zo = sfft.irfft2(rfft_o * H, s=pad_shape, workers=-1)
            
            if p:
                zm = zm[p:p + orig_shape[0], p:p + orig_shape[1]]
                zo = zo[p:p + orig_shape[0], p:p + orig_shape[1]]
                
            w = weights[i]
            fm += w * zm
            fo += w * zo
    else:
        if not any(np.isclose(fixed_alpha, a) for a in ALPHA_ANCHORS):
            raise ValueError(f"fixed_alpha must be one of {ALPHA_ANCHORS}")
        alpha = np.full(lm_hr.shape, float(fixed_alpha))
        H = _get_spline_filter(pad_shape, float(fixed_alpha))
        zm = sfft.irfft2(rfft_m * H, s=pad_shape, workers=-1)
        zo = sfft.irfft2(rfft_o * H, s=pad_shape, workers=-1)
        if p:
            fm = zm[p:p + orig_shape[0], p:p + orig_shape[1]]
            fo = zo[p:p + orig_shape[0], p:p + orig_shape[1]]
        else:
            fm, fo = zm, zo

    return (*from_log_coordinates(fm, fo), alpha)

def invariance_tests():
    m = np.full((12, 15), 2.0)
    o = np.full_like(m, 1.25)
    mh, oh, ah = frac_riemann_sr(m, o, 2)
    assert mh.shape == oh.shape == ah.shape == (24, 30)
    assert np.allclose(mh, 2.0, atol=1e-10) and np.allclose(oh, 1.25, atol=1e-10)
    e = np.array([[0.0, 1.0]])
    assert np.allclose(alpha_map(e), [[ALPHA_MAX, ALPHA_MIN]])
    w = _blend_weights(np.linspace(ALPHA_MIN, ALPHA_MAX, 101)[None, :])
    assert np.allclose(w.sum(axis=0), 1.0) and np.all(w >= 0)
    return {
        "constant_preserved": True,
        "alpha_direction_correct": True,
        "weights_convex": True,
        "finite_positive": bool(np.isfinite(mh).all() and np.all(mh > 0) and np.all(oh > 0))
    }

if __name__ == "__main__":
    print(invariance_tests())
