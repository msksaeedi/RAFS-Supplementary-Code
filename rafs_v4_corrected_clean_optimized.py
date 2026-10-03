"""Corrected RAFS-v4 core with shared-spectrum FFT optimization.

Experimental scalar-field interpolation/smoothing heuristic. The separable
centered cardinal B-spline response is
    H_alpha(fx, fy) = sinc(fx)**(alpha + 1) * sinc(fy)**(alpha + 1)
Reflect padding is used before the FFT. Flat pixels map to ALPHA_MAX and strong
edges to ALPHA_MIN; this is not a detail-recovery method.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy.ndimage import zoom, sobel

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
def _frequency_response(shape, alpha):
    """Cached real-FFT transfer function (returned array is treated read-only)."""
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.rfftfreq(shape[1])[None, :]
    return np.sinc(fx) ** (alpha + 1.0) * np.sinc(fy) ** (alpha + 1.0)


def _validate_alpha(alpha, name="alpha"):
    if not any(np.isclose(alpha, x) for x in ALPHA_ANCHORS):
        raise ValueError(f"{name} must be one of {ALPHA_ANCHORS}")
    return float(alpha)


def _pad_for_filter(a, pad):
    p = min(max(int(pad), 0), max(0, min(a.shape[-2:]) - 1))
    return (np.pad(a, ((0, 0), (p, p), (p, p)), mode="reflect"), p) if p else (a, 0)


def spline_prefilter(a, alpha, pad=24):
    """Apply the centered fractional cardinal B-spline Fourier response."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("a must be a finite 2-D array")
    alpha = _validate_alpha(alpha)
    ap, p = _pad_for_filter(a[None, ...], pad)
    h = _frequency_response(ap.shape[-2:], alpha)
    z = np.fft.irfft2(np.fft.rfft2(ap, axes=(-2, -1)) * h[None, ...],
                      s=ap.shape[-2:], axes=(-2, -1))[0]
    return z[p:p + a.shape[0], p:p + a.shape[1]] if p else z


def _spline_filter_pair(lm, lo, anchors=ALPHA_ANCHORS, pad=24):
    """Filter two fields using one shared forward FFT for all alpha anchors."""
    pair = np.stack((lm, lo))
    ap, p = _pad_for_filter(pair, pad)
    spectrum = np.fft.rfft2(ap, axes=(-2, -1))
    shape = ap.shape[-2:]
    outputs = []
    for alpha in anchors:
        h = _frequency_response(shape, float(alpha))
        z = np.fft.irfft2(spectrum * h[None, ...], s=shape, axes=(-2, -1))
        if p:
            z = z[:, p:p + lm.shape[0], p:p + lm.shape[1]]
        outputs.append(z)
    return outputs


def edge_fraction(log_m, log_omega, scale=EDGE_SCALE):
    """Bounded joint Sobel gradient magnitude in LR log-coordinate units."""
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("edge scale must be finite and positive")
    def grad(a):
        return sobel(a, axis=1, mode="reflect") / 8.0, sobel(a, axis=0, mode="reflect") / 8.0
    mx, my = grad(log_m); ox, oy = grad(log_omega)
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


def frac_riemann_sr(m, omega, scale=2, *, adaptive=True, fixed_alpha=2.0):
    """Return (m_hr, omega_hr, alpha_hr) using corrected RAFS-v4."""
    lm, lo = to_log_coordinates(m, omega)
    lm_hr, lo_hr = _resize(lm, scale), _resize(lo, scale)
    if adaptive:
        alpha = alpha_map(_resize(edge_fraction(lm, lo), scale))
        weights = _blend_weights(alpha)
        # Sharing the forward FFT is numerically equivalent to the original
        # per-field/per-anchor FFTs and avoids repeating eight transforms.
        filtered = _spline_filter_pair(lm_hr, lo_hr)
        fm = np.zeros_like(lm_hr); fo = np.zeros_like(lo_hr)
        for i, (zm, zo) in enumerate(filtered):
            fm += weights[i] * zm
            fo += weights[i] * zo
    else:
        fixed_alpha = _validate_alpha(fixed_alpha, "fixed_alpha")
        alpha = np.full(lm_hr.shape, fixed_alpha)
        fm, fo = _spline_filter_pair(lm_hr, lo_hr, (fixed_alpha,))[0]
    return (*from_log_coordinates(fm, fo), alpha)


def invariance_tests():
    m = np.full((12, 15), 2.0); o = np.full_like(m, 1.25)
    mh, oh, ah = frac_riemann_sr(m, o, 2)
    assert mh.shape == oh.shape == ah.shape == (24, 30)
    assert np.allclose(mh, 2.0, atol=1e-10) and np.allclose(oh, 1.25, atol=1e-10)
    e = np.array([[0.0, 1.0]])
    assert np.allclose(alpha_map(e), [[ALPHA_MAX, ALPHA_MIN]])
    w = _blend_weights(np.linspace(ALPHA_MIN, ALPHA_MAX, 101)[None, :])
    assert np.allclose(w.sum(axis=0), 1.0) and np.all(w >= 0)
    return {"constant_preserved": True, "alpha_direction_correct": True,
            "weights_convex": True, "finite_positive": bool(np.isfinite(mh).all() and np.all(mh > 0) and np.all(oh > 0))}


if __name__ == "__main__":
    print(invariance_tests())
