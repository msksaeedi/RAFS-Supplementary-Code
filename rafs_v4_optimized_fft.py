"""Corrected RAFS-v4 core: log-field SR with fractional B-spline prefilter.

This is an experimental scalar-field interpolation/smoothing heuristic.  The
separable centered cardinal B-spline response used by the implementation is

    H_alpha(fx, fy) = sinc(fx)**(alpha + 1) * sinc(fy)**(alpha + 1)

where numpy.sinc(x)=sin(pi*x)/(pi*x).  Reflect padding is used before the FFT.
The alpha map is reset explicitly: flat pixels use ALPHA_MAX (more smoothing)
and strong edges use ALPHA_MIN (less smoothing).  No detail-recovery claim is
made.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import zoom, sobel
from scipy import fft as _fft

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


def spline_prefilter(a, alpha, pad=24):
    """Apply the centered fractional cardinal B-spline Fourier response."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("a must be a finite 2-D array")
    if not any(np.isclose(alpha, x) for x in ALPHA_ANCHORS):
        raise ValueError(f"alpha must be one of {ALPHA_ANCHORS}")
    alpha = float(alpha)
    p = min(max(int(pad), 0), max(0, min(a.shape) - 1))
    ap = np.pad(a, ((p, p), (p, p)), mode="reflect") if p else a
    fy = np.fft.fftfreq(ap.shape[0])[:, None]
    fx = np.fft.rfftfreq(ap.shape[1])[None, :]
    H = np.sinc(fx) ** (alpha + 1.0) * np.sinc(fy) ** (alpha + 1.0)
    z = np.fft.irfft2(np.fft.rfft2(ap) * H, s=ap.shape)
    return z[p:p + a.shape[0], p:p + a.shape[1]] if p else z


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


def _prefilter_pair_batched_stream(log_m, log_omega, alpha, pad=24, workers=-1):
    """Filter two same-shaped fields with shared forward FFT, streaming alpha blend.

    Only two neighboring alpha responses are blended at each pixel.  This avoids
    materializing a (4,H,W) weight tensor and reuses the two forward transforms.
    """
    x = np.stack((log_m, log_omega), axis=0)
    p = min(max(int(pad), 0), max(0, min(x.shape[-2:]) - 1))
    xp = np.pad(x, ((0, 0), (p, p), (p, p)), mode="reflect") if p else x
    fy = _fft.fftfreq(xp.shape[-2])[:, None]
    fx = _fft.rfftfreq(xp.shape[-1])[None, :]
    spectrum = _fft.rfft2(xp, axes=(-2, -1), workers=workers)
    target = np.clip(alpha, ALPHA_MIN, ALPHA_MAX)
    anchors = np.asarray(ALPHA_ANCHORS, dtype=float)
    idx = np.clip(np.searchsorted(anchors, target, side="right") - 1,
                  0, len(anchors) - 2)
    q = (target - anchors[idx]) / (anchors[idx + 1] - anchors[idx])
    out_m = np.empty_like(log_m); out_o = np.empty_like(log_omega)
    previous = None
    for i, a in enumerate(anchors):
        response = np.sinc(fx) ** (a + 1.0) * np.sinc(fy) ** (a + 1.0)
        z = _fft.irfft2(spectrum * response[None, :, :], s=xp.shape[-2:],
                        axes=(-2, -1), workers=workers)
        if p:
            z = z[:, p:p + log_m.shape[0], p:p + log_m.shape[1]]
        if previous is not None:
            mask = idx == i - 1
            qi = q[mask]
            out_m[mask] = previous[0][mask] * (1.0 - qi) + z[0][mask] * qi
            out_o[mask] = previous[1][mask] * (1.0 - qi) + z[1][mask] * qi
        previous = z
    return out_m, out_o


def _prefilter_pair_batched(log_m, log_omega, alpha, pad=24, workers=-1):
    """Apply one shared forward FFT to a pair of fields at fixed alpha."""
    x = np.stack((log_m, log_omega), axis=0)
    p = min(max(int(pad), 0), max(0, min(x.shape[-2:]) - 1))
    xp = np.pad(x, ((0, 0), (p, p), (p, p)), mode="reflect") if p else x
    fy = _fft.fftfreq(xp.shape[-2])[:, None]
    fx = _fft.rfftfreq(xp.shape[-1])[None, :]
    response = np.sinc(fx) ** (alpha + 1.0) * np.sinc(fy) ** (alpha + 1.0)
    spectrum = _fft.rfft2(xp, axes=(-2, -1), workers=workers)
    z = _fft.irfft2(spectrum * response[None, :, :], s=xp.shape[-2:],
                    axes=(-2, -1), workers=workers)
    if p:
        z = z[:, p:p + log_m.shape[0], p:p + log_m.shape[1]]
    return z[0], z[1]


def frac_riemann_sr(m, omega, scale=2, *, adaptive=True, fixed_alpha=2.0):
    """Return ``(m_hr, omega_hr, alpha_hr)`` using optimized RAFS-v4.

    Linear zoom remains scipy.ndimage.zoom (independent arrays benchmark faster
    than stacked/custom alternatives). FFT filtering shares forward transforms.
    """
    lm, lo = to_log_coordinates(m, omega)
    lm_hr, lo_hr = _resize(lm, scale), _resize(lo, scale)
    if adaptive:
        alpha = alpha_map(_resize(edge_fraction(lm, lo), scale))
        fm, fo = _prefilter_pair_batched_stream(lm_hr, lo_hr, alpha)
    else:
        if not any(np.isclose(fixed_alpha, a) for a in ALPHA_ANCHORS):
            raise ValueError(f"fixed_alpha must be one of {ALPHA_ANCHORS}")
        alpha = np.full(lm_hr.shape, float(fixed_alpha))
        fm, fo = _prefilter_pair_batched(lm_hr, lo_hr, float(fixed_alpha))
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
