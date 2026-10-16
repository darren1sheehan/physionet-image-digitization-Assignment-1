"""Decoding (probability map -> signal) and lead-consistency post-processing variants, for offline evaluation.

Decoders (applied per column of each row's probability map):
    mean       probability-weighted mean y over the whole column (the author's pixel_to_series_exp)
    parabolic  argmax refined by a parabola through the peak and its two neighbours (3rd place)
    window     probability-weighted mean y within +-WINDOW px of the argmax (3rd place's "top-K" idea)

Post-processing (P1 variants) on a dict of 12 leads plus the row-2 lead II segment:
    p1   as submitted in S1+P1: fuse the two lead II predictions, then enforce II = I + III (weights II 2, I 1,
         III 1) and aVR + aVL + aVF = 0
    p1b  offset-safe: match the row-2 II baseline to the rhythm strip before fusing, and correct only the
         time-varying part of each mismatch, so no step is created in lead II
    p1t  1st place rules: fuse, then apply each identity only if its mean absolute mismatch is below 0.01 mV
"""
import numpy as np
import torch

WINDOW = 3


@torch.no_grad()
def decode(prob: torch.Tensor, method: str) -> torch.Tensor:
    """prob: (4, H, W) probabilities -> (4, W) row positions in pixels (float)."""
    p = prob.float()
    H = p.shape[1]
    y = torch.arange(H, device=p.device, dtype=torch.float32).reshape(1, H, 1)
    if method == "mean":
        return (p * y).sum(1) / (p.sum(1) + 1e-8)
    i = p.argmax(1)  # (4, W)
    if method == "parabolic":
        il, ir = (i - 1).clamp(0, H - 1), (i + 1).clamp(0, H - 1)
        yc = p.gather(1, i[:, None]).squeeze(1)
        yl = p.gather(1, il[:, None]).squeeze(1)
        yr = p.gather(1, ir[:, None]).squeeze(1)
        denom = yl - 2 * yc + yr
        delta = torch.where(denom < -1e-8, (yl - yr) / (2 * denom), torch.zeros_like(denom)).clamp(-1, 1)
        return i.float() + delta
    if method == "window":
        offsets = torch.arange(-WINDOW, WINDOW + 1, device=p.device).reshape(1, -1, 1)
        idx = (i[:, None, :] + offsets).clamp(0, H - 1)  # (4, 2W+1, W)
        w = p.gather(1, idx)
        return (w * idx.float()).sum(1) / (w.sum(1) + 1e-8)
    raise ValueError(method)


def _split(s):
    return {k: np.asarray(v, dtype=np.float64).copy() for k, v in s.items()}


def p1(leads, ii_short):
    s = _split(leads)
    ii_short = np.asarray(ii_short, dtype=np.float64)
    n = min(len(ii_short), len(s['I']), len(s['III']))
    m = min(len(s['aVR']), len(s['aVL']), len(s['aVF']))
    s['II'][:n] = 0.5 * (s['II'][:n] + ii_short[:n])
    r = s['II'][:n] - s['I'][:n] - s['III'][:n]
    s['II'][:n] -= 0.2 * r
    s['I'][:n] += 0.4 * r
    s['III'][:n] += 0.4 * r
    t = (s['aVR'][:m] + s['aVL'][:m] + s['aVF'][:m]) / 3
    for k in ('aVR', 'aVL', 'aVF'):
        s[k][:m] -= t
    return s


def p1b(leads, ii_short):
    s = _split(leads)
    ii_short = np.asarray(ii_short, dtype=np.float64)
    n = min(len(ii_short), len(s['I']), len(s['III']))
    m = min(len(s['aVR']), len(s['aVL']), len(s['aVF']))
    head = s['II'][:n]
    s['II'][:n] = 0.5 * (head + (ii_short[:n] - ii_short[:n].mean() + head.mean()))
    r = s['II'][:n] - s['I'][:n] - s['III'][:n]
    r = r - r.mean()
    s['II'][:n] -= 0.2 * r
    s['I'][:n] += 0.4 * r
    s['III'][:n] += 0.4 * r
    t = (s['aVR'][:m] + s['aVL'][:m] + s['aVF'][:m]) / 3
    t = t - t.mean()
    for k in ('aVR', 'aVL', 'aVF'):
        s[k][:m] -= t
    return s


def p1t(leads, ii_short, threshold=0.01):
    s = _split(leads)
    ii_short = np.asarray(ii_short, dtype=np.float64)
    n = min(len(ii_short), len(s['I']), len(s['III']))
    m = min(len(s['aVR']), len(s['aVL']), len(s['aVF']))
    s['II'][:n] = 0.5 * (s['II'][:n] + ii_short[:n])
    r = s['II'][:n] - s['I'][:n] - s['III'][:n]
    if np.abs(r).mean() < threshold:
        s['II'][:n] -= 0.2 * r
        s['I'][:n] += 0.4 * r
        s['III'][:n] += 0.4 * r
    t = s['aVR'][:m] + s['aVL'][:m] + s['aVF'][:m]
    if np.abs(t).mean() < threshold:
        for k in ('aVR', 'aVL', 'aVF'):
            s[k][:m] -= t / 3
    return s


POSTPROCESS = {"none": lambda leads, ii_short: _split(leads), "p1": p1, "p1b": p1b, "p1t": p1t}
