# ─────────────────────────────────────────────────────────────────────────────
# wer_eval/metrics.py  —  WER / CER / substitution / deletion / insertion
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field


@dataclass
class EditStats:
    """Accumulated edit-distance counts for a corpus."""
    hits:         int = 0
    substitutions: int = 0
    deletions:    int = 0
    insertions:   int = 0
    ref_len:      int = 0   # total reference tokens

    def add(self, h: int, s: int, d: int, i: int, r: int) -> None:
        self.hits          += h
        self.substitutions += s
        self.deletions     += d
        self.insertions    += i
        self.ref_len       += r

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return self.errors / self.ref_len if self.ref_len else 0.0

    def summary(self) -> dict:
        return {
            "wer":           round(self.wer * 100, 2),
            "hits":          self.hits,
            "substitutions": self.substitutions,
            "deletions":     self.deletions,
            "insertions":    self.insertions,
            "ref_tokens":    self.ref_len,
        }


def _levenshtein_counts(ref: list, hyp: list) -> tuple[int, int, int, int]:
    """
    Classic dynamic-programming alignment.
    Returns (hits, substitutions, deletions, insertions).
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, hits, subs, dels, ins)
    INF = float("inf")
    dp = [[(INF, 0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = (0, 0, 0, 0, 0)
    for i in range(1, n + 1):
        cost, h, s, d, ins = dp[i - 1][0]
        dp[i][0] = (cost + 1, h, s, d + 1, ins)
    for j in range(1, m + 1):
        cost, h, s, d, ins = dp[0][j - 1]
        dp[0][j] = (cost + 1, h, s, d, ins + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                c, h, s, d, ins = dp[i - 1][j - 1]
                dp[i][j] = (c, h + 1, s, d, ins)
            else:
                # substitution
                cs, hs, ss, ds, inss = dp[i - 1][j - 1]
                opt_s = (cs + 1, hs, ss + 1, ds, inss)
                # deletion
                cd, hd, sd, dd, insd = dp[i - 1][j]
                opt_d = (cd + 1, hd, sd, dd + 1, insd)
                # insertion
                ci, hi, si, di, insi = dp[i][j - 1]
                opt_i = (ci + 1, hi, si, di, insi + 1)
                dp[i][j] = min(opt_s, opt_d, opt_i, key=lambda x: x[0])

    _, h, s, d, ins = dp[n][m]
    return h, s, d, ins


def sample_wer(ref: str, hyp: str) -> dict:
    """
    Compute per-sample WER and edit breakdown.
    Returns a dict with wer, hits, substitutions, deletions, insertions.
    """
    r_tok = ref.split()
    h_tok = hyp.split()

    if not r_tok:
        return {"wer": 0.0, "hits": 0, "substitutions": 0,
                "deletions": 0, "insertions": len(h_tok), "ref_tokens": 0}

    h, s, d, ins = _levenshtein_counts(r_tok, h_tok)
    errors = s + d + ins
    wer = errors / len(r_tok)
    return {
        "wer":           round(wer * 100, 2),
        "hits":          h,
        "substitutions": s,
        "deletions":     d,
        "insertions":    ins,
        "ref_tokens":    len(r_tok),
    }


def sample_cer(ref: str, hyp: str) -> float:
    """Character error rate (no tokenisation)."""
    r = list(ref.replace(" ", ""))
    h = list(hyp.replace(" ", ""))
    if not r:
        return 0.0
    _, s, d, ins = _levenshtein_counts(r, h)
    return round((s + d + ins) / len(r) * 100, 2)
