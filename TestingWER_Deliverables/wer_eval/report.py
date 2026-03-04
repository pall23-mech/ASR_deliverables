# ─────────────────────────────────────────────────────────────────────────────
# wer_eval/report.py  —  JSON / CSV persistence and terminal summary
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


# ── Persistence ───────────────────────────────────────────────────────────────

def save_json(results: list[dict], summary: dict, path: str) -> None:
    out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary":      summary,
        "samples":      results,
    }
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"  → Saved JSON: {path}")


def save_csv(results: list[dict], path: str) -> None:
    if not results:
        return
    keys = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"  → Saved CSV:  {path}")


# ── Plot ─────────────────────────────────────────────────────────────────────

def save_plot(results: list[dict], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  ⚠  matplotlib not installed — skipping plot")
        return

    durations = [r.get("duration_s", 0) for r in results]
    wers      = [r.get("wer", 0) for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(durations, wers, alpha=0.5, s=20, color="#4a90d9", edgecolors="none")

    # Trend line
    if len(durations) > 2:
        z = np.polyfit(durations, wers, 1)
        x_line = np.linspace(min(durations), max(durations), 200)
        ax.plot(x_line, np.poly1d(z)(x_line), color="#d94a4a", linewidth=1.5,
                label="linear trend")
        ax.legend()

    # Median reference line
    med = float(np.median(wers))
    ax.axhline(med, color="#888888", linestyle="--", linewidth=1,
               label=f"median WER = {med:.1f}%")
    ax.legend()

    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("WER (%)")
    ax.set_title("WER vs. Recording Duration")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → Saved plot: {path}")


# ── Terminal summary ──────────────────────────────────────────────────────────

def print_summary(summary: dict, results: list[dict], n_examples: int = 10) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  WER EVALUATION SUMMARY")
    print(sep)
    print(f"  Backend   : {summary.get('backend', '?')}")
    print(f"  Dataset   : {summary.get('dataset', '?')}  [{summary.get('split', '?')}]")
    print(f"  Samples   : {summary.get('n_samples', '?')}")
    print(f"  Skipped   : {summary.get('n_skipped', 0)}  (empty reference or hypothesis)")
    print(sep)
    print(f"  WER       : {summary.get('wer', '?'):.2f}%")
    print(f"  CER       : {summary.get('cer', '?'):.2f}%")
    print(f"  Sub / Del / Ins : "
          f"{summary.get('substitutions','?')} / "
          f"{summary.get('deletions','?')} / "
          f"{summary.get('insertions','?')}")
    print(f"  Ref tokens: {summary.get('ref_tokens', '?')}")
    print(sep)

    # Worst examples
    ranked = sorted(results, key=lambda r: r.get("wer", 0), reverse=True)
    print(f"\n  Top-{n_examples} worst samples:")
    for i, r in enumerate(ranked[:n_examples], 1):
        print(f"\n  [{i}] idx={r['idx']}  WER={r['wer']:.1f}%  dur={r.get('duration_s', '?'):.1f}s")
        print(f"      REF: {r['reference'][:120]}")
        print(f"      HYP: {r['hypothesis'][:120]}")

    print(f"\n{sep}\n")
