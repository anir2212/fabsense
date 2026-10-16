"""Render publication-style figures only from saved measured JSON/CSV outputs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def photo(input_path: Path, output: Path):
    report = json.loads(input_path.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
    colors = {"balanced": "#2d706b", "skewed": "#c36c3b"}
    for row_index, rate in enumerate((0.05, 0.08)):
        for mix in ("balanced", "skewed"):
            rows = sorted((row for row in report["scenarios"] if row["mix"] == mix and
                           row["arrival_rate_per_minute"] == rate),
                          key=lambda row: row["max_train_size"])
            sizes = [row["max_train_size"] for row in rows]
            for column, key in enumerate(("throughput_lots_per_day", "p90_cycle_minutes")):
                ax = axes[row_index, column]
                values = [row["summary"][key]["mean"] for row in rows]
                lower = [row["summary"][key]["ci95"][0] for row in rows]
                upper = [row["summary"][key]["ci95"][1] for row in rows]
                ax.plot(sizes, values, marker="o", color=colors[mix], label=mix)
                ax.fill_between(sizes, lower, upper, color=colors[mix], alpha=0.12)
        axes[row_index, 0].set_ylabel(f"{rate} lots/min\nThroughput (lots/day)")
        axes[row_index, 1].set_ylabel(f"{rate} lots/min\np90 cycle (minutes)")
        for ax in axes[row_index]:
            ax.set_xscale("log", base=2)
            ax.set_xticks((1, 2, 4, 8, 12, 16, 24), labels=("1", "2", "4", "8", "12", "16", "24"))
            ax.grid(alpha=0.2)
    for ax in axes[1]:
        ax.set_xlabel("Maximum train size (lots)")
    axes[0, 0].legend(title="Recipe mix")
    fig.suptitle("Photo-bay train size: throughput and cycle-time tradeoff (20 replications)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def agreement(csv_path: Path, report_path: Path, output: Path):
    rows = list(csv.DictReader(csv_path.open()))
    report = json.loads(report_path.read_text())["workstations"]["agreement"]
    by_part: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        by_part[row["workstation"], int(row["part"])].append(float(row["measured"]))
    parts = sorted({part for _, part in by_part})
    old = np.array([np.mean(by_part["old", part]) for part in parts])
    new = np.array([np.mean(by_part["new", part]) for part in parts])
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    ax.scatter((old + new) / 2, new - old, s=52, color="#2d706b")
    for part, x, y in zip(parts, (old + new) / 2, new - old, strict=True):
        ax.annotate(str(part), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    for key, style in (("new_minus_old_mean", "-"), ("bland_altman_lower", "--"),
                       ("bland_altman_upper", "--")):
        ax.axhline(report[key], linestyle=style, color="#c36c3b")
    ax.set(xlabel="Old/new mean by part", ylabel="New − old measurement",
           title="Workstation agreement (10 parts)")
    ax.grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    photo(Path("runs/photo_grid.json"), Path("figures/photo_tradeoff.png"))
    agreement(Path("runs/msa/measurements.csv"), Path("runs/msa/results.json"),
              Path("figures/msa_agreement.png"))
