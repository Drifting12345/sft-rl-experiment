#!/usr/bin/env python3
"""Compact paper figures from the same CSV / JSONL / JSON as index.html."""
from __future__ import annotations

import argparse
import colorsys
import csv
import json
import math
import re
from pathlib import Path


def integer(value, field, where):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{where}: invalid {field}")
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{where}: invalid {field}") from exc
    if not math.isfinite(number) or number < 0 or not number.is_integer() or number > 2**53 - 1:
        raise ValueError(f"{where}: {field} must be a nonnegative safe integer")
    return int(number)


def read_records(paths):
    records, expected = [], None
    for path in paths:
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() in (".jsonl", ".ndjson"):
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
        elif path.suffix.lower() == ".json":
            data = json.loads(text)
            if isinstance(data, dict):
                if "expected_samples" in data:
                    count = integer(data["expected_samples"], "expected_samples", str(path))
                    if expected is not None and expected != count:
                        raise ValueError("Conflicting expected_samples values")
                    expected = count
                data = data.get("records")
        else:
            data = list(csv.DictReader(text.splitlines(keepends=True)))
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a record list")
        records.extend(data)
    return records, expected


def initial_bin(correct, total):
    return 0 if correct == 0 else (correct * 20 + total - 1) // total


def natural_key(text):
    return tuple((0, int(x)) if x.isdigit() else (1, x.casefold()) for x in re.split(r"(\d+)", text))


def prepare(records, expected=None):
    samples, epochs = {}, set()
    for index, record in enumerate(records, 1):
        where = f"record {index}"
        if not isinstance(record, dict):
            raise ValueError(f"{where}: expected an object")
        identifier = record.get("sample_id")
        if isinstance(identifier, bool) or not isinstance(identifier, (str, int, float)):
            raise ValueError(f"{where}: invalid sample_id")
        sid = str(identifier)
        if not sid.strip() or len(sid) > 500:
            raise ValueError(f"{where}: invalid sample_id")
        epoch, correct, total, update = [integer(record.get(k), k, where) for k in
                                         ("epoch", "eval_correct", "eval_total", "update_count")]
        if total == 0 or correct > total:
            raise ValueError(f"{where}: expected 0 <= eval_correct <= eval_total and eval_total > 0")
        if epoch == 0 and update != 0:
            raise ValueError(f"{where}: baseline update_count must be zero")
        sample = samples.setdefault(sid, {})
        if epoch in sample:
            raise ValueError(f"Duplicate sample/epoch pair: {sid}, {epoch}")
        sample[epoch] = (correct, total, update)
        epochs.add(epoch)
    if not samples:
        raise ValueError("No records")
    if expected is not None and len(samples) != expected:
        raise ValueError(f"Expected {expected} samples, found {len(samples)}")
    for sid, values in samples.items():
        if 0 not in values:
            raise ValueError(f"Missing epoch=0 baseline for {sid}")
    def order(sid):
        c, n, _ = samples[sid][0]
        return initial_bin(c, n), c / n, natural_key(sid)
    identifiers = sorted(samples, key=order)
    return samples, identifiers, sorted(epochs)


def demo_records(count=1000, epochs=12, seed=20261105):
    import numpy as np
    rng = np.random.default_rng(seed)
    result = []
    for i in range(count):
        p0 = 0.0 if i % 5 == 0 else float(rng.uniform(.001, .999))
        mode, delay, previous = i % 6, int(rng.integers(2, max(3, epochs // 2 + 1))), p0
        for epoch in range(epochs + 1):
            t = epoch / epochs
            if mode == 0:
                p = p0 + (1 - p0) * max(0, (epoch - delay) / max(1, epochs - delay)) * .94
            elif mode == 1:
                p = p0 + (1 - p0) * t * .9
            elif mode == 2:
                p = p0 * (1 - t * .98)
            elif mode == 3:
                p = p0
            elif mode == 4:
                p = p0 + (1 - p0) * math.sin(t * math.pi) * .5
            else:
                p = max(0, min(1, p0 + (float(rng.uniform()) - .5) * .05)) if epoch else p0
            correct = int(rng.binomial(256, min(1, max(0, p))))
            chance = max(0, min(1, 1 - previous**8 - (1 - previous)**8))
            update = int(epoch > 0 and rng.uniform() < chance)
            result.append(dict(sample_id=f"sample_{i + 1:04d}", epoch=epoch,
                               eval_correct=correct, eval_total=256, update_count=update))
            previous = p
    return result


def status_palettes():
    """Matched lightness across hues: hue encodes update, lightness encodes p."""
    return [[(*colorsys.hls_to_rgb(hue, .89 - .53 * b / 20, .68), 1.0)
             for b in range(21)] for hue in (29 / 360, 209 / 360)]


def plot(records, output, *, expected=None, layout="bivariate", width=7.2, height=4.7,
         dpi=600, formats=("png", "svg"), title=None, bin_filter=None, mark_fraction=.13):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    samples, ids, epochs = prepare(records, expected)
    if bin_filter is not None:
        ids = [sid for sid in ids if initial_bin(*samples[sid][0][:2]) == bin_filter]
    if not ids:
        raise ValueError("No samples match --initial-bin")
    N, T = len(ids), len(epochs)
    if N * T > 5_000_000:
        raise ValueError("More than 5 million cells; split or filter the data")
    rates = np.full((N, T), np.nan)
    bins = np.full((N, T), np.nan)
    updates = np.full((N, T), np.nan)
    groups = []
    for i, sid in enumerate(ids):
        c0, n0, _ = samples[sid][0]
        groups.append(initial_bin(c0, n0))
        for j, epoch in enumerate(epochs):
            if epoch in samples[sid]:
                c, n, u = samples[sid][epoch]
                rates[i, j], bins[i, j] = c / n, initial_bin(c, n)
                if epoch > 0:
                    updates[i, j] = int(u > 0)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.labelsize": 8, "axes.titlesize": 8.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 6.5,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.linewidth": .4, "savefig.facecolor": "white"})
    # Identical 5% bins to the browser, with zero observed successes separate.
    colors = ["#f0f3f7"] + [plt.cm.Blues(v) for v in np.linspace(.08, .87, 20)]
    cmap = ListedColormap(colors)
    cmap.set_bad("#d8d8dc")
    norm = BoundaryNorm(np.arange(-.5, 21.5), cmap.N)
    fig = plt.figure(figsize=(width, height), facecolor="white")
    grid = fig.add_gridspec(2, 2 if layout == "split" else 1,
                           width_ratios=[1, 1] if layout == "split" else [1],
                           height_ratios=[1, .12 if layout == "bivariate" else .035],
                           hspace=.26, wspace=.13,
                           left=.14, right=.985, bottom=.17, top=.90 if title else .94)
    ax = fig.add_subplot(grid[0, 0])
    if layout == "bivariate":
        palettes = np.asarray(status_palettes())
        rgba = np.full((N, T, 4), [0.84, 0.84, 0.86, 1.0])
        valid = np.isfinite(bins)
        training = valid & np.isfinite(updates)
        rr, cc = np.where(training)
        rgba[rr, cc] = palettes[updates[rr, cc].astype(int), bins[rr, cc].astype(int)]
        for j, epoch in enumerate(epochs):
            if epoch == 0:
                rr = np.where(valid[:, j])[0]
                rgba[rr, j] = plt.cm.Greys(.10 + .67 * bins[rr, j] / 20)
        im = ax.imshow(rgba, aspect="auto", interpolation="nearest", origin="upper")
        if np.any(~valid):
            for i, j in zip(*np.where(~valid)):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                          facecolor="none", edgecolor="#72727b",
                                          linewidth=0, hatch="////"))
        ax.axvline(.5, color="#ffffff", linewidth=.6)
    else:
        im = ax.imshow(bins, aspect="auto", interpolation="nearest", origin="upper", cmap=cmap, norm=norm)
    axes = [ax]
    # Show only initial-bin labels, never individual sample numbers.
    transitions = [0] + [i for i in range(1, N) if groups[i] != groups[i - 1]] + [N]
    ypositions, ylabels = [], []
    heat_height_pt = height * 72 * (.90 - .17) / 1.30
    last_label = -1e9
    for start, stop in zip(transitions[:-1], transitions[1:]):
        mid = (start + stop - 1) / 2
        if (mid - last_label) / N * heat_height_pt >= 8.5:
            group = groups[start]
            ypositions.append(mid)
            ylabels.append("0%" if group == 0 else f"{(group - 1) * 5}-{group * 5}%")
            last_label = mid
    ax.set_yticks(ypositions, ylabels)
    ax.set_ylabel("Initial success-rate group\n(fixed sample order)", labelpad=7)
    ax.set_title("(a) Single-sample success rate" if layout == "split" else
                 "Hue: reward-update status; shade: single-sample success rate" if layout == "bivariate" else
                 "Success rate with reward-update signals", loc="left", pad=7)

    if layout == "split":
        other = fig.add_subplot(grid[0, 1], sharey=ax)
        update_colors = ListedColormap(["#fafafa", "#303c50"])
        update_colors.set_bad("#d8d8dc")
        other.imshow(updates, aspect="auto", interpolation="nearest", origin="upper", cmap=update_colors,
                     norm=BoundaryNorm([-.5, .5, 1.5], 2))
        other.tick_params(axis="y", left=False, labelleft=False)
        other.set_title("(b) Effective reward signal", loc="left", pad=7)
        axes.append(other)
    elif layout == "overlay":
        rows, cols = np.where(updates == 1)
        segments = [[(j - mark_fraction / 2, i), (j + mark_fraction / 2, i)] for i, j in zip(rows, cols)]
        row_height_pt = max(.03, heat_height_pt / N)
        ax.add_collection(LineCollection(segments, colors="#15243b", linewidths=row_height_pt * .62,
                                         antialiased=False, capstyle="butt"))

    tick_indices = sorted(set(np.linspace(0, T - 1, min(T, 7), dtype=int)))
    for panel in axes:
        panel.set_xticks(tick_indices, ["0" if epochs[j] == 0 else str(epochs[j]) for j in tick_indices])
        panel.set_xlabel("RL epoch (0 = baseline)", labelpad=5)
        panel.tick_params(length=2, width=.4)
        for boundary in transitions[1:-1]:
            panel.axhline(boundary - .5, color="#ffffff", linewidth=.32, alpha=.9)
        panel.set_ylim(N - .5, -.5)
        panel.set_xlim(-.5, T - .5)
        for spine in panel.spines.values():
            spine.set_edgecolor("#929aaa")
    cax = fig.add_subplot(grid[1, 0])
    if layout == "bivariate":
        # One two-dimensional legend for the single heatmap, not another data panel.
        cax.imshow(np.asarray(status_palettes())[::-1], aspect="auto", interpolation="nearest")
        cax.set_yticks([0, 1], ["Update", "No update"])
        cax.set_xticks([0, 5, 10, 15, 20], ["0%", "25%", "50%", "75%", "100%"])
        cax.set_xlabel("Single-sample success rate (5% bins)", fontsize=7, labelpad=3)
        cax.tick_params(length=0, pad=4)
        for spine in cax.spines.values():
            spine.set_visible(False)
    else:
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 5, 10, 15, 20])
        cb.ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        cb.set_label("Single-sample success rate (5% bins)", labelpad=3, fontsize=7)
        cb.outline.set_linewidth(.35)
    if layout == "split":
        lax = fig.add_subplot(grid[1, 1])
        lax.axis("off")
        lax.legend(handles=[Patch(facecolor="#303c50", label="Signal > 0"),
                            Patch(facecolor="#fafafa", edgecolor="#aaa", linewidth=.4, label="Signal = 0")],
                   loc="center", frameon=False, ncol=2, fontsize=6.8,
                   handlelength=1.0, columnspacing=1.1)
    elif layout == "overlay":
        fig.legend(handles=[Line2D([0], [0], color="#15243b", linewidth=1.3, label="Effective reward signal > 0")],
                   loc="lower right", bbox_to_anchor=(.99, .015), frameon=False, fontsize=7)
    baseline_label = ("Baseline: grayscale; update status is not applicable" if layout == "bivariate" else
                      "Gray in panel (b): baseline / missing" if layout == "split" else
                      "Gray: zero observed successes; missing cells use a separate gray")
    if layout == "bivariate" and np.any(~np.isfinite(rates)):
        baseline_label += "; hatching: missing"
    fig.text(.14, .032, f"n = {N:,} samples; each row is one sample.  {baseline_label}.", fontsize=6.3, color="#555d69")
    if title:
        fig.suptitle(title, x=.14, y=.983, ha="left", fontsize=10, fontweight="medium")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        path = output.with_suffix("." + extension)
        fig.savefig(path, dpi=dpi, metadata={"Creator": "RL sample trajectories"})
        print(path.resolve())
    plt.close(fig)
    return dict(samples=N, epochs=len(epochs) - 1, missing=int(np.isnan(rates).sum()),
                update_cells=int(np.nansum(updates)), width=width, height=height, dpi=dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="CSV / JSONL / JSON; multiple files are merged")
    parser.add_argument("--output", type=Path, default=Path("compact-trajectories"), help="Output prefix")
    parser.add_argument("--layout", choices=("bivariate", "split", "overlay"), default="bivariate")
    parser.add_argument("--width", type=float, default=7.2, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=4.7, help="Figure height in inches")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--formats", nargs="+", choices=("png", "svg", "pdf"), default=["png", "svg"])
    parser.add_argument("--title", help="Optional title; omitted by default for paper figures")
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--initial-bin", type=int, choices=range(21), help="0=zero, 1=(0,5%%], ..., 20=(95,100%%]")
    parser.add_argument("--demo-samples", type=int, help="Generate explicitly simulated data instead of reading files")
    parser.add_argument("--demo-epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    if args.width < 3 or args.height < 2.5 or args.dpi < 72 or args.width * args.height * args.dpi**2 > 150_000_000:
        parser.error("Use width >= 3, height >= 2.5, DPI >= 72, and at most 150 million pixels")
    if args.demo_samples is not None:
        if args.inputs:
            parser.error("Use either input files or --demo-samples")
        if not 1 <= args.demo_samples <= 50000 or not 1 <= args.demo_epochs <= 200:
            parser.error("Demo: 1-50000 samples, 1-200 epochs")
        records, expected = demo_records(args.demo_samples, args.demo_epochs, args.seed), args.demo_samples
        title = args.title or f"SIMULATED DATA - {args.demo_samples:,} samples, {args.demo_epochs} RL epochs"
    else:
        if not args.inputs:
            parser.error("Provide experiment files, or use --demo-samples for a labeled demo")
        records, expected = read_records(args.inputs)
        title = args.title
    if args.expected_samples is not None:
        if expected is not None and expected != args.expected_samples:
            parser.error("--expected-samples conflicts with file metadata")
        expected = args.expected_samples
    try:
        summary = plot(records, args.output, expected=expected, layout=args.layout, width=args.width,
                       height=args.height, dpi=args.dpi, formats=args.formats, title=title,
                       bin_filter=args.initial_bin)
        print(json.dumps(summary, ensure_ascii=False))
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
