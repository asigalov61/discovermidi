#! /usr/bin/python3

r'''###############################################################################
###################################################################################
#
#	Plots and Reports Python Module
#	Version 1.0
#
#	Project Los Angeles
#
#	Tegridy Code 2025
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
###################################################################################
###################################################################################
#
#   Copyright 2025 Project Los Angeles / Tegridy Code
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
###################################################################################
###################################################################################
#
#   Critical packages
#
#   1pip install scikit-learn
#   !pip install scipy
#   !pip install matplotlib
#   !pip install tqdm
#   !pip install numpy==1.24.4
#
###################################################################################
'''

###################################################################################

import os
import json
import tqdm
import random
import math
import statistics

from collections import Counter

from typing import List, Dict, Any, Optional

from collections import Counter, defaultdict

import numpy as np

from sklearn.preprocessing import QuantileTransformer

from scipy.signal import savgol_filter

import matplotlib.pyplot as plt

###################################################################################

def read_jsonl(file_name='./Discover-MIDI-Dataset/DATA/Averages/all_midis_averages_data', 
               file_ext='.jsonl',
               max_lines=-1,
               verbose=True
              ):
    
    """Read a JSON Lines (jsonl) file and return its records as a list of Python objects.
    
    This helper opens a `.jsonl` file (optionally appending a file extension if none
    is provided), iterates through each line, parses JSON objects, and accumulates
    them into a list. It supports an optional `max_lines` limit, prints progress
    when `verbose` is True, and tolerates corrupted lines by skipping them while
    continuing processing.
    
    Parameters
    ----------
    file_name : str, optional
        Path to the jsonl file to read. If the provided path has no extension,
        `file_ext` is appended. Default:
        `'./Discover-MIDI-Dataset/DATA/Averages/all_midis_averages_data'`.
    file_ext : str, optional
        Extension to append when `file_name` has no extension. Default: `'.jsonl'`.
    max_lines : int, optional
        Maximum number of JSON records to read. If `max_lines` is negative (default),
        the function reads the entire file. If positive, reading stops after that
        many successfully parsed records. Default: `-1`.
    verbose : bool, optional
        If True, print progress and status messages and enable a `tqdm` progress
        iterator. If False, suppress progress output. Default: True.
    
    Returns
    -------
    list
        A list of parsed JSON objects (typically dictionaries) in the same order
        they appear in the file. Corrupted lines that fail JSON parsing are skipped
        and not included in the returned list.
    
    Raises
    ------
    FileNotFoundError
        If the resolved `file_name` does not exist when attempting to open it.
    OSError, IOError
        For other I/O related errors when opening or reading the file.
    
    Behavioral details
    ------------------
    - If `file_name` has no extension, `file_ext` is appended before opening.
    - Each line is parsed with `json.loads`. Lines that raise `json.JSONDecodeError`
      are skipped; a brief error message is printed when `verbose` is True.
    - A `KeyboardInterrupt` during iteration is caught: the function prints a
      stopping message (when `verbose` is True), closes the file, and returns the
      records parsed up to that point.
    - The function uses `tqdm` for a progress bar when `verbose` is True; the
      progress bar is disabled when `verbose` is False.
    - The function reads the file line-by-line and keeps only the parsed objects in
      memory; however, the returned list contains all parsed records and may be
      large depending on the file size.
    
    Notes
    -----
    - The function assumes UTF-8 (or the system default) encoding when opening the
      file. If your jsonl uses a different encoding, open the file separately and
      pass a path adjusted for that encoding or modify the function to accept an
      `encoding` parameter.
    - If strict validation or schema enforcement is required, validate each parsed
      record after reading.
    
    """

    if verbose:
        print('=' * 70)
        print('Reading jsonl file...')
        print('=' * 70)

    if not os.path.splitext(file_name)[1]:
        file_name += file_ext

    with open(file_name, 'r') as f:

        records = []
        gl_count = 0
        
        for i, line in tqdm.tqdm(enumerate(f), disable=not verbose):
            
            try:
                record = json.loads(line)
                records.append(record)
                gl_count += 1

                if max_lines > 0:
                    if gl_count == max_lines:
                        break

            except KeyboardInterrupt:
                if verbose:
                    print('=' * 70)
                    print('Stoping...')
                    print('=' * 70)
                    
                f.close()
    
                return records
               
            except json.JSONDecodeError:
                if verbose:
                    print('=' * 70)
                    print('[ERROR] Line', i, 'is corrupted! Skipping it...')
                    print('=' * 70)
                    
                continue
                
    f.close()
    
    if verbose:
        print('=' * 70)
        print('Loaded total of', gl_count, 'jsonl records.')
        print('=' * 70)
        print('Done!')
        print('=' * 70)

    return records

###################################################################################

def load_features_matrixes(features_matrixes_path='./Discover-MIDI-Dataset/DATA/Features Matrixes/features_matrixes.npz',
                           verbose=True
                          ):

    """Load a NumPy .npz archive containing precomputed feature matrices and return the array.
    
    This helper loads the array stored under the key 'features_matrixes' from a `.npz` file and optionally
    prints progress messages. It is a thin convenience wrapper around `numpy.load` that centralizes the
    default dataset path and consistent verbose logging.
    
    Parameters
    ----------
    features_matrixes_path : str, optional
        Path to a `.npz` file that contains a NumPy archive with a `features_matrixes` entry.
        Default: './Discover-MIDI-Dataset/DATA/Features Matrixes/features_matrixes.npz'.
    verbose : bool, optional
        If True, print simple progress messages and the shape of the loaded array. Default: True.
    
    Returns
    -------
    numpy.ndarray
        The array stored under the `features_matrixes` key in the archive. The shape and dtype depend on
        how the data were saved.
    
    Raises
    ------
    FileNotFoundError
        If the file at `features_matrixes_path` does not exist (raised by `numpy.load`).
    KeyError
        If the archive does not contain a `features_matrixes` entry.
    OSError
        If the file cannot be read as a NumPy archive or is corrupted (raised by `numpy.load`).
    
    Notes
    -----
    - This function expects the `.npz` archive to include an array named exactly `'features_matrixes'`.
    - The function does not validate the dimensionality or contents of the loaded array; downstream code
      should perform any required checks (for example, verifying the number of columns).
    - The function uses `numpy.load` which returns an `NpzFile` object; indexing with the key returns the
      stored array.
    """

    if verbose:
        print('=' * 70)
        print('Loading features matrixes...')

    fmats = np.load(features_matrixes_path)['features_matrixes']

    if verbose:
        
        print('Done!')
        print('=' * 70)

        print('Loaded features matrixes shape:', fmats.shape)
        print('=' * 70)        

    return fmats

###################################################################################

def plot_features_matrixes(features_matrixes,
                           scale_factor=128,
                           window_len=96,
                           poly_order=5,                           
                           figsize=(20, 10),
                           title="Features Matrixes Visualization",
                           fontsize=24,
                           linewidth=2,
                           dpi=300,
                           save_plot=None
                          ):

    """Plot a smoothed, color-coded visualization of aggregated feature vectors.
    
    This function accepts a 2D array of feature vectors (each 961 elements long), aggregates them by summation across rows,
    applies a quantile normalization and scaling step, smooths the resulting 1D signal with a Savitzky–Golay filter,
    and draws a multi-series plot that highlights predefined feature ranges (e.g., delta times, durations, MIDI-related
    features, harmonic chords). The plot can be displayed interactively or saved to disk.
    
    Parameters
    ----------
    features_matrixes : array-like, shape (n_samples, 961)
        A 2D array or array-like object where each row is a 961-dimensional feature vector. The function sums across
        rows to produce a single 961-length signal to visualize.
    scale_factor : int, optional (default=128)
        Multiplicative factor applied after quantile-transforming the aggregated signal. Larger values increase the
        dynamic range before smoothing.
    window_len : int, optional (default=96)
        Window length parameter for `scipy.signal.savgol_filter`. Must be a positive odd integer less than or equal to
        the signal length. Controls the smoothing window size.
    poly_order : int, optional (default=5)
        Polynomial order parameter for `savgol_filter`. Must be less than `window_len`. Controls the smoothness shape.
    figsize : tuple, optional (default=(20, 10))
        Figure size passed to `matplotlib.pyplot.subplots` (width, height) in inches.
    title : str, optional (default="Features Matrixes Visualization")
        Title text for the plot.
    fontsize : int, optional (default=24)
        Base font size used for the title, axis labels, and legend.
    linewidth : float, optional (default=2)
        Line width for each plotted feature-range curve.
    dpi : int, optional (default=300)
        Dots-per-inch resolution used when creating the figure and when saving to file.
    save_plot : str or None, optional (default=None)
        If a file path (string) is provided, the figure is saved to this path using `fig.savefig`. If `None`, the plot
        is displayed with `plt.show()`.
    
    Raises
    ------
    ValueError
        If `features_matrixes` does not have 961 columns (i.e., `features_matrixes.shape[1] != 961`).
    ValueError
        If `window_len` is not a positive odd integer or if `poly_order >= window_len` (these are requirements of
        `savgol_filter`). (Note: the function does not currently validate these explicitly; invalid values will raise
        errors from `savgol_filter`.)
    
    Notes
    -----
    - The function uses `sklearn.preprocessing.QuantileTransformer` with `n_quantiles=200` and `output_distribution="uniform"`
      to map the aggregated feature sums into a uniform distribution before scaling. This reduces the influence of outliers.
    - After quantile normalization the values are multiplied by `scale_factor` and cast to `int32` before smoothing.
    - Predefined feature index ranges and colors are used to separate and visually emphasize semantic groups:
      - Delta start times: indices 0–127
      - Durations: indices 128–255
      - MIDI patches (instruments): indices 256–383
      - MIDI pitches (instruments): indices 384–511
      - MIDI pitches (drums): indices 512–639
      - Harmonic chords: indices 640–960
    - The function draws a lightly shaded background (`axvspan`) for each range and a vertical separator line at each range end.
    - Side effects: creates a Matplotlib figure. If `save_plot` is provided the figure is written to disk; otherwise it is shown.
      In both cases the figure is closed at the end of the function to free resources.
    """
    
    if features_matrixes.shape[1] != 961:
        raise ValueError("Inputs must be 961-dimensional vectors.")

    fmats_sums = np.sum(features_matrixes, axis=0)

    qt = QuantileTransformer(
        n_quantiles=200, 
        output_distribution="uniform"
    )
    
    fmats_sums_norm = qt.fit_transform(fmats_sums.reshape(-1, 1)).flatten()
    fmats_sums_scaled = (fmats_sums_norm * scale_factor).astype(np.int32)

    fmats_sums_curve = savgol_filter(fmats_sums_scaled, window_len, poly_order)
        
    # Feature ranges
    ranges = {
        "Delta start times": (0, 128),
        "Durations": (128, 256),
        "MIDI patches (instruments)": (256, 384),
        "MIDI pitches (instruments)": (384, 512),
        "MIDI pitches (drums)": (512, 640),
        "Harmonic chords": (640, 961),
    }

    colors = {
        "Delta start times": "#4C72B0",
        "Durations": "#55A868",
        "MIDI patches (instruments)": "#C44E52",
        "MIDI pitches (instruments)": "#8172B2",
        "MIDI pitches (drums)": "#CCB974",
        "Harmonic chords": "#64B5CD",
    }

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="white")
    ax.set_facecolor("white")

    for label, (start, end) in ranges.items():
        ax.plot(
            np.arange(start, end),
            fmats_sums_curve[start:end],
            color=colors[label],
            linewidth=linewidth,
            label=label,
        )

        ax.axvspan(start, end, color=colors[label], alpha=0.08)
        ax.axvline(end, color="black", linewidth=0.6, alpha=0.4)

    ax.set_title(title, fontsize=fontsize, pad=20)
    ax.set_xlabel("Feature Index", fontsize=fontsize-6)
    ax.set_ylabel("Value", fontsize=fontsize-6)

    ax.grid(alpha=0.25)
    ax.legend(fontsize=fontsize-8, frameon=True)

    fig.tight_layout()

    if save_plot is not None:
        fig.savefig(
            save_plot,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white"
        )
    
    else:
        plt.show()
        
    plt.close()

###################################################################################

def report_features_counts(features_counts: List[Dict[str, Any]],
                           report_sample_size: int = -1,
                           verbose: bool = True,
                           human_readable_tokens: bool = False,
                           top_k_overall: int = 50,
                           validate_mappings: bool = True
                          ) -> Dict[str, Any]:

    """
    Generate a comprehensive report summarizing tokenized MIDI features counts.
    
    This function ingests a list of per-item feature-count dictionaries and computes aggregated and
    per-item statistics, per-group summaries, top-feature lists, distribution metrics (entropy, Gini),
    coverage of top-k tokens, and optional mapping validation for human-readable token labels. The
    report is returned as a structured dictionary and (optionally) printed as a human-readable text
    summary when `verbose` is True.
    
    Args:
        features_counts (List[Dict[str, Any]]):
            Required. A list where each element is a dictionary representing one item. Each item
            dictionary is expected to contain at least one of the following keys:
              - "features_counts": an iterable of (feature_index, count) pairs (or two-element lists/tuples).
              - "md5" or "id": optional MIDI identifier used for example lists; if absent, "<no-id>" is used.
            Each feature index is interpreted as a raw integer token index in the 0..1088 range.
        report_sample_size (int, optional):
            Number of items to analyze from the provided list. If None or negative (default -1),
            the entire input list is analyzed. If a non-negative integer smaller than the total
            number of items is provided, a random sample of that size is taken.
        verbose (bool, optional):
            If True (default), a human-readable summary text is printed to stdout and included in the
            returned report under "summary_text". When False, printing is suppressed but the structured
            report is still returned.
        human_readable_tokens (bool, optional):
            If True, token indices in top lists and per-group top lists are converted to human-readable
            labels (e.g., "dt_0", "patch_12", "pitch_drum_5", "chord_10", "vel_3"). If False (default),
            raw integer indices are used in those lists.
        top_k_overall (int, optional):
            How many top tokens to include in the overall top-features list (default 50).
        validate_mappings (bool, optional):
            If True (default), run a post-hoc validation that checks whether labels that should map
            to 0..127 actually do so; any detected mapping issues are returned in the "mapping_issues"
            list in the report.
    
    Returns:
        Dict[str, Any]: A dictionary containing the full report with the following top-level keys:
            - "summary_text" (str): A multi-line human-readable summary (also printed when verbose).
            - "summary" (dict): High-level numeric summary including:
                * "n_total_items": total number of input items
                * "n_analyzed_items": number of items actually analyzed (after sampling)
                * "total_feature_tokens": total token counts summed across analyzed items
                * "unique_features_used": number of distinct feature indices with non-zero counts
                * "sparsity": fraction of unused features (1 - unique/1089)
                * "feature_entropy_bits": Shannon entropy (bits) of the global token distribution
                * "feature_gini": Gini coefficient of the global token distribution
                * "coverage_top_10": fraction of tokens covered by the top 10 tokens
                * "coverage_top_50": fraction of tokens covered by the top 50 tokens
                * "drum_item_fraction": fraction of analyzed items that contain any drum-pitch tokens
                * "group_summaries": per-group summary dict (see below)
                * "stats_total_tokens": descriptive stats for total tokens per item (count, mean, median, std, min, max, 25%, 75%)
                * "stats_features_per_item": descriptive stats for number of nonzero features per item
            - "top_features" (dict):
                * "overall_top": list of top-k tuples. Each tuple is either (label_or_index, count, [example_ids...])
                  depending on human_readable_tokens. Example ids are limited to three per token.
                * "per_group_top": mapping from group name to a list of up to 20 (label_or_index, count) tuples.
            - "per_item_stats" (List[dict]): Per-item dictionaries with keys:
                * "id": item identifier (md5, id, or "<no-id>")
                * "total_tokens": sum of counts for that item
                * "nonzero_features": number of non-zero feature entries for that item
                * "unique_feature_fraction": nonzero_features / 1089
                * "top_5_features": top five features for that item (labels or raw indices depending on human_readable_tokens)
            - "raw_counters" (dict): mapping raw feature_index -> total count across analyzed items
            - "mapping_issues" (List[dict]): list of mapping issue objects (each {"raw_idx": int, "label": str})
              detected when validate_mappings is True; empty list otherwise.
    
    Feature group definitions and label mapping:
        The function uses a fixed partitioning of the 0..1088 token space into named groups:
          - delta_start_times: indices 0..127
          - durations: indices 128..255 (mapped to "dur_0" .. "dur_127")
          - patches_instruments: indices 256..383 (mapped to "patch_0" .. "patch_127")
          - drum_patch_token: index 384 (special token; also counted under patches_instruments)
          - pitches_instruments: indices 384..511 (mapped to "pitch_inst_0" .. "pitch_inst_127")
          - pitches_drums: indices 512..639 (mapped to "pitch_drum_0" .. "pitch_drum_127")
          - harmonic_chords: indices 640..960 (mapped to "chord_0" .. "chord_320")
          - velocities: indices 961..1088 (mapped to "vel_0" .. "vel_127")
        The human-readable label function follows these mappings; when human_readable_tokens is False,
        raw integer indices are used in outputs.
    
    Validation behavior:
        When validate_mappings is True the function attempts to parse numeric suffixes from generated
        labels and ensures that groups expected to be in the 0..127 range actually map to that range.
        Any violations are collected in "mapping_issues". The validation step is conservative and only
        flags labels whose numeric suffix cannot be parsed or falls outside the expected range.
    
    Performance and robustness notes:
        - The function tolerates malformed or missing per-item data: items without "features_counts"
          are treated as empty; feature pairs that cannot be parsed to two integers are skipped.
        - Sampling uses random.sample when a subset is requested; results will vary unless the caller
          seeds Python's random module externally.
        - The function uses numpy for numeric summaries and may raise an ImportError if numpy is not
          available in the runtime environment.
        - The function accumulates example item ids for each feature (up to 5 examples stored, 3 shown
          in the overall top list) to help inspect representative items.
    
    Raises:
        TypeError: If `features_counts` is not a list.
        (No other exceptions are intentionally raised; malformed feature pairs are skipped silently.)
    
    Example:
        >>> items = [
        ...     {"id": "a1", "features_counts": [[0, 2], [256, 1], [512, 3]]},
        ...     {"id": "b2", "features_counts": [[128, 1], [384, 2], [640, 1]]}
        ... ]
        >>> report = report_features_counts(items, report_sample_size=-1, verbose=False, human_readable_tokens=True)
        >>> print(report["summary"]["n_analyzed_items"])
        2
        >>> print(report["top_features"]["overall_top"][:3])
        [('dt_0', 2, ['a1']), ('pitch_drum_0', 3, ['a1']), ('patch_drum_128', 2, ['b2'])]
    
    """

    # --- basic validation and sampling ---
    if not isinstance(features_counts, list):
        raise TypeError("features_counts must be a list of dicts")

    n_total = len(features_counts)
    if n_total == 0:
        return {
            "summary_text": "No feature items provided.",
            "summary": {},
            "top_features": {},
            "per_item_stats": [],
            "raw_counters": {},
            "mapping_issues": []
        }

    if report_sample_size is None or report_sample_size < 0:
        sample = features_counts
    else:
        k = min(report_sample_size, n_total)
        sample = features_counts if k == n_total else random.sample(features_counts, k)

    # --- feature group definitions (raw indices) ---
    groups = {
        "delta_start_times": (0, 128),        # [0,128)
        "durations": (128, 256),              # [128,256)
        "patches_instruments": (256, 384),    # [256,384)
        "drum_patch_token": (384, 385),       # token 384 reserved for drum patch
        "pitches_instruments": (384, 512),    # [384,512) instrument pitches (we map with -384)
        "pitches_drums": (512, 640),          # [512,640) drum pitches
        "harmonic_chords": (640, 961),        # [640,961)
        "velocities": (961, 1089)             # [961,1089)
    }

    def feature_group(idx: int) -> Optional[str]:
        for name, (lo, hi) in groups.items():
            if lo <= idx < hi:
                return name
        return None

    # --- human-readable label mapping (final mapping per user's instruction) ---
    def human_label(idx: int) -> str:
        # delta start times 0..127
        if 0 <= idx < 128:
            return f"dt_{idx}"
        # durations 128..255 -> 0..127
        if 128 <= idx < 256:
            return f"dur_{idx - 128}"
        # instrument patches 256..383 -> 0..127
        if 256 <= idx < 384:
            return f"patch_{idx - 256}"
        # drum patch token 384 -> patch_drum_128 (384 - 256 == 128)
        if idx == 384:
            return f"patch_drum_{idx - 256}"
        # instrument pitches 384..511 -> pitch_inst_0..127 (use raw - 384)
        if 384 <= idx < 512:
            return f"pitch_inst_{idx - 384}"
        # drum pitches 512..639 -> pitch_drum_0..127
        if 512 <= idx < 640:
            return f"pitch_drum_{idx - 512}"
        # harmonic chords 640..960 -> chord_0..320
        if 640 <= idx < 961:
            return f"chord_{idx - 640}"
        # velocities 961..1088 -> vel_0..127
        if 961 <= idx < 1089:
            return f"vel_{idx - 961}"
        # fallback
        return str(idx)

    # --- aggregators ---
    global_counter = Counter()
    per_group_counter = {
        "delta_start_times": Counter(),
        "durations": Counter(),
        "patches_instruments": Counter(),
        "pitches_instruments": Counter(),
        "pitches_drums": Counter(),
        "harmonic_chords": Counter(),
        "velocities": Counter()
    }
    per_item_stats = []
    total_tokens_per_item = []
    features_per_item = []
    presence_counts_per_group = {k: 0 for k in per_group_counter}
    examples_per_feature = defaultdict(list)

    # iterate with progress bar (progress shown when verbose is True)
    for item in tqdm.tqdm(sample, disable=not verbose, desc="Processing feature items"):
        item_id = item.get("md5", item.get("id", "<no-id>"))
        feats = item.get("features_counts", [])
        parsed = []
        for pair in feats:
            if not pair:
                continue
            try:
                fidx = int(pair[0])
                cnt = int(pair[1])
            except Exception:
                continue
            parsed.append((fidx, cnt))

        total_counts = sum(cnt for _, cnt in parsed)
        nonzero = len(parsed)
        total_tokens_per_item.append(total_counts)
        features_per_item.append(nonzero)

        groups_seen = set()
        for fidx, cnt in parsed:
            global_counter[fidx] += cnt
            grp = feature_group(fidx)
            # map raw feature into the appropriate per-group counter
            if grp == "delta_start_times":
                per_group_counter["delta_start_times"][fidx] += cnt
                groups_seen.add("delta_start_times")
            elif grp == "durations":
                per_group_counter["durations"][fidx] += cnt
                groups_seen.add("durations")
            elif grp == "patches_instruments":
                per_group_counter["patches_instruments"][fidx] += cnt
                groups_seen.add("patches_instruments")
            elif fidx == 384:
                # special drum patch token: count it under patches_instruments as well (but label as drum)
                per_group_counter["patches_instruments"][fidx] += cnt
                groups_seen.add("patches_instruments")
            elif grp == "pitches_instruments":
                per_group_counter["pitches_instruments"][fidx] += cnt
                groups_seen.add("pitches_instruments")
            elif grp == "pitches_drums":
                per_group_counter["pitches_drums"][fidx] += cnt
                groups_seen.add("pitches_drums")
            elif grp == "harmonic_chords":
                per_group_counter["harmonic_chords"][fidx] += cnt
                groups_seen.add("harmonic_chords")
            elif grp == "velocities":
                per_group_counter["velocities"][fidx] += cnt
                groups_seen.add("velocities")
            # store example ids (limit to 5)
            if len(examples_per_feature[fidx]) < 5:
                examples_per_feature[fidx].append(item_id)

        for g in groups_seen:
            presence_counts_per_group[g] += 1

        top5 = sorted(parsed, key=lambda x: x[1], reverse=True)[:5]
        top5_display = [(human_label(f), c) if human_readable_tokens else (f, c) for f, c in top5]

        per_item_stats.append({
            "id": item_id,
            "total_tokens": total_counts,
            "nonzero_features": nonzero,
            "unique_feature_fraction": (nonzero / 1089),
            "top_5_features": top5_display
        })

    # --- aggregated statistics ---
    total_feature_tokens = int(sum(global_counter.values()))
    unique_features_used = int(len([k for k, v in global_counter.items() if v > 0]))
    sparsity = 1.0 - (unique_features_used / 1089)

    # per-group summaries
    group_summaries = {}
    for gname, counter in per_group_counter.items():
        total = int(sum(counter.values()))
        unique = int(len([k for k, v in counter.items() if v > 0]))
        top5 = counter.most_common(5)
        pct_tokens = (total / total_feature_tokens * 100) if total_feature_tokens > 0 else 0.0
        presence_fraction = (presence_counts_per_group[gname] / len(sample)) if len(sample) > 0 else 0.0
        top5_display = [(human_label(f), c) if human_readable_tokens else (f, c) for f, c in top5]
        group_summaries[gname] = {
            "total_tokens": total,
            "unique_features": unique,
            "top5": top5_display,
            "pct_of_all_tokens": float(pct_tokens),
            "presence_fraction": float(presence_fraction)
        }

    # overall top features
    top_overall_raw = global_counter.most_common(top_k_overall)
    top_overall_display = [
        (human_label(f), c, examples_per_feature.get(f, [])[:3]) if human_readable_tokens else (f, c, examples_per_feature.get(f, [])[:3])
        for f, c in top_overall_raw
    ]

    # stats helpers
    def safe_stats(arr):
        if not arr:
            return {"count": 0}
        a = np.array(arr, dtype=float)
        return {
            "count": int(a.size),
            "sum": float(a.sum()),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "std": float(a.std(ddof=0)),
            "min": float(a.min()),
            "max": float(a.max()),
            "25%": float(np.percentile(a, 25)),
            "75%": float(np.percentile(a, 75))
        }

    stats_total_tokens = safe_stats(total_tokens_per_item)
    stats_features_per_item = safe_stats(features_per_item)

    # additional informative metrics
    def distribution_entropy(counter: Counter) -> float:
        vals = np.array(list(counter.values()), dtype=float)
        if vals.sum() <= 0:
            return 0.0
        p = vals / vals.sum()
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    def gini_coefficient(counter: Counter) -> float:
        vals = np.array(sorted(counter.values()))
        if vals.size == 0 or vals.sum() == 0:
            return 0.0
        n = vals.size
        index = np.arange(1, n + 1)
        return float((2 * (index * vals).sum()) / (n * vals.sum()) - (n + 1) / n)

    def top_k_coverage(counter: Counter, k: int) -> float:
        if not counter:
            return 0.0
        total = sum(counter.values())
        topk = sum(v for _, v in counter.most_common(k))
        return float(topk / total) if total > 0 else 0.0

    feature_entropy = distribution_entropy(global_counter)
    feature_gini = gini_coefficient(global_counter)
    coverage_top_10 = top_k_coverage(global_counter, 10)
    coverage_top_50 = top_k_coverage(global_counter, 50)

    # drum item fraction (items that include any drum pitch tokens)
    drum_items = 0
    for item in sample:
        feats = item.get("features_counts", [])
        has_drum = any((512 <= int(pair[0]) < 640) for pair in feats if pair)
        if has_drum:
            drum_items += 1
    drum_item_fraction = drum_items / len(sample)

    # --- mapping validation (optional) ---
    mapping_issues = []
    if validate_mappings:
        # check that labels that should be 0..127 are indeed in that range
        for raw_idx in global_counter.keys():
            lbl = human_label(raw_idx)
            # try to parse numeric suffix
            try:
                suffix = int(lbl.split('_')[-1])
            except Exception:
                continue
            # groups that must be 0..127
            if any(lbl.startswith(prefix) for prefix in ("patch_", "patch_drum_", "pitch_inst_", "pitch_drum_", "dur_", "vel_")):
                if not (0 <= suffix <= 127):
                    mapping_issues.append({"raw_idx": int(raw_idx), "label": lbl})
        # deduplicate
        mapping_issues = sorted({(m["raw_idx"], m["label"]) for m in mapping_issues}, key=lambda x: x[0])
        mapping_issues = [{"raw_idx": r, "label": l} for r, l in mapping_issues]

    # --- build human-readable summary text ---
    lines = []
    lines.append("Note: this report analyzes MIDI features (not MIDIs).")
    lines.append(f"Analyzed feature items: {len(sample)} (from {n_total} available).")
    lines.append(f"Total feature tokens across analyzed set: {total_feature_tokens:,}.")
    lines.append(f"Unique features used: {unique_features_used} / 1089. Sparsity: {sparsity:.3f}.")
    lines.append(f"Feature distribution entropy: {feature_entropy:.3f} bits. Gini coefficient: {feature_gini:.3f}.")
    lines.append(f"Top-10 token coverage: {coverage_top_10:.2%}; Top-50 coverage: {coverage_top_50:.2%}.")
    lines.append(f"Fraction of items containing drum pitches: {drum_item_fraction:.2%}.")
    lines.append("")
    lines.append("Per-group summary (group: total_tokens; unique_features; pct_of_all_tokens; presence_fraction):")
    for gname, gs in group_summaries.items():
        lines.append(
            f"  - {gname}: {gs['total_tokens']:,} tokens; {gs['unique_features']} unique; "
            f"{gs['pct_of_all_tokens']:.2f}% of tokens; present in {gs['presence_fraction']:.2%} of items."
        )
    lines.append("")
    lines.append("Overall top features (label: total_count) and example ids:")
    for entry in top_overall_display[:10]:
        label, cnt, examples = entry
        ex = ", ".join(examples) if examples else "none"
        lines.append(f"  - {label}: {cnt:,}  (examples: {ex})")
    lines.append("")
    lines.append("Per-item totals (summary):")
    lines.append(
        f"  - total feature tokens per item: mean={stats_total_tokens.get('mean',0):.1f}, "
        f"median={stats_total_tokens.get('median',0):.1f}, std={stats_total_tokens.get('std',0):.1f}"
    )
    lines.append(
        f"  - nonzero feature count per item: mean={stats_features_per_item.get('mean',0):.1f}, "
        f"median={stats_features_per_item.get('median',0):.1f}"
    )
    lines.append("")
    lines.append("Top 5 features per group (label: total_count):")
    for gname, gs in group_summaries.items():
        top5_str = "; ".join(f"{lbl}:{c}" for lbl, c in gs["top5"])
        lines.append(f"  - {gname}: {top5_str if top5_str else 'none'}")

    if validate_mappings and mapping_issues:
        lines.append("")
        lines.append("Mapping issues detected (raw_idx -> label):")
        for m in mapping_issues:
            lines.append(f"  - {m['raw_idx']} -> {m['label']}")

    summary_text = "\n".join(lines)

    # --- structured top_features per group and overall ---
    top_features = {
        "overall_top": top_overall_display,
        "per_group_top": {
            g: ([(human_label(f) if human_readable_tokens else f, c) for f, c in per_group_counter[g].most_common(20)])
            for g in per_group_counter
        }
    }

    # --- final report dict ---
    report = {
        "summary_text": summary_text,
        "summary": {
            "n_total_items": n_total,
            "n_analyzed_items": len(sample),
            "total_feature_tokens": total_feature_tokens,
            "unique_features_used": unique_features_used,
            "sparsity": float(sparsity),
            "feature_entropy_bits": float(feature_entropy),
            "feature_gini": float(feature_gini),
            "coverage_top_10": float(coverage_top_10),
            "coverage_top_50": float(coverage_top_50),
            "drum_item_fraction": float(drum_item_fraction),
            "group_summaries": group_summaries,
            "stats_total_tokens": stats_total_tokens,
            "stats_features_per_item": stats_features_per_item
        },
        "top_features": top_features,
        "per_item_stats": per_item_stats,
        "raw_counters": dict(global_counter),
        "mapping_issues": mapping_issues
    }

    if verbose:
        print(summary_text)

    return report

###################################################################################

# General MIDI families mapping (0-based patch numbers)
_GM_FAMILIES = {
    'Piano': list(range(0, 8)),
    'Chromatic Percussion': list(range(8, 16)),
    'Organ': list(range(16, 24)),
    'Guitar': list(range(24, 32)),
    'Bass': list(range(32, 40)),
    'Strings': list(range(40, 48)),
    'Ensemble': list(range(48, 56)),
    'Brass': list(range(56, 64)),
    'Reed': list(range(64, 72)),
    'Pipe': list(range(72, 80)),
    'Synth Lead': list(range(80, 88)),
    'Synth Pad': list(range(88, 96)),
    'Synth Effects': list(range(96, 104)),
    'Ethnic': list(range(104, 112)),
    'Percussive': list(range(112, 120)),
    'Sound Effects': list(range(120, 128)),
    'Drums': [128]
}
###################################################################################

_PATCH_TO_FAMILY = {}

for fam, patches in _GM_FAMILIES.items():
    for p in patches:
        _PATCH_TO_FAMILY[p] = fam
        
###################################################################################

def _pitch_to_pc_oct(pitch):
    
    pc = int(pitch) % 12
    octave = int(pitch) // 12
    
    return pc, octave

###################################################################################

def _entropy(counter):
    
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in counter.values():
        p = v / total
        if p > 0:
            ent -= p * math.log2(p)
            
    return ent

###################################################################################

def _gini(counter):
    
    vals = sorted(counter.values())
    n = len(vals)
    if n == 0:
        return 0.0
    total = sum(vals)
    if total == 0:
        return 0.0
    cum = 0
    for i, v in enumerate(vals, start=1):
        cum += (2 * i - n - 1) * v
        
    return cum / (n * total)

###################################################################################

def _dominant_fraction(counter):
    
    if not counter:
        return 0.0
    total = sum(counter.values())
    top = counter.most_common(1)[0][1]
    
    return top / total if total > 0 else 0.0

###################################################################################

def _classify_midi(file_patch_counter, has_drums, total_events):
    
    unique_non_drum = len([p for p in file_patch_counter.keys() if p != 128])
    if total_events == 0:
        return 'empty'
    if has_drums and unique_non_drum == 0:
        return 'drum'
    dom_frac = _dominant_fraction(file_patch_counter)
    if (not has_drums) and dom_frac >= 0.80 and unique_non_drum <= 2:
        return 'melody'
    if has_drums and unique_non_drum >= 1:
        return 'song'
    if len(file_patch_counter) >= 3 or dom_frac < 0.8:
        return 'song'
    
    return 'melody'

###################################################################################

def _guess_genre_from_families(family_counter, has_drums, patch_counter):
    
    if not family_counter:
        return 'unknown'
    
    total = sum(family_counter.values()) or 1
    fam_frac = {fam: cnt / total for fam, cnt in family_counter.items()}
    
    if fam_frac.get('Piano', 0) >= 0.7 and not has_drums:
        return 'solo piano'
    
    if fam_frac.get('Piano', 0) >= 0.4 and fam_frac.get('Strings', 0) >= 0.2:
        return 'classical / chamber'
    
    if fam_frac.get('Strings', 0) >= 0.4 and (fam_frac.get('Brass', 0) + fam_frac.get('Ensemble', 0) >= 0.2):
        return 'cinematic / orchestral'
    
    if (fam_frac.get('Synth Pad', 0) + fam_frac.get('Synth Lead', 0) + fam_frac.get('Synth Effects', 0)) >= 0.5:
        if has_drums:
            return 'electronic / synthpop'
        
        return 'ambient / synth'
    
    if fam_frac.get('Guitar', 0) >= 0.35 and fam_frac.get('Bass', 0) >= 0.15 and has_drums:
        return 'rock / pop'
    
    if fam_frac.get('Bass', 0) >= 0.4 and has_drums:
        return 'dance / electronic'
    
    if fam_frac.get('Reed', 0) >= 0.25 or fam_frac.get('Brass', 0) >= 0.25:
        return 'jazz / big band'
    
    if fam_frac.get('Percussive', 0) >= 0.5 or (128 in patch_counter and sum(v for k,v in patch_counter.items() if k==128) / sum(patch_counter.values()) >= 0.6):
        return 'drum loop / percussion'
    
    if len([f for f, c in family_counter.items() if c > 0]) <= 2 and not has_drums:
        return 'acoustic / small ensemble'
    
    return 'mixed / multi-instrument'

###################################################################################

def _chao1_estimator(counter):
    
    """
    Chao1 estimator for species richness from frequency counts in counter.
    Returns estimated richness (float).
    """
    
    S_obs = len([k for k, v in counter.items() if v > 0])
    freqs = Counter(counter.values())
    f1 = freqs.get(1, 0)
    f2 = freqs.get(2, 0)
    
    if f2 > 0:
        return S_obs + (f1 * f1) / (2.0 * f2)
    else:
        # fallback when no doubletons: bias-corrected form
        return S_obs + f1 * (f1 - 1) / 2.0
    
###################################################################################

def report_pitches_patches_counts(pitches_patches_counts,
                                  report_sample_size=-1,
                                  verbose=True
                                 ):
    
    """
    Analyze a collection of MIDI pitch/patch frequency summaries and produce a compact
    human-readable summary plus a structured report of global statistics.
    
    This function ingests a list of per-file pitch/patch frequency dictionaries (the
    expected output of a prior MIDI parsing step) and computes aggregated counts,
    diversity metrics, distribution summaries, simple MIDI-type and genre guesses,
    and other diagnostics useful for dataset exploration and quality checks.
    
    Parameters
    ----------
    pitches_patches_counts : list of dict
        A list where each element summarizes one MIDI file. Each element must be a
        mapping-like object (typically a dict) that contains, at minimum, the key
        ``'pp_counts'`` whose value is an iterable of 2-tuples ``(key, count)``.
        - ``key`` must itself be a 2-tuple or 2-list ``(pitch, patch)`` where
          ``pitch`` and ``patch`` are integers (pitch: 0-127 MIDI note number;
          patch: 0-127 General MIDI patch, with 128 reserved for drums in this
          codebase).
        - ``count`` must be an integer (number of events for that pitch/patch pair).
        Example element:
            {'pp_counts': [((60, 0), 12), ((64, 0), 8), ((36, 128), 24)]}
        The function ignores malformed entries (non-2-tuples, non-integer counts,
        or keys that cannot be coerced to two ints).
    
    report_sample_size : int or None, optional
        Controls how many files from ``pitches_patches_counts`` are analyzed:
        - ``None`` or ``-1`` : analyze all files (default).
        - ``0`` : analyze a small sample of up to 10 files (``min(10, n_files)``).
        - positive integer ``r`` : analyze up to ``min(r, n_files)`` files.
        Negative values other than ``-1`` are treated as "all files".
        The chosen subset is the first ``sample_size`` entries in the provided list.
    
    verbose : bool, optional
        If True (default) the function:
          - shows a progress iterator (``tqdm``) while processing files, and
          - prints a multi-line human-readable summary to stdout at the end.
        If False, processing is silent and no summary is printed.
    
    Returns
    -------
    report : dict
        A dictionary with three top-level keys:
        - ``'summary_text'`` : str
            A human-readable multi-line summary describing the analyzed sample,
            counts, top items, diversity metrics, and notes.
        - ``'summary_dict'`` : dict
            A structured summary containing numeric and list fields suitable for
            programmatic consumption. Important keys include:
              - ``n_files_total`` : total number of input entries
              - ``n_files_analyzed`` : number of files actually analyzed (sample size)
              - ``sampled`` : bool, True if a subset was analyzed
              - ``total_events`` : total summed event counts across analyzed files
              - ``used_pairs`` : number of distinct (pitch,patch) pairs observed
              - ``possible_pairs`` : constant (128 * 129) used for sparsity calc
              - ``sparsity`` : fraction of unused pitch/patch pairs (1 - used/possible)
              - ``distinct_pitches`` : number of distinct pitch values observed
              - ``distinct_patches`` : number of distinct patch values observed
              - ``distinct_non_drum_patches`` : number of distinct patches excluding 128
              - ``top_global_patches`` : list of (patch, count) tuples sorted by count
              - ``top_global_pitches`` : list of (pitch, count) tuples sorted by count
              - ``top_global_pitch_patch_pairs`` : list of ((pitch,patch), count)
              - ``top_global_families`` : list of (family_name, count)
              - ``pitchclass_distribution`` : dict mapping 0..11 to counts (pitch classes)
              - ``octave_distribution`` : dict mapping octave numbers to counts
              - ``global_metrics`` : dict of computed metrics (entropy, gini, simpson,
                pielou, chao1 estimates, normalized entropies, etc.)
              - ``topk_coverage_pitches`` / ``topk_coverage_patches`` : dicts of coverage
                fractions for top-1,3,5,10,20 items
              - ``cumulative_top_pp_coverage_sample`` : list of tuples describing
                cumulative coverage for the top pitch-patch pairs
              - ``midi_type_counts`` : dict counting classified file types (e.g., 'song',
                'melody', 'drum', 'empty')
              - ``genre_guess_counts`` : dict counting guessed genres from family mix
        - ``'global'`` : dict
            Raw aggregated counters and mappings:
              - ``pitch_counter`` : dict mapping pitch -> total count
              - ``patch_counter`` : dict mapping patch -> total count
              - ``pp_counter`` : dict mapping "pitch,patch" string -> total count
              - ``family_counter`` : dict mapping GM family name -> total count
              - ``pitchclass_counter`` : dict mapping 0..11 -> counts
              - ``octave_counter`` : dict mapping octave -> counts
    
    Behavior and computed metrics
    -----------------------------
    The function performs the following high-level operations:
    - Normalizes and validates each per-file ``pp_counts`` entry, coercing pitch
      and patch to integers and ignoring invalid items.
    - Aggregates global counters for pitches, patches, pitch-patch pairs,
      pitch-classes (mod 12), octaves, and General MIDI families (using an
      internal GM family mapping).
    - Detects presence of drums (patch == 128) and classifies each file into a
      simple MIDI type via ``_classify_midi`` (e.g., 'melody', 'song', 'drum',
      'empty') and guesses a coarse genre from family mix via ``_guess_genre_from_families``.
    - Computes diversity and dominance metrics:
      - Shannon entropy (bits) for pitches, patches, families
      - Gini coefficient for patches/pitches
      - Pielou evenness (entropy / log2(S))
      - Simpson index (sum p^2)
      - Chao1 richness estimator for pitches and patches
      - Top-k coverage fractions and cumulative coverage for top pitch-patch pairs
    - Computes sparsity relative to the full pitch x patch space (128 * 129).
    - Produces both a human-readable summary string and a structured dictionary
      suitable for downstream programmatic analysis or visualization.
    
    Assumptions and conventions
    ---------------------------
    - Pitches are standard MIDI note numbers (integers). Octave is computed as
      ``pitch // 12`` and pitch-class as ``pitch % 12``.
    - Patch numbers follow the internal mapping where 0..127 are GM patches and
      128 is used to indicate drums in this dataset.
    - The function treats missing or malformed ``pp_counts`` entries as empty and
      continues processing other files.
    - The GM family mapping used to produce family-level summaries is internal to
      the module and maps patch numbers to family names (e.g., 'Piano', 'Guitar',
      'Synth Pad', etc.). Unknown patch numbers are labeled 'Unknown'.
    
    Errors and exceptions
    ---------------------
    - Raises ``TypeError`` if ``pitches_patches_counts`` is not a list.
    - The function is robust to malformed per-file entries: it silently skips
      items that cannot be parsed as ``((pitch, patch), count)`` pairs.
    - Division-by-zero is avoided internally; many derived metrics fall back to
      safe defaults (e.g., 0.0) when insufficient data is present.
    
    Performance
    -----------
    - Time complexity is linear in the total number of valid pitch-patch events
      across the analyzed files (i.e., O(total_events + number_of_pairs)).
    - Memory usage is proportional to the number of distinct pitches, patches,
      and pitch-patch pairs observed.
    
    Side effects
    ------------
    - If ``verbose`` is True, a progress bar (``tqdm``) is shown during processing
      and the final human-readable summary is printed to stdout.
    
    Example
    -------
    >>> sample = [
    ...     {'pp_counts': [((60, 0), 12), ((64, 0), 8), ((36, 128), 24)]},
    ...     {'pp_counts': [((60, 0), 5), ((67, 24), 10)]}
    ... ]
    >>> report = report_pitches_patches_counts(sample, report_sample_size=0, verbose=False)
    >>> print(report['summary_dict']['n_files_analyzed'])
    2
    >>> print(report['summary_dict']['top_global_patches'][:3])
    [(0, 25), (128, 24), (24, 10)]
    
    Notes
    -----
    - The function is intended for exploratory analysis of pre-aggregated MIDI event
      counts. It does not parse raw MIDI files; it expects the input to already be
      summarized into pitch/patch counts per file.
    - The human-readable summary is designed for quick inspection; use ``summary_dict``
      and ``global`` for programmatic consumption and plotting.
    """

    if not isinstance(pitches_patches_counts, list):
        raise TypeError("pitches_patches_counts must be a list of dicts")

    n_files = len(pitches_patches_counts)

    # Determine sample size up front according to requested semantics
    if report_sample_size is None:
        sample_size = n_files  # treat None as -1 (all)
    else:
        try:
            r = int(report_sample_size)
        except Exception:
            r = -1
        if r == -1:
            sample_size = n_files
        elif r == 0:
            sample_size = min(10, n_files)
        elif r > 0:
            sample_size = min(r, n_files)
        else:
            # fallback: treat negative other than -1 as all
            sample_size = n_files

    # Choose the dataset to analyze (sample or full)
    if sample_size < n_files:
        data = pitches_patches_counts[:sample_size]
        sampled = True
    else:
        data = pitches_patches_counts
        sampled = False

    # Global aggregators (for the chosen data)
    global_total_events = 0
    global_pitch_counter = Counter()
    global_patch_counter = Counter()
    global_pp_counter = Counter()
    global_pc_counter = Counter()
    global_octave_counter = Counter()
    global_family_counter = Counter()

    midi_type_counts = Counter()
    genre_guess_counts = Counter()

    iterator = tqdm.tqdm(data, disable=not verbose)
    for entry in iterator:
        raw_pp = entry.get('pp_counts', []) or []
        normalized = []
        for item in raw_pp:
            try:
                key, cnt = item
            except Exception:
                continue
            if isinstance(key, (list, tuple)) and len(key) == 2:
                try:
                    pitch = int(key[0])
                    patch = int(key[1])
                except Exception:
                    continue
            else:
                continue
            try:
                cnt = int(cnt)
            except Exception:
                continue
            normalized.append(((pitch, patch), cnt))

        file_patch_counter = Counter()
        file_family_counter = Counter()
        total_events = 0
        has_drums = False
        only_drums = True if normalized else False

        for (pitch, patch), cnt in normalized:
            if cnt <= 0:
                continue
            total_events += cnt
            file_patch_counter[patch] += cnt
            global_pitch_counter[pitch] += cnt
            global_patch_counter[patch] += cnt
            global_pp_counter[(pitch, patch)] += cnt

            pc, octv = _pitch_to_pc_oct(pitch)
            global_pc_counter[pc] += cnt
            global_octave_counter[octv] += cnt

            fam = _PATCH_TO_FAMILY.get(patch, 'Unknown')
            file_family_counter[fam] += cnt
            global_family_counter[fam] += cnt

            if patch == 128:
                has_drums = True
            else:
                only_drums = False

        midi_type = _classify_midi(file_patch_counter, has_drums, total_events)
        midi_type_counts[midi_type] += 1

        genre_guess = _guess_genre_from_families(file_family_counter, has_drums, file_patch_counter)
        genre_guess_counts[genre_guess] += 1

        global_total_events += total_events

    # Global derived stats
    top_global_pitches = global_pitch_counter.most_common(50)
    top_global_patches = global_patch_counter.most_common(50)
    top_global_pp = global_pp_counter.most_common(50)
    top_global_families = global_family_counter.most_common(20)

    pitch_entropy_global = _entropy(global_pitch_counter)
    patch_entropy_global = _entropy(global_patch_counter)
    family_entropy_global = _entropy(global_family_counter)
    pitch_gini_global = _gini(global_pitch_counter)
    patch_gini_global = _gini(global_patch_counter)

    # Additional diversity metrics
    unique_pitches = len(global_pitch_counter)
    unique_patches = len(global_patch_counter)
    total_events = global_total_events or 1  # avoid div by zero

    # Pielou evenness for pitches and patches
    pielou_pitch = pitch_entropy_global / math.log2(unique_pitches) if unique_pitches > 1 else 0.0
    pielou_patch = patch_entropy_global / math.log2(unique_patches) if unique_patches > 1 else 0.0

    # Simpson index (dominance) for pitches and patches
    def simpson_index(counter):
        total = sum(counter.values()) or 1
        s = 0.0
        for v in counter.values():
            p = v / total
            s += p * p
        return s

    simpson_pitch = simpson_index(global_pitch_counter)
    simpson_patch = simpson_index(global_patch_counter)

    # Chao1 richness estimators
    chao1_pitches = _chao1_estimator(global_pitch_counter)
    chao1_patches = _chao1_estimator(global_patch_counter)

    # Normalized entropies (0-1)
    norm_pitch_entropy = pitch_entropy_global / math.log2(unique_pitches) if unique_pitches > 1 else 0.0
    norm_patch_entropy = patch_entropy_global / math.log2(unique_patches) if unique_patches > 1 else 0.0
    norm_family_entropy = family_entropy_global / math.log2(len(global_family_counter)) if len(global_family_counter) > 1 else 0.0

    # Pitch-class and octave distributions
    pitchclass_distribution = {pc: global_pc_counter.get(pc, 0) for pc in range(12)}
    if global_octave_counter:
        min_oct = min(global_octave_counter.keys())
        max_oct = max(global_octave_counter.keys())
    else:
        min_oct, max_oct = 0, 0
    octave_distribution = {octv: global_octave_counter.get(octv, 0) for octv in range(min_oct, max_oct + 1)}

    # Sparsity
    POSSIBLE_PITCHES = 128
    POSSIBLE_PATCHES = 129
    possible_pairs = POSSIBLE_PITCHES * POSSIBLE_PATCHES
    used_pairs = len(global_pp_counter)
    sparsity = 1.0 - (used_pairs / possible_pairs) if possible_pairs else 1.0

    # Top-k coverage (global)
    def top_k_coverage(counter, k):
        total = sum(counter.values()) or 1
        top_sum = sum(v for _, v in counter.most_common(k))
        return top_sum / total

    topk = {k: top_k_coverage(global_pitch_counter, k) for k in (1, 3, 5, 10, 20)}
    topk_patches = {k: top_k_coverage(global_patch_counter, k) for k in (1, 3, 5, 10, 20)}

    # Cumulative coverage for top N pitch-patch pairs
    total_pp = sum(global_pp_counter.values()) or 1
    cum = 0
    cum_coverage = []
    for i, ((p, patch), cnt) in enumerate(global_pp_counter.most_common(100), start=1):
        cum += cnt
        cum_coverage.append((i, (p, patch), cnt, cum / total_pp))

    # Family mix description
    family_total = sum(global_family_counter.values()) or 1
    family_mix = [(fam, cnt, cnt / family_total) for fam, cnt in top_global_families]
    dominant_family, dom_cnt = top_global_families[0] if top_global_families else ('None', 0)
    dominant_family_frac = (dom_cnt / family_total) if family_total else 0.0
    if dominant_family_frac >= 0.7:
        family_character = f"Dominated by {dominant_family} ({dominant_family_frac:.0%})"
    elif dominant_family_frac >= 0.4:
        family_character = f"Strong presence of {dominant_family} ({dominant_family_frac:.0%}) with other families"
    else:
        family_character = "Diverse mix of patch families"

    # Build human-readable summary (no per-file stats)
    lines = []
    lines.append(f"Files analyzed: {sample_size} {'(sampled)' if sampled else '(all)'}")
    lines.append(f"Total events: {global_total_events:,}")
    lines.append(f"Unique pitch-patch pairs observed: {used_pairs} / {possible_pairs} (sparsity {sparsity:.4f})")
    lines.append("")
    lines.append("Instrument and patch overview")
    lines.append(f"  Distinct patches observed: {unique_patches}")
    non_drum_patches = len([p for p in global_patch_counter.keys() if p != 128])
    lines.append(f"  Distinct non-drum instruments observed: {non_drum_patches}")
    lines.append("")
    lines.append("Global top patches and families")
    for patch, cnt in top_global_patches[:10]:
        fam = _PATCH_TO_FAMILY.get(patch, 'Unknown')
        lines.append(f"  Patch {patch:3d} ({fam:14s}): {cnt:,}")
    lines.append("")
    lines.append("Top patch families")
    for fam, cnt, frac in family_mix[:8]:
        lines.append(f"  {fam:20s} {cnt:,}  ({frac:.1%})")
    lines.append(f"Family mix summary: {family_character}")
    lines.append("")
    lines.append("Global diversity metrics")
    lines.append(f"  Pitch entropy: {pitch_entropy_global:.4f} bits; Pitch Gini: {pitch_gini_global:.4f}; Pielou: {pielou_pitch:.4f}; Simpson: {simpson_pitch:.4f}")
    lines.append(f"  Patch entropy: {patch_entropy_global:.4f} bits; Patch Gini: {patch_gini_global:.4f}; Pielou: {pielou_patch:.4f}; Simpson: {simpson_patch:.4f}")
    lines.append(f"  Family entropy: {family_entropy_global:.4f} bits; Normalized family entropy: {norm_family_entropy:.4f}")
    lines.append(f"  Chao1 estimated pitch richness: {chao1_pitches:.2f}; Chao1 estimated patch richness: {chao1_patches:.2f}")
    lines.append("")
    lines.append("Top-k coverage (pitches)")
    lines.append("  " + ", ".join(f"top{k}:{topk[k]:.1%}" for k in sorted(topk.keys())))
    lines.append("Top-k coverage (patches)")
    lines.append("  " + ", ".join(f"top{k}:{topk_patches[k]:.1%}" for k in sorted(topk_patches.keys())))
    lines.append("")
    lines.append("Pitch-class distribution (0=C ... 11=B)")
    total_pc = sum(pitchclass_distribution.values()) or 1
    pc_line = "  " + ", ".join(f"{pc}:{pitchclass_distribution[pc]:,} ({pitchclass_distribution[pc]/total_pc:.1%})" for pc in range(12))
    lines.append(pc_line)
    lines.append("")
    lines.append("Octave distribution")
    if octave_distribution:
        od_line = "  " + ", ".join(f"{octv}:{octave_distribution[octv]:,}" for octv in sorted(octave_distribution.keys()))
        lines.append(od_line)
    else:
        lines.append("  (no octave data)")
    lines.append("")
    lines.append("File type and genre counts (aggregated)")
    for t, c in midi_type_counts.most_common():
        lines.append(f"  {t:10s}: {c}")
    lines.append("")
    lines.append("Genre guesses from patch mix")
    for g, c in genre_guess_counts.most_common():
        lines.append(f"  {g:20s}: {c}")
    lines.append("")
    lines.append("Notes")
    lines.append("  - Metrics computed on sample of entries" if sampled else "  - Metrics computed on full dataset")

    summary_text = "\n".join(lines)

    # Build structured summary dict (no per-file stats)
    summary_dict = {
        'n_files_total': n_files,
        'n_files_analyzed': sample_size,
        'sampled': sampled,
        'total_events': global_total_events,
        'used_pairs': used_pairs,
        'possible_pairs': possible_pairs,
        'sparsity': sparsity,
        'distinct_pitches': unique_pitches,
        'distinct_patches': unique_patches,
        'distinct_non_drum_patches': non_drum_patches,
        'top_global_patches': top_global_patches,
        'top_global_pitches': top_global_pitches,
        'top_global_pitch_patch_pairs': top_global_pp,
        'top_global_families': top_global_families,
        'pitchclass_distribution': pitchclass_distribution,
        'octave_distribution': octave_distribution,
        'global_metrics': {
            'pitch_entropy_bits': round(pitch_entropy_global, 4),
            'patch_entropy_bits': round(patch_entropy_global, 4),
            'family_entropy_bits': round(family_entropy_global, 4),
            'pitch_gini': round(pitch_gini_global, 4),
            'patch_gini': round(patch_gini_global, 4),
            'pielou_pitch': round(pielou_pitch, 4),
            'pielou_patch': round(pielou_patch, 4),
            'simpson_pitch': round(simpson_pitch, 6),
            'simpson_patch': round(simpson_patch, 6),
            'chao1_pitches': round(chao1_pitches, 2),
            'chao1_patches': round(chao1_patches, 2),
            'normalized_pitch_entropy': round(norm_pitch_entropy, 4),
            'normalized_patch_entropy': round(norm_patch_entropy, 4),
            'normalized_family_entropy': round(norm_family_entropy, 4)
        },
        'topk_coverage_pitches': topk,
        'topk_coverage_patches': topk_patches,
        'cumulative_top_pp_coverage_sample': cum_coverage,
        'midi_type_counts': dict(midi_type_counts),
        'genre_guess_counts': dict(genre_guess_counts),
    }

    report = {
        'summary_text': summary_text,
        'summary_dict': summary_dict,
        'global': {
            'pitch_counter': dict(global_pitch_counter),
            'patch_counter': dict(global_patch_counter),
            'pp_counter': {f"{p},{patch}": cnt for (p, patch), cnt in global_pp_counter.items()},
            'family_counter': dict(global_family_counter),
            'pitchclass_counter': pitchclass_distribution,
            'octave_counter': octave_distribution
        }
    }

    if verbose:
        print(summary_text)

    return report

###################################################################################
# This is the end of the Plots and Reports Python Module
###################################################################################