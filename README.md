# DNA Trace Explorer

A local interactive Python/Streamlit viewer for raw nanopore recordings exported as `.h5`/`.hdf5` files. Designed for comparing 1 kb DNA and λ-DNA through silicon nitride nanopores in LiCl, NaCl, KCl, RbCl and CsCl, and exporting figures for an ICEAN poster.

## Run on your computer

1. Install Python 3.11 or 3.12 if needed.
2. Download and unzip this folder.
3. Open Terminal in the `DNA_Trace_Explorer` folder.
4. Set up a virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   streamlit run app.py
   ```

   **Windows:** activate with `.venv\\Scripts\\activate` and use `python` instead of `python3`.

5. The browser app opens automatically. Put your H5 files in the `data` folder next to `app.py`, or enter the full path to an existing folder of H5 files. **Do not upload 200–300 MB files if they are already available locally.**

Suggested filenames (automatically parsed):

- `CsCl_400mV_1kb.h5`
- `CsCl_400mV_lambda.h5`
- `LiCl_400mV_1kb.h5`
- `LiCl_400mV_lambda.h5`

The app also supports manually correcting labels or sampling rates in the file metadata editor. Filenames and H5 files are never modified by this.

## Features

- Load many H5 recordings from local storage, or via drag-and-drop uploads.
- Automatically detect salt, applied voltage, and DNA length from filenames.
- Inspect the H5 dataset and sampling-rate metadata.
- Select 50 s, 100 s, or custom-length windows; use a **different start time for each salt**.
- View coloured stacked traces or an overlay; interactively zoom and pan.
- Compare 1 kb and λ-DNA in aligned panels.
- Inspect individual milliseconds of raw current in the Event zoom tab.
- Export **vector PDF** or **300-dpi PNG** suitable for PowerPoint and print posters.
- Export a CSV audit trail containing files and selected time regions.

## Tested NanoSense-style file

Example `CsCl_400mV_1kb.h5` contains one numeric HDF5 dataset called `/selected_data` with 32,000,002 float64 samples. Root attributes include `sampling_rate = 200000.0`, `duration = 260.0`, `original_file = '400.abf'`.

**Important:** 32,000,002 samples at 200 kHz represent **160.00001 seconds**, not 260 s. The file metadata's `duration` is the original recording duration, not the length of `selected_data`. Selection offset within the original ABF is not stored.

**Current units:** The original H5 provides numeric signal values but **does not declare units**. The default y-axis text is `Current (nA)` because the example has values around 14, which is plausible for this experiment. Verify against NanoSense/ABF before citing nA; it is editable in the left sidebar.

## Plotting and sampling accuracy

A 100-second window at 200 kHz contains **20 million samples**. Plotting each raw point would be too slow. Instead, the app reads selected windows in bounded batches and calculates the mean, minimum and maximum of each display bin. Min/max envelopes preserve the existence of brief excursions that mean-only decimation might hide, but they do **not** preserve exact event shape or dwell time at 50–100-second scales.

Use **Event zoom** to examine 1–100 ms regions at high resolution. The app does not automatically find events, measure dwell times, or assign DNA conformations. For robust dwell-time/topology analysis, use the event-extraction workflow on the raw data, not on the decimated overview plots.

## Privacy

When launched locally, all traces stay on your computer. This app does not upload data to a separate website. The optional Streamlit file uploader does keep temporary copies inside the app's local environment for the duration of the session.

## Troubleshooting

- If no files appear, confirm the H5 folder path and suffix `.h5` / `.hdf5`.
- If salt, DNA or voltage show as 'Unknown', adjust the labels in the metadata editor.
- If signal time is wrong, check the `sampling_rate_hz` column, which should be in samples per second.
- If the long-range plots look like thick bands, reduce window length and/or increase the bin count. Noise and rare spikes legitimately contribute to min-max ranges.

## Important update — v3: exact raw verification

**Why this matters:** v2's "raw" long-window mode *skipped samples*. This is not digital low-pass filtering, but it can omit brief nanopore blockades and cannot be used to confirm that a plot is a faithful view of every measured sample.

**Verify the exact file first:**

1. Load the folder containing the specific `.h5` file that produced your reference plot.
2. Open **Exact raw verification** and select that precise file path (not just the salt label).
3. Set start = `0 s`, duration = `20 s` and leave baseline centering and processing off (this view ignores the overview processing controls).
4. Compare the printed HDF5 dataset, sampling rate, raw min/median/max, and first five numeric signal samples to those obtained by a separate reader of the same file.
5. This mode uses *every sample*, no averaging, no sample skipping, no extra filter and no interpolation. It uses a normal Matplotlib line, similar to the reference screenshot. It is deliberately limited to 5 million samples per plot for memory/performance.

The 50–100 s multi-salt plot now defaults to **Min–max spikes (no averaging)**. It reads every sample within each time bin and draws a vertical mark spanning the bin's min and max. It retains the extrema but **does not reproduce exact within-bin event timing or shapes**. Use millisecond zoom or the exact raw view for event-level analysis.

If exact raw still differs, the discrepancy is not caused by the app's sample skipping. Check the exact HDF5 file and dataset, selected interval, units and any preprocessing by the program that generated the reference plot. A screenshot alone cannot establish that the source arrays are identical.
