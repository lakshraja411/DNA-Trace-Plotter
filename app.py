"""DNA Trace Explorer — raw HDF5 nanopore current viewer for ICEAN posters.

Launch locally:  streamlit run app.py
"""
from __future__ import annotations

import io
import math
import numpy as np
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from trace_utils import SALTS, SALT_COLORS, inspect_file, read_decimated, read_envelope, read_raw_window, read_exact_window

st.set_page_config(page_title="DNA Trace Explorer", page_icon="🧬", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.3rem; max-width: 1450px}
h1, h2, h3 {letter-spacing: -.02em}
[data-testid="stMetric"] {border:1px solid #ddd4; border-radius: 9px; padding: 10px}
</style>""", unsafe_allow_html=True)

PALETTES = {
    'Original': SALT_COLORS,
    'Matplotlib default-like': {
        'LiCl': '#1f77b4', 'NaCl': '#2ca02c', 'KCl': '#ffbf00', 'RbCl': '#ff7f0e', 'CsCl': '#d62728', 'Unknown': '#667085'
    },
    'Colorblind-friendly': {
        'LiCl': '#0072B2', 'NaCl': '#009E73', 'KCl': '#E69F00', 'RbCl': '#D55E00', 'CsCl': '#CC79A7', 'Unknown': '#7a7a7a'
    },
    'Cool tones': {
        'LiCl': '#3366CC', 'NaCl': '#109618', 'KCl': '#B82E2E', 'RbCl': '#994499', 'CsCl': '#0099C6', 'Unknown': '#7a7a7a'
    },
}

@st.cache_data(show_spinner=False, max_entries=200)
def cached_inspect(path: str, mtime_ns: int):
    return inspect_file(path)

@st.cache_data(show_spinner=False, max_entries=120)
def cached_envelope(path: str, ds: str, fs: float, start: float,
                    seconds: float, bins: int, mtime_ns: int):
    return read_envelope(path, ds, fs, start, seconds, bins)

@st.cache_data(show_spinner=False, max_entries=120)
def cached_decimated(path: str, ds: str, fs: float, start: float,
                     seconds: float, max_points: int, mtime_ns: int):
    return read_decimated(path, ds, fs, start, seconds, max_points)

@st.cache_data(show_spinner=False, max_entries=40)
def cached_zoom(path: str, ds: str, fs: float, start: float,
                millis: float, mtime_ns: int):
    return read_raw_window(path, ds, fs, start, millis)


def stamp(record: dict) -> int:
    return Path(record['path']).stat().st_mtime_ns


def valid_duration(rec: dict) -> float:
    fs = rec.get('sampling_rate_hz')
    return rec['n_samples'] / float(fs) if fs and fs > 0 else 0.0


def annotate_voltage(v):
    return f"{v:g} mV" if v is not None and pd.notna(v) else "Unknown voltage"


def filter_records(records: list[dict], dna: str, voltage: str) -> list[dict]:
    return [r for r in records if r['dna'] == dna and annotate_voltage(r.get('voltage_mV')) == voltage]


def choose_records(records: list[dict], prefix: str):
    selection = []
    for salt in SALTS:
        options = [r for r in records if r['salt'] == salt and valid_duration(r) > 0]
        if not options:
            continue
        if len(options) == 1:
            chosen = options[0]
            st.caption(f"{salt}: {chosen['name']}  ·  {valid_duration(chosen):.1f} s")
        else:
            names = [r['name'] for r in options]
            name = st.selectbox(f"{salt} recording", names, key=f"{prefix}_{salt}_select")
            chosen = next(r for r in options if r['name'] == name)
        selection.append(chosen)
    return selection


def select_starts(records: list[dict], window: float, prefix: str) -> list[tuple[dict, float]]:
    starts: list[tuple[dict, float]] = []
    if not records:
        return starts
    st.markdown('**Choose your region within each selected recording**')
    st.caption('Times begin at the start of `selected_data`, not necessarily at the start of the original ABF recording.')
    columns = st.columns(min(5, len(records)))
    for idx, rec in enumerate(records):
        available = valid_duration(rec)
        mx = max(0.0, available-window)
        with columns[idx % len(columns)]:
            val = st.number_input(
                f"{rec['salt']} start (s)", min_value=0.0,
                max_value=round(mx, 4), value=0.0, step=1.0,
                format='%.2f', key=f"{prefix}_{rec['salt']}_{rec['dna']}_{rec['name']}_start",
            )
            if available < window:
                st.warning(f"Only {available:.1f} s; this row will be shorter.")
            starts.append((rec, float(val)))
    return starts


def get_colors(palette_name: str, overrides: dict[str, str]) -> dict[str, str]:
    colors = dict(PALETTES[palette_name])
    colors.update(overrides)
    return colors


def get_trace_data(rec, start, seconds, plot_mode, point_budget):
    if plot_mode == 'Sample-skip preview (not event-faithful)':
        raw = cached_decimated(rec['path'], rec['dataset'], float(rec['sampling_rate_hz']), start, seconds, point_budget, stamp(rec))
        return {'kind': 'raw', **raw}
    env = cached_envelope(rec['path'], rec['dataset'], float(rec['sampling_rate_hz']), start, seconds, point_budget, stamp(rec))
    return {'kind': 'env', **env}


def collect_traces(picks, secs, point_budget, plot_mode):
    traces = []
    for rec, start in picks:
        nsec = min(secs, max(0., valid_duration(rec) - start))
        if nsec <= 0:
            continue
        data = get_trace_data(rec, start, nsec, plot_mode, point_budget)
        if data.get('t', []).size:
            traces.append((rec, start, data))
    return traces


def visual_data(data, baseline_center):
    if data['kind'] == 'raw':
        base = float(pd.Series(data['current']).median()) if baseline_center else 0.0
        return data['current'] - base
    base = float(pd.Series(data['mean']).median()) if baseline_center else 0.0
    return data['min'] - base, data['max'] - base, data['mean'] - base


def plotly_comparison(traces, unit: str, baseline_center: bool, plot_mode: str, overlay: bool, colors):
    nr = 1 if overlay else len(traces)
    subtitles = None if overlay else [f"{r['salt']}  ·  {r['name']}  ·  region starts {start:g}s" for r, start, _ in traces]
    fig = make_subplots(rows=nr, cols=1, shared_xaxes=not overlay,
                        vertical_spacing=(0.075 if nr > 1 else 0), subplot_titles=subtitles)
    for i, (r, start, data) in enumerate(traces):
        row = 1 if overlay else i + 1
        color = colors.get(r['salt'], colors.get('Unknown', '#667085'))
        if data['kind'] == 'raw':
            yy = visual_data(data, baseline_center)
            fig.add_trace(go.Scattergl(x=data['t'], y=yy, mode='lines', name=r['salt'], showlegend=overlay,
                                       line=dict(color=color, width=0.8),
                                       hovertemplate='t=%{x:.3f}s<br>I=%{y:.3f}<extra>' + r['salt'] + '</extra>'),
                          row=row, col=1)
        else:
            low, hi, mean = visual_data(data, baseline_center)
            if plot_mode == 'Min–max spikes (no averaging)':
                xx = np.column_stack((data['t'], data['t'], np.full(len(data['t']), np.nan))).reshape(-1)
                yy = np.column_stack((low, hi, np.full(len(low), np.nan))).reshape(-1)
                fig.add_trace(go.Scattergl(x=xx, y=yy, mode='lines', name=r['salt'], showlegend=overlay,
                                           line=dict(color=color, width=0.65), hoverinfo='skip'), row=row, col=1)
            else:
                if plot_mode == 'Envelope + mean':
                    fig.add_trace(go.Scatter(x=data['t'], y=low, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=row, col=1)
                    fig.add_trace(go.Scatter(x=data['t'], y=hi, mode='lines', fill='tonexty',
                                             fillcolor='rgba(120,130,150,0.15)' if overlay else 'rgba(100,130,160,0.17)',
                                             line=dict(width=0), showlegend=False, hoverinfo='skip'), row=row, col=1)
                fig.add_trace(go.Scatter(x=data['t'], y=mean, mode='lines', name=r['salt'], showlegend=overlay,
                                         line=dict(color=color, width=1.0),
                                         hovertemplate='t=%{x:.3f}s<br>I=%{y:.3f}<extra>' + r['salt'] + '</extra>'),
                              row=row, col=1)
        if not overlay:
            fig.update_yaxes(title_text=f"Current ({unit})" if i == nr // 2 else "", row=row, col=1)
    fig.update_xaxes(title_text='Time since selected region start (s)', row=nr, col=1)
    if overlay:
        fig.update_yaxes(title_text=f'Current ({unit})', row=1, col=1)
    fig.update_layout(height=max(420, 220 * nr), margin=dict(l=65, r=15, t=54, b=54),
                      template='plotly_white', hovermode='x unified',
                      legend=dict(orientation='h', y=1.15, x=0))
    return fig


def mpl_comparison(traces, unit: str, baseline_center: bool, plot_mode: str,
                   overlay: bool, colors, show_title: bool = True):
    nr = 1 if overlay else len(traces)
    fig, axes = plt.subplots(nr, 1, figsize=(12, max(3.4, 2.0 * nr + 1.1)), sharex=not overlay, squeeze=False)
    axes = axes[:, 0]
    for i, (r, start, data) in enumerate(traces):
        ax = axes[0 if overlay else i]
        color = colors.get(r['salt'], colors.get('Unknown', '#667085'))
        if data['kind'] == 'raw':
            yy = visual_data(data, baseline_center)
            ax.plot(data['t'], yy, color=color, lw=.55, label=r['salt'])
        else:
            low, hi, mean = visual_data(data, baseline_center)
            if plot_mode == 'Min–max spikes (no averaging)':
                ax.vlines(data['t'], low, hi, color=color, lw=.45, alpha=.8, label=r['salt'])
            else:
                if plot_mode == 'Envelope + mean':
                    ax.fill_between(data['t'], low, hi, color=color, alpha=.14, lw=0)
                ax.plot(data['t'], mean, color=color, lw=.8, label=r['salt'])
        if not overlay:
            ax.text(.005, .95, f"{r['salt']}  |  region {start:g}–{start + (data['t'][-1] if data['t'].size else 0):.1f} s",
                    transform=ax.transAxes, va='top', fontsize=11, fontweight='bold', color=color)
        ax.grid(alpha=.13, lw=.6)
        ax.tick_params(labelsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    if overlay:
        axes[0].legend(ncol=min(5, len(traces)), frameon=False, loc='best')
    axes[len(axes)//2].set_ylabel(f'Current ({unit})', fontsize=11)
    axes[-1].set_xlabel('Time since selected region start (s)', fontsize=11)
    if show_title and traces:
        r = traces[0][0]
        fig.suptitle(f"DNA translocation current traces  |  {r['dna']}  |  {annotate_voltage(r.get('voltage_mV'))}",
                     fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, .96] if show_title else None)
    return fig


def download_figure(fig, basename: str, key: str):
    c1, c2 = st.columns(2)
    pdf = io.BytesIO()
    fig.savefig(pdf, format='pdf', bbox_inches='tight', facecolor='white')
    c1.download_button('📄 Download vector PDF', data=pdf.getvalue(), file_name=f'{basename}.pdf',
                       mime='application/pdf', key=f'{key}_pdf')
    png = io.BytesIO()
    fig.savefig(png, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    c2.download_button('🖼️ Download 300-dpi PNG', data=png.getvalue(), file_name=f'{basename}.png',
                       mime='image/png', key=f'{key}_png')
    plt.close(fig)


st.title('🧬 DNA Trace Explorer')
st.caption('ICEAN poster tool · HDF5 raw-current visualization · LiCl, NaCl, KCl, RbCl, CsCl · λ-DNA and 1 kb')

with st.sidebar:
    st.header('1 · Load HDF5 recordings')
    st.info('For large 200 kHz files, run this app locally and use a folder path. That avoids uploading gigabytes.')
    source = st.radio('Where are the files?', ['Local folder', 'Upload files'], horizontal=True)
    all_paths: list[str] = []
    if source == 'Local folder':
        folder = st.text_input('Folder containing .h5 files', value=str((Path(__file__).parent / 'data').resolve()))
        d = Path(folder).expanduser()
        if d.is_dir():
            all_paths = [str(p.resolve()) for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in ('.h5', '.hdf5')]
            st.caption(f'Found {len(all_paths)} HDF5 files')
        else:
            st.warning('Folder does not exist. Create it and place your HDF5 files there.')
    else:
        uploads = st.file_uploader('Upload .h5 / .hdf5', type=['h5', 'hdf5'], accept_multiple_files=True,
                                   help='Large recordings consume additional memory during upload.')
        persisted = st.session_state.setdefault('upload_cache', {})
        upload_dir = Path(tempfile.gettempdir()) / 'dna_trace_explorer_uploads'
        upload_dir.mkdir(exist_ok=True)
        for up in uploads or []:
            cachekey = f'{up.name}::{up.size}'
            if cachekey not in persisted or not Path(persisted[cachekey]).exists():
                name = Path(up.name).name
                target = upload_dir / f'{abs(hash(cachekey))}_{name}'
                with target.open('wb') as dest:
                    up.seek(0)
                    shutil.copyfileobj(up, dest, length=8 * 1024 * 1024)
                persisted[cachekey] = str(target)
            all_paths.append(persisted[cachekey])
    st.divider()
    st.header('2 · Plot preferences')
    unit = st.text_input('Current unit', value='nA')
    plot_mode = st.selectbox('Long-window plotting mode', ['Min–max spikes (no averaging)', 'Sample-skip preview (not event-faithful)', 'Envelope + mean', 'Mean only'], index=0,
                             help='Min–max spikes preserves all sample extrema in each display interval. Sample-skip preview can miss narrow events. Exact raw is in its own tab.')
    point_budget = st.slider('Target points/bins per trace', min_value=1000, max_value=200000, step=1000, value=8000,
                             help='In raw mode this is the maximum number of displayed points. In envelope/mean mode this is the number of bins.')
    center_baseline = st.checkbox('Baseline-center traces', value=False,
                                  help='Subtract the median level so shape is easier to compare.')
    st.divider()
    st.header('3 · Colour controls')
    def _apply_palette():
        palette = PALETTES[st.session_state['palette_name']]
        for salt in SALTS:
            st.session_state[f'clr_{salt}'] = palette[salt]
    palette_name = st.selectbox('Palette preset', list(PALETTES.keys()), index=1,
                                key='palette_name', on_change=_apply_palette)
    default_palette = PALETTES[palette_name]
    color_overrides = {}
    with st.expander('Custom colours for each salt', expanded=False):
        for salt in SALTS:
            color_overrides[salt] = st.color_picker(f'{salt} colour', value=default_palette[salt], key=f'clr_{salt}')
    active_colors = get_colors(palette_name, color_overrides)

metadata = []
errors = []
for p in all_paths:
    try:
        metadata.append(cached_inspect(p, Path(p).stat().st_mtime_ns))
    except Exception as e:
        errors.append(f'{Path(p).name}: {e}')

for error in errors:
    st.warning(error)
if not metadata:
    st.info('Add your .h5 recordings using the left sidebar to begin. Suggested names: `CsCl_400mV_1kb.h5` and `LiCl_400mV_lambda.h5`.')
    st.markdown('''**Supported NanoSense-style file:** numeric dataset `selected_data`, file attribute `sampling_rate` (samples/s). The app also looks for other numeric 1-D datasets if `selected_data` is absent.''')
    st.stop()

with st.expander('Inspect / correct labels automatically detected from filenames', expanded=False):
    st.caption('Edit salt, DNA identity, applied voltage or sampling rate here if filename recognition is wrong. Nothing is changed in your original HDF5 files.')
    df = pd.DataFrame(metadata)
    df['signal_duration_s'] = df['signal_duration_s'].round(3)
    display_cols = ['name', 'salt', 'dna', 'voltage_mV', 'sampling_rate_hz', 'signal_duration_s', 'metadata_duration_s', 'dataset', 'size_mb']
    edited = st.data_editor(df[display_cols], hide_index=True, use_container_width=True,
                            disabled=['name', 'signal_duration_s', 'metadata_duration_s', 'dataset', 'size_mb'],
                            column_config={
                                'salt': st.column_config.SelectboxColumn('Electrolyte', options=[*SALTS, 'Unknown']),
                                'dna': st.column_config.SelectboxColumn('DNA', options=['1 kb', 'λ-DNA', 'Unknown']),
                                'voltage_mV': st.column_config.NumberColumn('Voltage (mV)', format='%.1f'),
                                'sampling_rate_hz': st.column_config.NumberColumn('Sampling (Hz)', format='%.0f'),
                                'signal_duration_s': st.column_config.NumberColumn('Selected data (s)', format='%.3f'),
                                'metadata_duration_s': st.column_config.NumberColumn('Original metadata duration (s)', format='%.1f'),
                                'size_mb': st.column_config.NumberColumn('Size (MiB)', format='%.1f')},
                            key='metadata_editor')
    for i, rec in enumerate(metadata):
        for k in ('salt', 'dna', 'voltage_mV', 'sampling_rate_hz'):
            v = edited.iloc[i][k]
            rec[k] = None if pd.isna(v) else v

valid = [r for r in metadata if valid_duration(r) > 0]
if len(valid) < len(metadata):
    st.warning('Some files have no valid sampling rate. Set it in the metadata editor above to include them.')
if not valid:
    st.stop()

n1, n2, n3 = st.columns(3)
n1.metric('Usable files', len(valid))
n2.metric('Distinct salts', len({r['salt'] for r in valid if r['salt'] != 'Unknown'}))
n3.metric('DNA types', ', '.join(sorted({r['dna'] for r in valid}))[:45])

tab1, tab2, tab3, tab4, tab_exact = st.tabs(['🎨 Five-salt comparison', '🧬 λ-DNA vs 1 kb', '🔎 Event zoom (milliseconds)', '📁 File inspector', '🧪 Exact raw verification'])

with tab1:
    st.subheader('Comparative raw-current panels')
    c1, c2, c3 = st.columns([1, 1, 1])
    dna_opts = [s for s in ('λ-DNA', '1 kb', 'Unknown') if any(r['dna'] == s for r in valid)]
    with c1:
        dna = st.selectbox('DNA type', dna_opts, key='compare_dna')
    available_v = sorted({annotate_voltage(r.get('voltage_mV')) for r in valid if r['dna'] == dna})
    with c2:
        voltage = st.selectbox('Applied voltage', available_v, key='compare_v')
    with c3:
        seconds = st.number_input('Window length (seconds)', min_value=0.01, max_value=3600., value=50., step=10., format='%.2f')
    view = st.radio('Layout', ['Stacked panels', 'Overlay'], horizontal=True,
                    help='Stacked panels are usually clearer for a conference poster.')
    selected = filter_records(valid, dna, voltage)
    picks = choose_records(selected, 'single')
    if not picks:
        st.warning('No labelled recording found for this selection. Correct filenames/labels in the inspector above.')
    else:
        missing = [s for s in SALTS if s not in {r['salt'] for r in picks}]
        if missing:
            st.info(f"Available now: {', '.join(r['salt'] for r in picks)}. To fill all five panels, add: {', '.join(missing)}.")
        starts = select_starts(picks, seconds, 'cmp')
        with st.spinner('Reading selected windows and building plots…'):
            traces = collect_traces(starts, seconds, point_budget, plot_mode)
        if traces:
            fig = plotly_comparison(traces, unit, center_baseline, plot_mode, view == 'Overlay', active_colors)
            st.plotly_chart(fig, use_container_width=True, config={'displaylogo': False, 'scrollZoom': True})
            if plot_mode == 'Sample-skip preview (not event-faithful)':
                st.warning('Sample-skip mode can omit short events or exaggerate the visual difference between recordings. Use min–max spikes for overviews, or Exact Raw for a one-to-one check.')
            elif plot_mode == 'Min–max spikes (no averaging)':
                st.caption('Min–max spikes preserve both extrema of every display time-bin with no smoothing; horizontal positions within each bin are approximate. This is a visualization, not raw event waveforms.')
            elif plot_mode == 'Envelope + mean':
                st.caption('Envelope mode preserves brief deep blockades that may disappear in ordinary downsampling.')
            expfig = mpl_comparison(traces, unit, center_baseline, plot_mode, view == 'Overlay', active_colors)
            download_figure(expfig, f'DNA_{dna.replace("λ", "lambda").replace(" ", "")}_{voltage.replace(" ", "")}_salts', 'single_compare')
            rows = []
            for r, start, data in traces:
                rows.append({'file': r['name'], 'salt': r['salt'], 'DNA': r['dna'],
                             'voltage_mV': r['voltage_mV'], 'window_start_s': start,
                             'window_duration_s': min(seconds, valid_duration(r) - start),
                             'dataset': r['dataset'], 'sampling_rate_hz': r['sampling_rate_hz'],
                             'plot_mode': plot_mode, 'current_unit_assumption': unit})
            st.download_button('⬇️ Export figure source/regions (CSV)', pd.DataFrame(rows).to_csv(index=False).encode(),
                               file_name='selected_regions.csv', mime='text/csv', key='cmp_csv')

with tab2:
    st.subheader('Side-by-side DNA length comparison')
    st.caption('Each row is a salt; left is 1 kb and right is λ-DNA, at matched voltage where recordings exist.')
    all_v = sorted({annotate_voltage(r.get('voltage_mV')) for r in valid})
    v2 = st.selectbox('Applied voltage for both DNA types', all_v, key='pair_voltage')
    salts2 = st.multiselect('Salts to include', SALTS, default=list(SALTS), key='pair_salts')
    win2 = st.number_input('Window length for both DNA types (seconds)', min_value=.01, max_value=3600., value=50., step=10., key='pair_window')
    pair_rows = []
    with st.expander('Set individual starts for both DNA lengths', expanded=True):
        for salt in salts2:
            matches = []
            for dna_label in ('1 kb', 'λ-DNA'):
                candidates = [r for r in valid if r['salt'] == salt and r['dna'] == dna_label and annotate_voltage(r.get('voltage_mV')) == v2]
                if candidates:
                    matches.append((dna_label, candidates[0]))
            if not matches:
                continue
            s_cols = st.columns([1, 1, 1])
            with s_cols[0]:
                st.markdown(f'**{salt}**')
            picks_pair = {}
            for i, (dna_label, r) in enumerate(matches):
                with s_cols[1 if dna_label == '1 kb' else 2]:
                    maxstart = max(0.0, valid_duration(r) - win2)
                    start = st.number_input(f'{dna_label} start (s)  ·  {salt}', min_value=0.,
                                            max_value=round(maxstart, 4), value=0., step=1., format='%.2f',
                                            key=f'pair_{salt}_{dna_label}_start')
                    picks_pair[dna_label] = (r, start)
            pair_rows.append((salt, picks_pair))
    if not pair_rows:
        st.info('Add labelled 1 kb or λ-DNA HDF5 files for the selected voltage to compare them.')
    else:
        with st.spinner('Reading 1 kb and λ-DNA windows…'):
            fig2 = make_subplots(rows=len(pair_rows), cols=2,
                                 subplot_titles=[f'{salt} · {dn}' for salt, _ in pair_rows for dn in ('1 kb', 'λ-DNA')],
                                 shared_xaxes=True, vertical_spacing=max(0.02, 0.08 / len(pair_rows)), horizontal_spacing=.09)
            fig_m, axs = plt.subplots(len(pair_rows), 2, figsize=(13, max(3.5, 2.3 * len(pair_rows) + 1)), squeeze=False, sharex=True)
            for i, (salt, picks_pair) in enumerate(pair_rows):
                for col, dna_label in enumerate(('1 kb', 'λ-DNA')):
                    ax = axs[i, col]
                    if dna_label not in picks_pair:
                        ax.text(.5, .5, 'No file', transform=ax.transAxes, ha='center', va='center', color='gray')
                        fig2.add_annotation(x=.5, y=.5, text='No file', showarrow=False, row=i + 1, col=col + 1)
                        continue
                    rec, start = picks_pair[dna_label]
                    actual = min(win2, valid_duration(rec) - start)
                    data = get_trace_data(rec, start, actual, plot_mode, point_budget)
                    colr = active_colors[salt]
                    if data['kind'] == 'raw':
                        yy = visual_data(data, center_baseline)
                        fig2.add_trace(go.Scattergl(x=data['t'], y=yy, mode='lines', line=dict(color=colr, width=0.8), showlegend=False), row=i + 1, col=col + 1)
                        ax.plot(data['t'], yy, color=colr, lw=.55)
                    else:
                        low, high, mean = visual_data(data, center_baseline)
                        if plot_mode == 'Min–max spikes (no averaging)':
                            xx = np.column_stack((data['t'], data['t'], np.full(len(data['t']), np.nan))).reshape(-1)
                            yy = np.column_stack((low, high, np.full(len(low), np.nan))).reshape(-1)
                            fig2.add_trace(go.Scattergl(x=xx, y=yy, mode='lines', line=dict(color=colr, width=.65), showlegend=False, hoverinfo='skip'), row=i + 1, col=col + 1)
                            ax.vlines(data['t'], low, high, color=colr, lw=.45, alpha=.8)
                        else:
                            if plot_mode == 'Envelope + mean':
                                fig2.add_trace(go.Scatter(x=data['t'], y=low, line=dict(width=0), hoverinfo='skip', showlegend=False), row=i + 1, col=col + 1)
                                fig2.add_trace(go.Scatter(x=data['t'], y=high, fill='tonexty', fillcolor='rgba(120,130,150,.15)', line=dict(width=0), hoverinfo='skip', showlegend=False), row=i + 1, col=col + 1)
                                ax.fill_between(data['t'], low, high, color=colr, alpha=.14, lw=0)
                            fig2.add_trace(go.Scatter(x=data['t'], y=mean, mode='lines', line=dict(color=colr, width=1), showlegend=False), row=i + 1, col=col + 1)
                            ax.plot(data['t'], mean, color=colr, lw=.8)
                    ax.text(.015, .94, f'{salt} · {dna_label}', transform=ax.transAxes, va='top', color=colr, fontsize=11, fontweight='bold')
                    ax.spines[['right', 'top']].set_visible(False)
                    ax.grid(alpha=.12)
                    if col == 0:
                        ax.set_ylabel(f'Current ({unit})')
            for ax in axs[-1, :]:
                ax.set_xlabel('Time since region start (s)')
            fig_m.suptitle(f'DNA length comparison · {v2}', fontsize=16, fontweight='bold')
            fig_m.tight_layout(rect=[0, 0, 1, .96])
            fig2.update_layout(height=max(450, 220 * len(pair_rows)), template='plotly_white', showlegend=False,
                               margin=dict(l=45, r=15, t=50, b=35))
            st.plotly_chart(fig2, use_container_width=True, config={'displaylogo': False, 'scrollZoom': True})
            download_figure(fig_m, f'DNA_length_comparison_{v2.replace(" ", "")}', 'len_compare')

with tab3:
    st.subheader('High-resolution event inspection')
    st.caption('Use this for individual millisecond-scale translocation events. The 50–100 s overview is too compressed for reliable dwell-time measurements.')
    by_name = {f"{r['name']} · {r['dataset']}": r for r in valid}
    s = st.selectbox('Recording', list(by_name), key='zoom_rec')
    rr = by_name[s]
    z1, z2 = st.columns(2)
    with z1:
        zstart = st.number_input('Start time from selected signal (s)', min_value=0.,
                                 max_value=round(valid_duration(rr), 5), value=0.,
                                 step=0.01, format='%.5f', key='zoom_start')
    with z2:
        zms = st.number_input('Zoom width (ms)', min_value=.1, max_value=5000., value=50.,
                              step=10., format='%.2f', key='zoom_ms')
    desired_samples = math.ceil(zms * 1e-3 * float(rr['sampling_rate_hz']))
    if desired_samples <= 30000:
        d = cached_zoom(rr['path'], rr['dataset'], float(rr['sampling_rate_hz']), zstart, zms, stamp(rr))
        figz = go.Figure(go.Scattergl(x=d['t_ms'], y=d['current'], mode='lines',
                                      line=dict(color=active_colors.get(rr['salt'], active_colors.get('Unknown', '#667085')), width=1)))
        tz = d['t_ms']; cy = d['current']
    else:
        de = cached_envelope(rr['path'], rr['dataset'], float(rr['sampling_rate_hz']), zstart, zms / 1000, 5000, stamp(rr))
        figz = go.Figure()
        figz.add_trace(go.Scatter(x=de['t'] * 1000, y=de['min'], line=dict(width=0), showlegend=False))
        figz.add_trace(go.Scatter(x=de['t'] * 1000, y=de['max'], fill='tonexty', fillcolor='rgba(90,120,160,.19)', line=dict(width=0), showlegend=False))
        figz.add_trace(go.Scatter(x=de['t'] * 1000, y=de['mean'], mode='lines', line=dict(color=active_colors.get(rr['salt'], active_colors.get('Unknown', '#667085')), width=1), showlegend=False))
        tz = de['t'] * 1000; cy = de['mean']
    figz.update_layout(template='plotly_white', height=460, xaxis_title='Time since zoom start (ms)',
                       yaxis_title=f'Current ({unit})', margin=dict(l=60, r=20, t=35, b=50),
                       title=f"{rr['name']} | selected-data time {zstart:g}s")
    st.plotly_chart(figz, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False})
    ff, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(tz, cy, color=active_colors.get(rr['salt'], active_colors.get('Unknown', '#667085')), lw=.8)
    ax.set(xlabel='Time since zoom start (ms)', ylabel=f'Current ({unit})')
    ax.spines[['top', 'right']].set_visible(False)
    ff.tight_layout()
    download_figure(ff, 'zoom_' + rr['name'].replace('.h5', '') + f'_{zstart:g}s', 'zoom_plot')

with tab4:
    st.subheader('Recording metadata and HDF5 structure')
    st.dataframe(pd.DataFrame([{
        'Filename': r['name'], 'Salt': r['salt'], 'DNA': r['dna'],
        'Voltage (mV)': r['voltage_mV'], 'Dataset': r['dataset'],
        'Samples': r['n_samples'], 'Sampling rate (Hz)': r['sampling_rate_hz'],
        'Selected duration (s)': valid_duration(r),
        'Original duration attribute (s)': r.get('metadata_duration_s'),
        'Size (MiB)': round(r['size_mb'], 1)
    } for r in metadata]), hide_index=True, use_container_width=True)
    mr = st.selectbox('Inspect file attributes', valid, format_func=lambda r: r['name'], key='inspect_rec')
    st.json(mr['file_attrs'])
    st.info('Dataset time zero is the beginning of `selected_data`. The original ABF timestamp or selection offset is not stored in your example file, so the viewer cannot reconstruct it.')
    st.caption('No event detection or topology inference is performed here. The tool only displays measured traces and calculates visualization summaries when needed.')

with tab_exact:
    st.subheader('Exact, unprocessed trace — verify against your reference')
    st.markdown('This view plots **every recorded sample** for the chosen window. No averaging, '
                'no stride-based skipping, no baseline subtraction and no digital filtering. '
                'For long comparisons, use the separate min–max overview mode.')
    exact_choice = st.selectbox('Choose the **exact** HDF5 file',
                                [r['path'] for r in valid], key='exact_file')
    er = next(r for r in valid if r['path'] == exact_choice)
    st.caption(f"Full path: `{er['path']}`  |  HDF5 dataset: `/{er['dataset']}`")
    st.caption(f"Sampling rate: {er['sampling_rate_hz']:g} samples/s  |  "
               f"Total selected-data duration: {valid_duration(er):.4f} s  |  "
               f"Original metadata duration: {er['metadata_duration_s']}")
    ec1, ec2, ec3 = st.columns([1, 1, 1])
    with ec1:
        start_exact = st.number_input('Exact trace start (s)', min_value=0.,
                                      max_value=max(0., valid_duration(er)-.0001),
                                      value=0., step=1., format='%.4f', key='exact_start')
    with ec2:
        window_exact = st.number_input('Exact trace window (s)', min_value=.01,
                                       max_value=25., value=20., step=1., key='exact_secs')
    with ec3:
        line_width = st.slider('Line width', min_value=.1, max_value=1.5,
                               value=.30, step=.05, key='exact_lw')
    show_grid = st.checkbox('Grid lines', value=False, key='exact_grid')
    y_manual = st.checkbox('Set vertical axis range manually', value=False, key='exact_y')
    if y_manual:
        yc1, yc2 = st.columns(2)
        with yc1: ymin = st.number_input('Y minimum', value=2., key='ymin')
        with yc2: ymax = st.number_input('Y maximum', value=9., key='ymax')
    total_exact = min(window_exact, max(0., valid_duration(er)-start_exact))
    if total_exact > 0:
        if math.ceil(total_exact * float(er['sampling_rate_hz'])) > 5_000_000:
            st.warning('More than 5 million samples requested. Reduce the window or use min–max overview.')
        else:
            with st.spinner('Reading every sample (no sample skipping)…'):
                exact = read_exact_window(er['path'], er['dataset'], float(er['sampling_rate_hz']),
                                          start_exact, total_exact)
            xx = exact['t']; yy = exact['current']
            finite = yy[np.isfinite(yy)]
            if finite.size:
                st.caption(f"**Verified sample range:** [{exact['start_index']:,}, {exact['stop_index']:,})  ·  "
                           f"**Points plotted:** {exact['n_samples']:,}  ·  "
                           f"**Raw min / median / max:** {finite.min():.5g} / {np.median(finite):.5g} / {finite.max():.5g} {unit}")
                st.caption(f"First five stored values in interval: `{yy[:5].tolist()}`")
            # Deliberately use Matplotlib (not Plotly): millions of exact samples in a browser trace are expensive.
            fig_exact, ax_exact = plt.subplots(figsize=(15, 5.2), dpi=110)
            ax_exact.plot(xx, yy, color=active_colors.get(er['salt'], '#1f77b4'),
                          linewidth=line_width, rasterized=True)
            ax_exact.set(xlabel='Time (s)', ylabel=f'Current ({unit})',
                         title=er['name'])
            ax_exact.set_xlim(start_exact, start_exact+total_exact)
            if y_manual and ymin < ymax:
                ax_exact.set_ylim(ymin, ymax)
            if show_grid:
                ax_exact.grid(alpha=.2)
            fig_exact.tight_layout()
            exact_png = io.BytesIO()
            fig_exact.savefig(exact_png, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            st.image(exact_png.getvalue(), use_container_width=True)
            st.download_button('Download exact raw verification PNG (150 dpi)',
                               data=exact_png.getvalue(),
                               file_name='EXACT_RAW_'+Path(er['name']).stem+f'_{start_exact:g}s.png',
                               mime='image/png', key='exact_png_export')
            st.caption('Exact view can take several seconds because it plots millions of samples. Poster-ready PDF and 300-dpi PNG are available in the comparison tabs.')
            plt.close(fig_exact)
    st.warning('If this exact raw plot still differs from the reference screenshot, the cause is not display downsampling. Check file path, HDF5 dataset, selected region, and whether the reference tool applies any preprocessing. Upload that **same LiCl .h5 file** for a direct comparison.')

st.divider()
st.caption('DNA Trace Explorer · built for comparing nanopore ionic-current traces · Original .h5 files are read-only and never modified')
