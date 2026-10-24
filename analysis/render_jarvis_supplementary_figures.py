"""Render separate SI figures, with every defined source-weight cell labelled."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
data_path = ROOT / 'source_data/JARVIS_source_diagnostics.json'
data = json.loads(data_path.read_text())
name_records = json.loads((ROOT / 'source_data/JARVIS_official_task_names.json').read_text())['tasks']
names = {x['internal_id']: x['display_name'] for x in name_records}
tasks = data['source_order']
rows = data['tasks']
assert tasks == [r['task'] for r in rows] and len(tasks) == 34
matrix = np.asarray([r['source_share_percent'] if r['source_share_percent'] is not None else [np.nan]*34 for r in rows])
coverage = np.array([r['coverage_percent'] for r in rows])
assert np.allclose(np.nansum(matrix, axis=1)[coverage > 0], 100, atol=1e-8)
assert np.allclose(coverage, [100*r['n_accepted']/r['n_test'] for r in rows])
out = ROOT / 'run_outputs' / 'figures'
out.mkdir(parents=True, exist_ok=True)
ink = '#20344b'
dark = '#193c67'
blue = '#74ace0'
classification = '#7d3c72'
labels = [f'{i+1:02}  {names[t]}' for i, t in enumerate(tasks)]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False})

def save(fig, stem):
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(out / f'{stem}.{ext}', dpi=240, facecolor='white')
        if ext == 'svg':
            path = out / f'{stem}.{ext}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)

# A separate portrait page gives the 34 task names and percentages ample space.
fig = plt.figure(figsize=(6.85, 8.15), facecolor='white')
ax = fig.add_axes([.365, .115, .525, .79])
y = np.arange(34)
ax.barh(y, 100-coverage, color=dark, height=.72)
ax.barh(y, coverage, left=100-coverage, color=blue, height=.72)
ax.set(xlim=(0, 100), ylim=(33.5, -.5))
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_yticks(y, labels, fontsize=9)
ax.tick_params(axis='y', length=0, pad=7)
ax.tick_params(axis='x', labelsize=9, pad=4)
for i, label in enumerate(ax.get_yticklabels()):
    label.set_color(classification if i >= 27 else ink)
ax.set_axisbelow(True)
ax.grid(axis='x', color='#dbe2e8', linewidth=.6)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color('#aab7c3')
ax.axhline(26.5, color=classification, lw=1.1)
ax.set_xlabel('Test samples (%)', fontsize=10, labelpad=7)
for i, value in enumerate(coverage):
    ax.text(1.035, i, f'{value:.1f}', transform=ax.get_yaxis_transform(),
            va='center', ha='left', fontsize=9, color=ink)
ax.text(1.035, 1.015, 'Accepted\n(%)', transform=ax.transAxes,
        ha='left', va='bottom', fontsize=8.5, color=ink)
fig.text(.015, .955, 'Correction acceptance across 34 JARVIS tasks',
         fontsize=12, fontweight='bold', color=ink)
fig.text(.015, .918, 'Target task', fontsize=9.5, fontweight='bold', color=ink)
fig.legend(handles=[Patch(facecolor=dark, label='Stage 1 anchor retained'),
                    Patch(facecolor=blue, label='Stage 2 correction applied')],
           loc='lower left', bbox_to_anchor=(.01, .018), ncol=2, frameon=False,
           fontsize=9, handlelength=1.2, columnspacing=1.5)
fig.text(.015, .073, '01-27: regression', fontsize=9, color=ink)
fig.text(.015, .050, '28-34: classification', fontsize=9, color=classification)
save(fig, 'jarvis_correction_acceptance')

# Full-width landscape figure: all 858 defined off-diagonal cells are annotated.
fig = plt.figure(figsize=(10.35, 5.85), facecolor='white')
hm = fig.add_axes([.218, .150, .775, .785])
cmap = plt.get_cmap('Blues').copy()
cmap.set_bad('#e9edf1')
display = matrix.copy()
np.fill_diagonal(display, np.nan)
im = hm.imshow(np.ma.masked_invalid(display), aspect='auto', vmin=0, vmax=25,
               cmap=cmap, interpolation='none')
hm.set_xlim(-.5, 33.5)
hm.set_ylim(33.5, -.5)
hm.set_xticks(np.arange(34), [f'{i+1:02}' for i in range(34)], fontsize=7.5)
hm.set_yticks(y, labels, fontsize=7.6)
hm.tick_params(axis='x', length=0, pad=3)
hm.tick_params(axis='y', length=0, pad=5)
for i, label in enumerate(hm.get_yticklabels()):
    label.set_color(classification if i >= 27 else ink)
for i, label in enumerate(hm.get_xticklabels()):
    label.set_color(classification if i >= 27 else ink)
hm.set_xticks(np.arange(-.5, 34, 1), minor=True)
hm.set_yticks(np.arange(-.5, 34, 1), minor=True)
hm.tick_params(which='minor', bottom=False, left=False)
hm.grid(which='minor', color='white', linewidth=.35)
for spine in hm.spines.values():
    spine.set_visible(False)
numeric_count = 0
na_count = 0
numeric_artists = []
for i in range(34):
    for j in range(34):
        if i == j:
            hm.text(j, i, '-', ha='center', va='center', fontsize=7.3, color='#647486')
        elif coverage[i] == 0:
            hm.text(j, i, 'N/A', ha='center', va='center', fontsize=6.3, color='#647486')
            na_count += 1
        else:
            numeric_artists.append(hm.text(j, i, f'{matrix[i,j]:.1f}', ha='center', va='center',
                                          fontsize=7.1, color='white' if matrix[i,j] >= 16 else dark))
            numeric_count += 1
hm.axhline(26.5, color=classification, lw=1.0, zorder=5)
hm.axvline(26.5, color=classification, lw=1.0, zorder=5)
hm.set_xlabel('Source expert ID (same task mapping as rows)', fontsize=8.5, labelpad=5)
fig.text(.012, .965, 'Non-target source weights within accepted corrections',
         fontsize=11, fontweight='bold', color=ink)
fig.text(.012, .938, 'Target task', fontsize=8, fontweight='bold', color=ink)
fig.text(.012, .053, '01-27: regression', color=ink, fontsize=8)
fig.text(.012, .025, '28-34: classification', color=classification, fontsize=8)
cax = fig.add_axes([.713, .044, .26, .015])
cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0, 5, 10, 15, 20, 25])
cb.ax.tick_params(labelsize=7.5, length=2, pad=2)
cb.outline.set_linewidth(.5)
fig.text(.695, .044, 'Diagnostic source weight (%)', ha='right', va='bottom', fontsize=8, color=ink)

# Check that the widest annotation fits within one cell at the actual PDF size.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
cell_width_px = hm.get_window_extent(renderer).width / 34
max_numeric_width_px = max(a.get_window_extent(renderer).width for a in numeric_artists)
assert max_numeric_width_px < cell_width_px * .96, (max_numeric_width_px, cell_width_px)
assert numeric_count == int((coverage > 0).sum())*33 == 858
assert na_count == int((coverage == 0).sum())*33 == 264
save(fig, 'jarvis_source_weights')

method_path = ROOT / 'source_data/JARVIS_acceptance_and_source_weights_method.json'
method = json.loads(method_path.read_text())
method.update({
    'active_figures': ['jarvis_correction_acceptance', 'jarvis_source_weights'],
    'placement': 'Diagnostic rendering of data used in main Fig. 4 and Supplementary Fig. S7',
    'annotations': 'Every defined non-target cell is labelled to one decimal place, including 0.0. Target-self cells are dashes. Each other cell in a zero-acceptance row is N/A. Rounded rows may differ from 100%. Exact values are in CSV.',
    'numeric_cell_count': numeric_count,
    'undefined_non_target_cell_count': na_count,
    'target_self_cell_count': 34,
    'figure_source_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
    'numeric_font_size_pt': 7.1,
    'max_numeric_width_fraction_of_cell': max_numeric_width_px/cell_width_px,
})
(out / 'jarvis_rendering_metadata.json').write_text(json.dumps(method, indent=2)+'\n')
print(json.dumps({k: method[k] for k in ['active_figures','numeric_cell_count','undefined_non_target_cell_count','target_self_cell_count','max_numeric_width_fraction_of_cell']}))
