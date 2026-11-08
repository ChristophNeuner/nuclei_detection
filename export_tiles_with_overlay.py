"""
export_tiles_with_overlay.py
============================
Exportiert alle analysierten Muscle-Tiles als 3-Panel-PNG:
  Links:  Original-Tile (aus WSI per OpenSlide)
  Mitte:  Gemergte Binärmaske aller vorhergesagten Nuclei
  Rechts: Original-Tile mit farbigem Nucleus-Overlay

Parallelisierung: --workers N startet N Prozesse (Standard: Anzahl CPUs / 2).
Jeder Worker bearbeitet eine WSI komplett und öffnet die Slide-Datei nur einmal.
Bereits exportierte PNGs werden automatisch übersprungen (--overwrite zum Erzwingen).

Verwendung:
  python export_tiles_with_overlay.py
  python export_tiles_with_overlay.py --workers 6
  python export_tiles_with_overlay.py --datasets mice_gastrocnemius_wt batch_5
  python export_tiles_with_overlay.py --include-empty
  python export_tiles_with_overlay.py --overwrite
  python export_tiles_with_overlay.py --wsi-root J:/datasets/Muskeln --preds-dir J:/nuclei_detection/dataset_dicts --output J:/nuclei_detection/tile_exports
"""

import argparse
import os
import pathlib
import pickle
import random
import re
import sys
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import openslide
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1. PosixPath-Fix: Pickles wurden auf Linux erstellt
# ---------------------------------------------------------------------------
pathlib.PosixPath = pathlib.WindowsPath

# ---------------------------------------------------------------------------
# 2. wsi_processing_pipeline einbinden (für Notebook-Funktionsreferenzen)
# ---------------------------------------------------------------------------
_wsi_pipeline = Path('J:/wsi_processing_pipeline')
sys.path.insert(0, str(_wsi_pipeline.parent))   # für "import wsi_processing_pipeline"
sys.path.insert(0, str(_wsi_pipeline))
sys.path.insert(0, str(_wsi_pipeline / 'tile_extraction'))
sys.path.insert(0, str(_wsi_pipeline / 'shared'))

# ---------------------------------------------------------------------------
# 3. Minimaler detectron2-Stub damit pickle.load() funktioniert
# ---------------------------------------------------------------------------

class _Instances:
    def __init__(self, image_size, **kwargs):
        object.__setattr__(self, '_image_size', image_size)
        object.__setattr__(self, '_fields', {})
        for k, v in kwargs.items():
            self._fields[k] = v

    def __setattr__(self, name, val):
        self._fields[name] = val

    def __getattr__(self, name):
        fields = object.__getattribute__(self, '_fields')
        if name in fields:
            return fields[name]
        raise AttributeError(name)

    def __setstate__(self, state):
        object.__setattr__(self, '_image_size', state.get('_image_size', (512, 512)))
        object.__setattr__(self, '_fields', state.get('_fields', {}))

    def __len__(self):
        for v in self._fields.values():
            return len(v)
        return 0

    def has(self, name):
        return name in self._fields

    def get(self, name):
        return self._fields[name]

    def to(self, device):
        new = _Instances(self._image_size)
        for k, v in self._fields.items():
            object.__getattribute__(new, '_fields')[k] = (
                v.to(device) if hasattr(v, 'to') else v
            )
        return new


class _Boxes:
    def __init__(self, tensor):
        self.tensor = tensor

    def to(self, device):
        return _Boxes(self.tensor.to(device))

    def __len__(self):
        return len(self.tensor)


def _install_detectron2_stub():
    for path, attrs in [
        ('detectron2', {}),
        ('detectron2.structures', {'Instances': _Instances, 'Boxes': _Boxes}),
        ('detectron2.structures.instances', {'Instances': _Instances}),
        ('detectron2.structures.boxes', {'Boxes': _Boxes}),
        ('detectron2.structures.masks', {}),
        ('detectron2.structures.rotated_boxes', {}),
        ('detectron2.layers', {}),
        ('detectron2.layers.shape_spec', {}),
        ('detectron2.config', {}),
        ('detectron2.data', {}),
        ('detectron2.data.catalog', {}),
        ('detectron2.engine', {}),
        ('detectron2.modeling', {}),
        ('detectron2.utils', {}),
        ('detectron2.utils.logger', {}),
    ]:
        m = types.ModuleType(path)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[path] = m


_install_detectron2_stub()

# ---------------------------------------------------------------------------
# 4. Notebook-Funktionen die Pickles referenzieren (in __main__ definiert)
# ---------------------------------------------------------------------------

from tile_extraction.tiles import scoring_function_muscle   # noqa: F401
from tile_extraction.util import pil_to_np_rgb              # noqa: F401


def naming_func_muscles_mice(wsi_path):
    split = Path(wsi_path).stem.split('-')
    return f'{split[0]}'[:-1]


def naming_func_muscles(wsi_path):
    return Path(wsi_path).stem


def scoring_function_muscles_mice(tissue_percent, combined_factor):
    if combined_factor == 0:
        return tissue_percent / 2
    return (100 / combined_factor) * tissue_percent / 100


# ---------------------------------------------------------------------------
# 5. Alle Muscle-Datensätze aus Abschnitt 8.4
# ---------------------------------------------------------------------------

MUSCLE_DATASETS = {
    'mice_gastrocnemius_wt':  'with_preds_mice_desminopathy_gastrocnemius_wt',
    'mice_gastrocnemius_hom': 'with_preds_mice_desminopathy_gastrocnemius_hom',
    'mice_soleus_wt':         'with_preds_mice_desminopathy_soleus_wt',
    'mice_soleus_hom':        'with_preds_mice_desminopathy_soleus_hom',
    'mice_1':                 'with_preds_mice_desminopathy(1)',
    'mice_2':                 'with_preds_mice_desminopathy(2)',
    'mice_3':                 'with_preds_mice_desminopathy(3)',
    'human':                  'with_preds_human_desminopathy',
    'batch_2':                'batch_2_with_preds',
    'batch_3':                'batch_3_with_preds',
    'batch_4':                'batch_4_with_preds',
    'batch_5':                'batch_5_with_preds',
}

# ---------------------------------------------------------------------------
# 6. WSI-Index
# ---------------------------------------------------------------------------

WSI_EXTENSIONS = {'.ndpi', '.mrxs', '.svs', '.tiff', '.tif'}
_wsi_index: dict[str, Path] = {}


def build_wsi_index(wsi_root: Path):
    global _wsi_index
    _wsi_index = {}
    for p in wsi_root.rglob('*'):
        if p.suffix.lower() in WSI_EXTENSIONS:
            _wsi_index[p.stem] = p
    print(f'WSI-Index: {len(_wsi_index)} Dateien unter {wsi_root}')


def find_wsi(stem: str) -> Path | None:
    return _wsi_index.get(stem)


# ---------------------------------------------------------------------------
# 7. Tile-Koordinaten aus Dateiname parsen
# ---------------------------------------------------------------------------
_COORD_RE = re.compile(r'-x(\d+)-y(\d+)-w(\d+)-h(\d+)')


def parse_tile_coords(stem: str) -> tuple[int, int, int, int] | None:
    m = _COORD_RE.search(stem)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return None


# ---------------------------------------------------------------------------
# 8. Overlay zeichnen
# ---------------------------------------------------------------------------

def _instance_colors(n: int) -> np.ndarray:
    colors = []
    for i in range(n):
        h = (i / max(n, 1)) * 6
        s, v = 0.9, 0.95
        c = v * s
        x = c * (1 - abs(h % 2 - 1))
        m = v - c
        r, g, b = [(c,x,0),(x,c,0),(0,c,x),(0,x,c),(x,0,c),(c,0,x)][int(h) % 6]
        colors.append(((r+m)*255, (g+m)*255, (b+m)*255))
    return np.array(colors, dtype=np.uint8)


def merge_masks(masks) -> np.ndarray:
    merged = None
    for m in masks:
        m_np = m.numpy() if hasattr(m, 'numpy') else np.asarray(m)
        merged = m_np if merged is None else np.logical_or(merged, m_np)
    return (merged.astype(np.uint8) * 255) if merged is not None else None


def draw_overlay(img_np: np.ndarray, masks, alpha: float = 0.45) -> np.ndarray:
    import cv2
    result = img_np.astype(np.float32)
    colors = _instance_colors(len(masks))
    for mask, color in zip(masks, colors):
        m = (mask.numpy() if hasattr(mask, 'numpy') else np.asarray(mask)).astype(bool)
        for c in range(3):
            result[:, :, c] = np.where(m,
                result[:, :, c] * (1 - alpha) + color[c] * alpha,
                result[:, :, c])
        contours, _ = cv2.findContours(
            m.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color.tolist(), 1)
    return np.clip(result, 0, 255).astype(np.uint8)


def save_three_panel(img_np, mask_np, overlay_np, out_path: Path, title: str = ''):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.93, bottom=0.01)
    axes[0].imshow(img_np);              axes[0].set_title('Original', fontsize=9)
    axes[1].imshow(mask_np, cmap='gray'); axes[1].set_title('Masken (merged)', fontsize=9)
    axes[2].imshow(overlay_np);          axes[2].set_title('Overlay', fontsize=9)
    for ax in axes:
        ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=7, y=0.99)
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# 9. Worker-Funktion (top-level, damit multiprocessing sie picklieren kann)
# ---------------------------------------------------------------------------

def _process_wsi_dir(args):
    """Verarbeitet alle (oder eine Stichprobe der) Tiles einer WSI. Läuft in separatem Prozess."""
    wsi_dir_str, wsi_path_str, wsi_export_dir_str, skip_empty, overwrite, sample_ratio = args

    wsi_dir = Path(wsi_dir_str)
    wsi_export_dir = Path(wsi_export_dir_str)
    wsi_export_dir.mkdir(parents=True, exist_ok=True)

    # Slide einmal öffnen (statt pro Tile)
    slide = None
    if wsi_path_str:
        try:
            slide = openslide.open_slide(wsi_path_str)
        except Exception:
            pass

    exported = skipped = errors = 0

    all_pkls = sorted(wsi_dir.glob('*.pickle'))
    if sample_ratio < 1.0:
        k = max(1, round(len(all_pkls) * sample_ratio))
        random.seed(42)
        pkl_files = sorted(random.sample(all_pkls, min(k, len(all_pkls))))
    else:
        pkl_files = all_pkls

    for pkl_path in pkl_files:
        out_path = wsi_export_dir / f'{pkl_path.stem}.png'

        if not overwrite and out_path.exists():
            skipped += 1
            continue

        try:
            d = load_pickle(pkl_path)

            if 'prediction' not in d:
                skipped += 1
                continue

            instances = d['prediction']['instances'].to('cpu')
            n = len(instances)

            if skip_empty and n == 0:
                skipped += 1
                continue

            img_np = None
            coords = parse_tile_coords(pkl_path.stem)

            # Fallback: Koordinaten aus tile.rectangle (batch_2-5 Format)
            if coords is None and d.get('tile') and d['tile'].rectangle is not None:
                try:
                    r = d['tile'].rectangle
                    coords = (int(r.ul.x), int(r.ul.y), int(r.width()), int(r.height()))
                except Exception:
                    pass

            if slide is not None and coords is not None:
                x, y, w, h = coords
                level = 0
                if d.get('tile') and d['tile'].level is not None:
                    level = d['tile'].level
                img = slide.read_region((x, y), level, (w, h)).convert('RGB')
                img_np = np.array(img)
            elif d.get('tile') and d['tile'].tile_path and Path(d['tile'].tile_path).exists():
                img_np = np.array(Image.open(d['tile'].tile_path).convert('RGB'))

            if img_np is None:
                skipped += 1  # WSI nicht auf Platte oder Koordinaten fehlen
                continue

            h_px, w_px = img_np.shape[:2]

            if n > 0 and instances.has('pred_masks'):
                masks = instances.get('pred_masks')
                mask_np = merge_masks(masks)
                overlay_np = draw_overlay(img_np, masks)
            else:
                mask_np = np.zeros((h_px, w_px), dtype=np.uint8)
                overlay_np = img_np.copy()

            x_str = f'x{coords[0]}_y{coords[1]}' if coords else ''
            title = f'{wsi_dir.name}  |  {x_str}  |  {n} Nuclei'
            save_three_panel(img_np, mask_np, overlay_np, out_path, title)
            exported += 1

        except Exception as e:
            errors += 1

    if slide is not None:
        slide.close()

    return wsi_dir.name, exported, skipped, errors


# ---------------------------------------------------------------------------
# 10. Export-Logik
# ---------------------------------------------------------------------------

def export_dataset(dataset_key, preds_dir, output_base_dir, skip_empty,
                   overwrite=False, workers=1, sample_ratio=1.0):
    if not preds_dir.exists():
        print(f'  [ÜBERSPRUNGEN] nicht gefunden: {preds_dir}')
        return 0, 0, 0

    export_dir = output_base_dir / dataset_key
    export_dir.mkdir(parents=True, exist_ok=True)

    wsi_dirs = [p for p in preds_dir.iterdir() if p.is_dir()]
    sample_str = f'{round(sample_ratio*100)}%' if sample_ratio < 1.0 else '100%'
    print(f'\n{dataset_key}: {len(wsi_dirs)} WSI(s), {workers} Worker, Stichprobe: {sample_str}')

    work_items = []
    for wsi_dir in wsi_dirs:
        wsi_path = find_wsi(wsi_dir.name)
        wsi_export_dir = export_dir / wsi_dir.name
        work_items.append((
            str(wsi_dir),
            str(wsi_path) if wsi_path else '',
            str(wsi_export_dir),
            skip_empty,
            overwrite,
            sample_ratio,
        ))

    exported = skipped = errors = 0

    if workers <= 1:
        for item in tqdm(work_items, desc=dataset_key, unit='wsi'):
            name, e, s, err = _process_wsi_dir(item)
            exported += e; skipped += s; errors += err
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_wsi_dir, item): item for item in work_items}
            with tqdm(total=len(work_items), desc=dataset_key, unit='wsi') as pbar:
                for future in as_completed(futures):
                    try:
                        name, e, s, err = future.result()
                        exported += e; skipped += s; errors += err
                        pbar.set_postfix(exp=exported, skip=skipped, err=errors)
                    except Exception as exc:
                        errors += 1
                        tqdm.write(f'  [FEHLER] Worker: {exc}')
                    pbar.update(1)

    return exported, skipped, errors


# ---------------------------------------------------------------------------
# 11. CLI
# ---------------------------------------------------------------------------

def parse_args():
    default_workers = max(1, os.cpu_count() // 2)
    parser = argparse.ArgumentParser(
        description='Exportiert Muscle-Tiles als 3-Panel-PNG (Original | Maske | Overlay).'
    )
    parser.add_argument(
        '--datasets', nargs='+',
        choices=list(MUSCLE_DATASETS.keys()) + ['all'],
        default=['all'],
        help='Datensätze zum Exportieren. Standard: all'
    )
    parser.add_argument(
        '--wsi-root', type=Path, default=Path('J:/datasets/Muskeln'),
        help='Wurzelverzeichnis der WSI-Dateien.'
    )
    parser.add_argument(
        '--preds-dir', type=Path, default=Path('J:/nuclei_detection/dataset_dicts'),
    )
    parser.add_argument(
        '--output', type=Path, default=Path('J:/nuclei_detection/tile_exports'),
    )
    parser.add_argument(
        '--include-empty', action='store_true',
        help='Auch Tiles ohne Nuclei exportieren'
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Bereits exportierte PNGs überschreiben (Standard: überspringen)'
    )
    parser.add_argument(
        '--workers', type=int, default=default_workers,
        help=f'Anzahl paralleler Prozesse. Standard: {default_workers} (CPU-Kerne / 2)'
    )
    parser.add_argument(
        '--sample', type=float, default=1.0, metavar='RATIO',
        help='Anteil der Tiles pro WSI (z.B. 0.1 für 10%%). Standard: 1.0 (alle)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    build_wsi_index(args.wsi_root)

    selected = (
        list(MUSCLE_DATASETS.keys()) if 'all' in args.datasets else args.datasets
    )

    print(f'Exportiere {len(selected)} Datensatz/Datensätze mit {args.workers} Worker(n)')
    if not args.overwrite:
        print('(Bereits exportierte PNGs werden übersprungen — --overwrite zum Erzwingen)')

    total_exp = total_skp = total_err = 0
    for key in selected:
        preds_dir = args.preds_dir / MUSCLE_DATASETS[key]
        e, s, err = export_dataset(
            key, preds_dir, args.output,
            skip_empty=not args.include_empty,
            overwrite=args.overwrite,
            workers=args.workers,
            sample_ratio=args.sample,
        )
        total_exp += e
        total_skp += s
        total_err += err

    print(f'\n{"="*50}')
    print(f'Exportiert:   {total_exp}')
    print(f'Übersprungen: {total_skp}')
    print(f'Fehler:       {total_err}')
    print(f'PNGs in:      {args.output}')
