"""
export_tiles_with_overlay.py
============================
Exportiert alle analysierten Muscle-Tiles als PNG-Dateien mit Nucleus-Overlay.

Liest die gespeicherten Dataset-Dicts (mit Predictions) aus:
    ./dataset_dicts/batch_{n}_with_preds/{wsi_name}/*.pickle

Speichert PNGs mit gezeichnetem Nucleus-Overlay in:
    ./tile_exports/batch_{n}/{wsi_name}/{tile_stem}_x{x}_y{y}_n{count}.png

Voraussetzungen:
    - wsi_processing_pipeline muss unter ../wsi_processing_pipeline/ liegen
    - detectron2 muss installiert sein
    - Die Predictions müssen vorher im Notebook erzeugt worden sein

Verwendung:
    python export_tiles_with_overlay.py [--batch 5] [--include-empty] [--output ./tile_exports]
"""

import argparse
import pickle
import sys
import os
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

# wsi_processing_pipeline einbinden (liegt eine Ebene über diesem Repo)
sys.path.append(str(Path(__file__).parent.parent / 'wsi_processing_pipeline'))
sys.path.append(str(Path(__file__).parent.parent / 'wsi_processing_pipeline' / 'tile_extraction'))

import detectron2
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

import matplotlib
matplotlib.use('Agg')  # non-interactive Backend – kein Display nötig


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def setup_metadata(dataset_name: str) -> detectron2.data.catalog.Metadata:
    """Registriert die Metadaten für den Visualizer (nur Klasse 'nucleus')."""
    try:
        meta = MetadataCatalog.get(dataset_name)
        meta.set(thing_classes=["nucleus"])
    except Exception:
        meta = MetadataCatalog.get(dataset_name)
    return meta


def draw_overlay(img_np: np.ndarray, instances, metadata) -> np.ndarray:
    """Zeichnet Nucleus-Masken und Bounding Boxes auf das Tile-Bild."""
    v = Visualizer(img_np, metadata=metadata, scale=1, instance_mode=ColorMode.IMAGE)
    return v.draw_instance_predictions(instances).get_image()


# ---------------------------------------------------------------------------
# Hauptlogik
# ---------------------------------------------------------------------------

def export_batch(batch_number: int,
                 preds_base_dir: Path,
                 output_base_dir: Path,
                 skip_empty_tiles: bool,
                 metadata):

    batch_preds_dir = preds_base_dir / f'batch_{batch_number}_with_preds'
    if not batch_preds_dir.exists():
        print(f'[FEHLER] Verzeichnis nicht gefunden: {batch_preds_dir}')
        sys.exit(1)

    batch_export_dir = output_base_dir / f'batch_{batch_number}'
    batch_export_dir.mkdir(parents=True, exist_ok=True)

    wsi_dirs = [p for p in batch_preds_dir.iterdir() if p.is_dir()]
    print(f'Batch {batch_number}: {len(wsi_dirs)} WSI(s) gefunden')
    print(f'Export-Ziel: {batch_export_dir}\n')

    total_exported = 0
    total_skipped  = 0
    total_errors   = 0

    for wsi_dir in tqdm(wsi_dirs, desc='WSIs', unit='wsi'):
        wsi_name = wsi_dir.name
        wsi_export_dir = batch_export_dir / wsi_name
        wsi_export_dir.mkdir(exist_ok=True)

        pickle_paths = sorted(wsi_dir.glob('*.pickle'))

        for pkl_path in tqdm(pickle_paths, desc=wsi_name, leave=False, unit='tile'):
            try:
                d = load_pickle(pkl_path)

                # Tile-Bild direkt aus der WSI lesen
                t = d['tile']
                img_pil = t.get_pil_image().convert('RGB')
                img_np  = np.array(img_pil)

                # Tile-Koordinaten (Level-0-Pixel) für Dateinamen
                x = int(t.rectangle.ul.x)
                y = int(t.rectangle.ul.y)

                # Predictions prüfen
                if 'prediction' not in d:
                    total_skipped += 1
                    continue

                instances = d['prediction']['instances'].to('cpu')
                n_nuclei  = len(instances)

                if skip_empty_tiles and n_nuclei == 0:
                    total_skipped += 1
                    continue

                # Overlay zeichnen und als PNG speichern
                img_overlay = draw_overlay(img_np, instances, metadata)
                out_path = wsi_export_dir / f'{pkl_path.stem}_x{x}_y{y}_n{n_nuclei}.png'
                PIL.Image.fromarray(img_overlay).save(str(out_path))
                total_exported += 1

            except Exception as e:
                tqdm.write(f'  [FEHLER] {pkl_path.name}: {e}')
                total_errors += 1

    print(f'\nFertig.')
    print(f'  Exportiert:   {total_exported}')
    print(f'  Übersprungen: {total_skipped}  (skip_empty_tiles={skip_empty_tiles})')
    print(f'  Fehler:       {total_errors}')
    print(f'  PNGs in:      {batch_export_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Exportiert Muscle-Tiles als PNG mit Nucleus-Overlay.'
    )
    parser.add_argument(
        '--batch', type=int, default=5, choices=[3, 4, 5],
        help='Batch-Nummer (3, 4 oder 5). Standard: 5'
    )
    parser.add_argument(
        '--preds-dir', type=Path, default=Path('./dataset_dicts'),
        help='Basis-Verzeichnis der Predictions. Standard: ./dataset_dicts'
    )
    parser.add_argument(
        '--output', type=Path, default=Path('./tile_exports'),
        help='Ausgabeverzeichnis für PNGs. Standard: ./tile_exports'
    )
    parser.add_argument(
        '--include-empty', action='store_true',
        help='Auch Tiles ohne erkannte Nuclei exportieren (nur Original-Bild mit leerem Overlay).'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Metadaten für Visualizer initialisieren
    dataset_name = f'batch_{args.batch}'
    metadata = setup_metadata(dataset_name)

    export_batch(
        batch_number=args.batch,
        preds_base_dir=args.preds_dir,
        output_base_dir=args.output,
        skip_empty_tiles=not args.include_empty,
        metadata=metadata,
    )
