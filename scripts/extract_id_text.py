

import os
import json
import argparse
import joblib
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_json(filepath: str) -> dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_id_text_mapping(datapath: str, output_path: str = None) -> Dict[str, Dict[str, str]]:

    datapath = Path(datapath)

    if not datapath.exists():
        raise FileNotFoundError(f"Data file not found: {datapath}")


    logger.info(f"Loading data file: {datapath}")
    dataset_dict = joblib.load(datapath)
    logger.info(f"Data loaded, total  {len(dataset_dict)} samples")


    splits_file = datapath.parent / "splits.json"
    if not splits_file.exists():
        raise FileNotFoundError(f"splits.json not found: {splits_file}")

    logger.info(f"Loading split file: {splits_file}")
    splits = read_json(str(splits_file))


    result = {
        'train': {},
        'val': {},
        'test': {}
    }


    stats = {
        'train': 0,
        'val': 0,
        'test': 0,
        'missing_text': 0,
        'missing_in_splits': 0
    }


    for sample_id, sample_data in dataset_dict.items():

        if 'text' not in sample_data:
            logger.warning(f"Sample {sample_id} missing text field")
            stats['missing_text'] += 1
            continue

        text = sample_data['text']


        if sample_id in splits.get('train', []):
            result['train'][sample_id] = text
            stats['train'] += 1
        elif sample_id in splits.get('val', []):
            result['val'][sample_id] = text
            stats['val'] += 1
        elif sample_id in splits.get('test', []):
            result['test'][sample_id] = text
            stats['test'] += 1
        else:
            logger.warning(f"Sample {sample_id} not in any split")
            stats['missing_in_splits'] += 1


    logger.info("\n" + "="*60)
    logger.info("Extraction statistics:")
    logger.info(f"  train: {stats['train']} samples")
    logger.info(f"  val: {stats['val']} samples")
    logger.info(f"  test: {stats['test']} samples")
    if stats['missing_text'] > 0:
        logger.warning(f"  missing text: {stats['missing_text']} samples")
    if stats['missing_in_splits'] > 0:
        logger.warning(f"  not in any split: {stats['missing_in_splits']} samples")
    logger.info("="*60)


    if output_path is None:
        output_path = datapath.parent / "id_text_mapping.json"
    else:
        output_path = Path(output_path)

        if output_path.is_dir():
            output_path = output_path / "id_text_mapping.json"
        elif not output_path.suffix:

            if not output_path.exists():
                output_path = output_path / "id_text_mapping.json"
            else:
                output_path = output_path.with_suffix('.json')


    output_path.parent.mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Done. Results saved to: {output_path}")

    return result


def save_as_csv(result: Dict[str, Dict[str, str]], output_dir: Path):

    import csv

    output_dir.mkdir(exist_ok=True, parents=True)

    for split_name, id_text_dict in result.items():
        csv_path = output_dir / f"{split_name}_id_text.csv"

        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'text'])

            for sample_id, text in sorted(id_text_dict.items()):
                writer.writerow([sample_id, text])

        logger.info(f"  {split_name} split CSV saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract train/val/test ID-to-text mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic
  python scripts/extract_id_text.py --datapath data/motionfix-dataset/motionfix.pth.tar

  # Custom output path
  python scripts/extract_id_text.py --datapath data/motionfix-dataset/motionfix.pth.tar --output output/id_text.json

  # Also save CSV per split
  python scripts/extract_id_text.py --datapath data/motionfix-dataset/motionfix.pth.tar --save-csv
        """
    )
    parser.add_argument(
        '--datapath',
        type=str,
        required=True,
        help='path to .pth.tar data file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='output JSON path (default: next to data file as id_text_mapping.json)'
    )
    parser.add_argument(
        '--save-csv',
        action='store_true',
        help='Also save CSV files (one per split)'
    )
    parser.add_argument(
        '--csv-dir',
        type=str,
        default=None,
        help='Directory for CSV output (default: same as JSON parent directory)'
    )

    args = parser.parse_args()


    result = extract_id_text_mapping(args.datapath, args.output)


    if args.save_csv:
        if args.csv_dir:
            csv_dir = Path(args.csv_dir)
        else:
            csv_dir = Path(args.output).parent if args.output else Path(args.datapath).parent / "id_text_csv"

        logger.info(f"\nSaving CSV files to: {csv_dir}")
        save_as_csv(result, csv_dir)


if __name__ == "__main__":
    main()

