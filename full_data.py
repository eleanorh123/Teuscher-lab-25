"""Full-development-data comparison. Test folders are never opened."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('NUMBA_CACHE_DIR', str(Path.home() / '.cache/numba'))
import experiment as e
from sklearn.model_selection import train_test_split


def make_manifest(source, out):
    rows, seen, duplicates, conflicts = [], {}, [], []
    original_counts = {}
    for label, name in enumerate(['training_real', 'training_fake']):
        files = sorted((source / name).glob('*.wav'))
        if len(files) < 10:
            raise ValueError(f'Too few top-level WAV files in {source / name}')
        print(f'{name}: {len(files)} clips', flush=True)
        original_counts[name] = len(files)
        for i, path in enumerate(files, 1):
            sha = e.digest(path)
            row = dict(path=str(path.resolve()), label=label, sha256=sha)
            if sha in seen:
                first = seen[sha]
                record = dict(sha256=sha, retained_path=first['path'],
                    duplicate_path=row['path'], retained_label=first['label'],
                    duplicate_label=label)
                if first['label'] != label:
                    conflicts.append(record)
                else:
                    duplicates.append(record)
            else:
                seen[sha] = row
                rows.append(row)
            if i % 5000 == 0:
                print(f'Checked {i}/{len(files)} files in {name}', flush=True)
    # Report every decision before splitting; source files are never altered.
    e.save_json(out / 'duplicates_excluded.json', duplicates)
    e.save_json(out / 'label_conflicts.json', conflicts)
    e.save_json(out / 'deduplication.json', dict(
        policy='Keep first lexicographic path per SHA256 within class; abort on cross-class matches',
        original_counts=original_counts, same_label_copies_excluded=len(duplicates),
        conflicting_copies=len(conflicts), unique_hashes=len(rows),
        retained_class_counts={str(label): sum(r['label'] == label for r in rows)
                               for label in [0, 1]}, source_files_modified=False))
    if conflicts:
        raise ValueError('Identical files have conflicting labels. See label_conflicts.json; no training launched.')
    print(f'Excluded {len(duplicates)} same-label copies; retained {len(rows)} unique clips. Audio files unchanged.', flush=True)
    train, val = train_test_split(range(len(rows)), test_size=0.2, random_state=42,
                                 stratify=[r['label'] for r in rows])
    for indices, split in [(train, 'train'), (val, 'validation')]:
        for i in indices:
            rows[i]['split'] = split
    with (out / 'manifest.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'label', 'sha256', 'split'])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/stash/tlab/ehall/for_data'))
    parser.add_argument('--results-root', type=Path,
        default=Path('/stash/tlab/ehall/results/ordered_mfcc'))
    args = parser.parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.results_root).free
    if free < 2 * 1024**3:
        raise RuntimeError('Less than 2 GiB free. Resolve storage before this run.')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = args.results_root / ('full_' + stamp)
    out.mkdir(exist_ok=False)
    print(f'RESULTS: {out}', flush=True)
    script = Path(e.__file__).resolve()
    # The pilot must already have enabled development-only preparation.
    if "getattr(args, 'pilot', False)" not in script.read_text():
        raise RuntimeError('experiment.py lacks the pilot adjustment. Share this error.')
    shutil.copy2(script, out / 'experiment.py')
    shutil.copy2(__file__, out / 'full_data.py')
    with (out / 'packages.txt').open('w') as f:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
    start = time.perf_counter()
    rows = make_manifest(args.data_root, out)
    counts = {split: sum(r['split'] == split for r in rows)
              for split in ['train', 'validation']}
    e.save_json(out / 'protocol.json', dict(
        purpose='full-development-data learning curves; one seed, not final test results',
        counts=counts, split_seed=42, training_seed=42, validation_fraction=0.2,
        epochs=20, duration_cap=4.0, n_mfcc=13, sample_rate=16000,
        n_fft=2048, hop_length=512, batch_size=32, learning_rate=0.001,
        hidden={'ann': 20, 'snn': 128}, beta=0.9, threads=2, device='cpu',
        test_set_used=False, source_group_overlap='not verified',
        duplicate_policy='One clip per SHA256; conflicting labels stop the run',
        padding='MFCC batch padding masked, no duration padding of short audio'))
    print(f'Split: {counts}. Extracting shared MFCCs once...', flush=True)
    e.prepare(SimpleNamespace(manifest=str(out / 'manifest.csv'),
        output=str(out / 'cache'), duration=4.0, sample_rate=16000, n_mfcc=13,
        n_fft=2048, hop_length=512, pilot=True))
    timings = {'preparation_seconds': time.perf_counter() - start}
    e.save_json(out / 'timings.json', timings)
    for model, hidden in [('ann', 20), ('snn', 128)]:
        print(f'\nStarting {model.upper()}: 20 epochs', flush=True)
        command = [sys.executable, '-u', str(out / 'experiment.py'), 'train',
            '--cache', str(out / 'cache'), '--output', str(out / model),
            '--model', model, '--hidden', str(hidden), '--epochs', '20',
            '--seed', '42', '--batch-size', '32', '--learning-rate', '0.001',
            '--beta', '0.9', '--device', 'cpu', '--threads', '2']
        start = time.perf_counter()
        with (out / f'{model}.log').open('w') as raw, (out / f'{model}_summary.log').open('w') as summary:
            with subprocess.Popen(command, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True) as process:
                for line in process.stdout:
                    raw.write(line)
                    raw.flush()
                    try:
                        row = json.loads(line)
                        message = (f"Epoch {row['epoch']:02d}/20 | "
                            f"Train accuracy {row['train']['accuracy']:.1%} | "
                            f"Validation accuracy {row['validation']['accuracy']:.1%} | "
                            f"Validation EER {row['validation']['eer']:.1%} | "
                            f"{row['seconds']:.1f}s")
                    except (ValueError, KeyError, TypeError):
                        message = line.rstrip()
                    print(message, flush=True)
                    summary.write(message + '\n')
                    summary.flush()
                code = process.wait()
            if code:
                raise RuntimeError(f'{model} failed. See {out / (model + ".log")}')
        timings[model + '_seconds'] = time.perf_counter() - start
        e.save_json(out / 'timings.json', timings)
    e.save_json(out / 'complete.json', dict(status='complete', epochs_per_model=20))
    print(f'FULL COMPARISON COMPLETE: {out}', flush=True)


if __name__ == '__main__':
    main()
