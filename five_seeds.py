import argparse
import ast
from collections import Counter
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

SEEDS = [101, 102, 103, 104, 105]
MODELS = {'ann': 20, 'snn': 128}
PACKAGES = ['torch', 'numpy', 'librosa', 'scikit-learn', 'snntorch']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_script(path):
    tree = ast.parse(Path(path).read_text())
    names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ValueError(f'Duplicate definitions in saved training script: {duplicates}')
    if not {'run', 'AudioSNN', 'TemporalANN'} <= set(names):
        raise ValueError('Unrecognized training script')


def summarize(output):
    rows = []
    for seed in SEEDS:
        for model in MODELS:
            marker = output / f'{model}_seed{seed}' / 'complete.json'
            if marker.exists():
                rows.append(read(marker))
    write(output / 'seed_results.json', rows)
    lines = ['VALIDATION RESULTS: checkpoint selected by minimum validation EER',
             'Standard deviation is sample SD across seeds (ddof=1).',
             'Model | Seed | Selected epoch | Accuracy | EER']
    for r in rows:
        lines.append(f"{r['model'].upper()} | {r['seed']} | {r['epoch']} | "
                     f"{r['accuracy']:.2%} | {r['eer']:.2%}")
    aggregate = {}
    for model in MODELS:
        chosen = [r for r in rows if r['model'] == model]
        if not chosen:
            continue
        result = {'completed_seeds': len(chosen)}
        for metric in ['accuracy', 'eer']:
            values = [r[metric] for r in chosen]
            result[metric + '_mean'] = statistics.mean(values)
            result[metric + '_sd'] = statistics.stdev(values) if len(values) > 1 else None
        aggregate[model] = result
        if len(chosen) > 1:
            lines.append(f"{model.upper()} ({len(chosen)}/5 seeds): "
                f"accuracy {result['accuracy_mean']:.2%} +/- {100*result['accuracy_sd']:.2f} pp; "
                f"EER {result['eer_mean']:.2%} +/- {100*result['eer_sd']:.2f} pp")
    write(output / 'aggregate.json', aggregate)
    (output / 'summary.txt').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(
        '/stash/tlab/ehall/results/ordered_mfcc/full_20261011_104648_551844'))
    parser.add_argument('--output', type=Path, default=Path(
        '/stash/tlab/ehall/results/ordered_mfcc/five_seeds_current_v1'))
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if read(source / 'complete.json').get('status') != 'complete':
        raise ValueError('Source full-data comparison is not complete')
    script = source / 'experiment.py'
    validate_script(script)
    cache = source / 'cache'
    meta = read(cache / 'metadata.json')
    if {r['split'] for r in meta['rows']} != {'train', 'validation'}:
        raise ValueError('Expected a development-only cache')
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    for model, hidden in MODELS.items():
        config = read(source / model / 'config.json')
        expected = dict(model=model, hidden=hidden, epochs=20, batch_size=32,
            learning_rate=0.001, beta=0.9, device='cpu', threads=2,
            cache_metadata_sha256=sha(cache / 'metadata.json'), versions=versions)
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f'Source configuration/environment mismatch: {model} {key}')
    for i, row in enumerate(meta['rows'], 1):
        if not (cache / row['feature']).is_file():
            raise FileNotFoundError(cache / row['feature'])
        if i % 10000 == 0:
            print(f'Checked {i}/{len(meta["rows"])} cached files', flush=True)
    spec = dict(source=str(source), seeds=SEEDS, models=MODELS, epochs=20,
        batch_size=32, learning_rate=0.001, beta=0.9, device='cpu', threads=2,
        script_sha256=sha(script), cache_metadata_sha256=sha(cache / 'metadata.json'),
        versions=versions, selection='minimum validation EER; earliest exact tie',
        scope='stability of current configurations, not a tuning sweep or held-out test')
    output.mkdir(parents=True, exist_ok=True)
    # Prevent accidentally starting a second concurrent copy of this wrapper.
    import fcntl
    lock = (output / 'runner.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('This five-seed wrapper is already running')
    protocol = output / 'protocol.json'
    if protocol.exists():
        if read(protocol) != spec:
            raise ValueError('Existing sweep has different settings; use a different output folder')
    else:
        write(protocol, spec)
    frozen = output / 'experiment.py'
    if frozen.exists() and sha(frozen) != spec['script_sha256']:
        raise ValueError('Saved sweep training script was modified')
    if not frozen.exists():
        shutil.copy2(script, frozen)
    print(f'RESULTS: {output}', flush=True)
    for seed in SEEDS:
        for model, hidden in MODELS.items():
            job = output / f'{model}_seed{seed}'
            job.mkdir(exist_ok=True)
            if (job / 'complete.json').exists():
                print(f'Skipping completed {model} seed {seed}', flush=True)
                continue
            attempt = job / ('attempt_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            attempt.mkdir()
            run = attempt / 'run'
            command = [sys.executable, '-u', str(frozen), 'train', '--cache', str(cache),
                '--output', str(run), '--model', model, '--hidden', str(hidden),
                '--epochs', '20', '--seed', str(seed), '--batch-size', '32',
                '--learning-rate', '0.001', '--beta', '0.9', '--device', 'cpu', '--threads', '2']
            write(attempt / 'command.json', command)
            print(f'START {model.upper()} seed {seed}', flush=True)
            with (attempt / 'raw.log').open('w') as raw, (attempt / 'summary.log').open('w') as summary:
                with subprocess.Popen(command, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True) as process:
                    for line in process.stdout:
                        raw.write(line)
                        raw.flush()
                        try:
                            row = json.loads(line)
                            msg = (f"{model.upper()} seed {seed} epoch {row['epoch']:02d}/20 | "
                                f"train {row['train']['accuracy']:.1%} | "
                                f"val {row['validation']['accuracy']:.1%} | "
                                f"EER {row['validation']['eer']:.1%} | {row['seconds']:.1f}s")
                        except (ValueError, TypeError, KeyError):
                            msg = line.rstrip()
                        print(msg, flush=True)
                        summary.write(msg + '\n')
                        summary.flush()
                    code = process.wait()
                if code:
                    raise RuntimeError(f'Run failed; see {attempt / "raw.log"}. Restarting retries this run.')
            history = read(run / 'history.json')
            if [r['epoch'] for r in history] != list(range(1, 21)) or not (run / 'best.pt').exists():
                raise RuntimeError('Incomplete run output; no completion recorded')
            best = min(history, key=lambda r: r['validation']['eer'])
            write(job / 'complete.json', dict(model=model, seed=seed, epoch=best['epoch'],
                accuracy=best['validation']['accuracy'], eer=best['validation']['eer'],
                checkpoint=str(run / 'best.pt'), run=str(run)))
            summarize(output)
    summarize(output)
    write(output / 'complete.json', dict(status='complete', runs=10))
    print((output / 'summary.txt').read_text(), flush=True)
    print('FIVE-SEED COMPARISON COMPLETE', flush=True)


if __name__ == '__main__':
    main()
