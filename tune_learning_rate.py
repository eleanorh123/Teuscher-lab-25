import argparse
import ast
from collections import Counter
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

RATES = [0.0003, 0.001, 0.003]
MODELS = {'ann': 20, 'snn': 128}
PACKAGES = ['torch', 'numpy', 'librosa', 'scikit-learn', 'snntorch']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def job_name(model, rate):
    return f'{model}_lr{str(rate).replace(".", "p")}'


def validate_script(path):
    tree = ast.parse(Path(path).read_text())
    names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    if any(n > 1 for n in Counter(names).values()):
        raise ValueError('Saved training script has duplicate definitions')
    if not {'run', 'AudioSNN', 'TemporalANN'} <= set(names):
        raise ValueError('Unrecognized training script')


def result_from_run(run, expected, reused=False):
    config = read(run / 'config.json')
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'{run}: configuration mismatch for {key}')
    history = read(run / 'history.json')
    if [r['epoch'] for r in history] != list(range(1, 21)):
        raise ValueError(f'{run}: not a complete 20-epoch run')
    for row in history:
        for metric in ['eer', 'accuracy']:
            value = row['validation'][metric]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid validation metric')
    checkpoint = run / 'best.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    best = min(history, key=lambda r: r['validation']['eer'])
    return dict(model=expected['model'], learning_rate=expected['learning_rate'], seed=42,
        selected_epoch=best['epoch'], accuracy=best['validation']['accuracy'],
        eer=best['validation']['eer'], checkpoint=str(checkpoint), reused=reused,
        run=str(run), config_sha256=sha(run / 'config.json'),
        history_sha256=sha(run / 'history.json'))


def summarize(out):
    rows = []
    for model in MODELS:
        for rate in RATES:
            marker = out / job_name(model, rate) / 'complete.json'
            if marker.exists():
                rows.append(read(marker))
    write(out / 'results.json', rows)
    lines = ['LEARNING-RATE TUNING: seed 42, validation only; not a multi-seed result.',
             'Model | Learning rate | Selected epoch | Val accuracy | Val EER | Reused']
    for row in rows:
        lines.append(f"{row['model'].upper()} | {row['learning_rate']} | {row['selected_epoch']} | "
            f"{row['accuracy']:.2%} | {row['eer']:.2%} | {row['reused']}")
    selected = {}
    for model in MODELS:
        candidates = [r for r in rows if r['model'] == model]
        if len(candidates) != len(RATES):
            lines.append(f'{model.upper()}: {len(candidates)}/3 candidates complete; no selection yet.')
            continue
        # Predeclared exact tie break: prefer existing LR, then the lower LR.
        best = min(candidates, key=lambda r: (r['eer'], r['learning_rate'] != 0.001, r['learning_rate']))
        selected[model] = dict(best, configuration_changed=best['learning_rate'] != 0.001)
        followup = ('Current five-seed runs remain applicable, subject to matching provenance.'
                    if best['learning_rate'] == 0.001 else
                    'Changed setting: repeat-seed follow-up needs review; not launched automatically.')
        lines.append(f"{model.upper()} selected LR {best['learning_rate']}: {followup}")
    write(out / 'selected.json', selected)
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path(
        '/stash/tlab/ehall/results/ordered_mfcc/full_20261011_104648_551844'))
    p.add_argument('--output', type=Path, default=Path(
        '/stash/tlab/ehall/results/ordered_mfcc/tuning_lr_v1'))
    p.add_argument('--check-only', action='store_true', help='Validate source and plan without starting training')
    args = p.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if read(source / 'complete.json').get('status') != 'complete':
        raise ValueError('Source full-data run is not complete')
    script, cache = source / 'experiment.py', source / 'cache'
    validate_script(script)
    meta = read(cache / 'metadata.json')
    if {r['split'] for r in meta['rows']} != {'train', 'validation'}:
        raise ValueError('Cache must contain training and validation only')
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    common = dict(command='train', epochs=20, seed=42, batch_size=32, beta=0.9,
        device='cpu', threads=2, cache_metadata_sha256=sha(cache / 'metadata.json'), versions=versions)
    baselines = {}
    for model, hidden in MODELS.items():
        baselines[model] = result_from_run(source / model,
            dict(common, model=model, hidden=hidden, learning_rate=0.001), reused=True)
    for row in meta['rows']:
        if not (cache / row['feature']).is_file():
            raise FileNotFoundError(cache / row['feature'])
    print('Source verified: 3 candidates per model, 2 reused runs, 4 new runs; no test evaluation.', flush=True)
    if args.check_only:
        print('CHECK ONLY: no training launched or results written.')
        return
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / 'runner.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('This tuning wrapper is already running')
    spec = dict(source=str(source), rates=RATES, hidden=MODELS, common=common,
        script_sha256=sha(script), baseline_provenance=baselines,
        selection='minimum validation EER; exact ties prefer LR .001 then lower LR',
        scope='limited single-seed learning-rate search; capacity and beta fixed')
    if (out / 'protocol.json').exists() and read(out / 'protocol.json') != spec:
        raise ValueError('Existing tuning folder has a different protocol; use a new output folder')
    write(out / 'protocol.json', spec)
    frozen = out / 'experiment.py'
    if frozen.exists() and sha(frozen) != sha(script):
        raise ValueError('Saved training script has changed')
    if not frozen.exists():
        shutil.copy2(script, frozen)
    for model in MODELS:
        job = out / job_name(model, 0.001)
        job.mkdir(exist_ok=True)
        write(job / 'complete.json', baselines[model])
    summarize(out)
    for rate in [0.0003, 0.003]:
        for model, hidden in MODELS.items():
            job = out / job_name(model, rate)
            job.mkdir(exist_ok=True)
            expected = dict(common, model=model, hidden=hidden, learning_rate=rate)
            if (job / 'complete.json').exists():
                old = read(job / 'complete.json')
                if result_from_run(Path(old['run']), expected) != old:
                    raise ValueError('Completed run changed since it was recorded')
                print(f'Skipping completed {model} LR {rate}', flush=True)
                continue
            attempt = job / ('attempt_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            attempt.mkdir()
            run = attempt / 'run'
            command = [sys.executable, '-u', str(frozen), 'train', '--cache', str(cache),
                '--output', str(run), '--model', model, '--hidden', str(hidden),
                '--learning-rate', str(rate), '--epochs', '20', '--seed', '42',
                '--batch-size', '32', '--beta', '0.9', '--device', 'cpu', '--threads', '2']
            write(attempt / 'command.json', command)
            print(f'START {model.upper()} LR {rate}: 20 epochs', flush=True)
            with (attempt / 'raw.log').open('w') as raw, (attempt / 'summary.log').open('w') as summary:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True) as process:
                    for line in process.stdout:
                        raw.write(line)
                        raw.flush()
                        try:
                            row = json.loads(line)
                            message = (f"{model.upper()} LR {rate} epoch {row['epoch']:02d}/20 | "
                                f"train {row['train']['accuracy']:.1%} | val {row['validation']['accuracy']:.1%} | "
                                f"EER {row['validation']['eer']:.1%} | {row['seconds']:.1f}s")
                        except (ValueError, KeyError, TypeError):
                            message = line.rstrip()
                        print(message, flush=True)
                        summary.write(message + '\n')
                        summary.flush()
                    code = process.wait()
                if code:
                    raise RuntimeError(f'Run failed: {attempt / "raw.log"}. Restart to retry this configuration.')
            write(job / 'complete.json', result_from_run(run, expected))
            summarize(out)
    write(out / 'complete.json', dict(status='complete', candidates=6, reused=2, new_runs=4))
    print((out / 'summary.txt').read_text(), flush=True)
    print(f'TUNING COMPLETE: {out}', flush=True)


if __name__ == '__main__':
    main()
