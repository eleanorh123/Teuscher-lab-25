'Standardize audio clip processing for both models. ANN is now GRU-based'
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest(args):
    rows, seen = [], {}
    for split, folders in [('development', [args.train_real, args.train_fake]),
                           ('test', [args.test_real, args.test_fake])]:
        for label, folder in enumerate(folders):
            folder = Path(folder).resolve()
            files = sorted(folder.glob('*.wav'))
            if not files:
                raise ValueError(f'No top-level WAV files: {folder}')
            for path in files:
                sha = digest(path)
                if sha in seen:
                    raise ValueError(f'Exact duplicate: {path} and {seen[sha]}')
                seen[sha] = str(path)
                rows.append(dict(path=str(path), label=label, split=split, sha256=sha))
    development = [i for i, r in enumerate(rows) if r['split'] == 'development']
    train, val = train_test_split(development, test_size=args.validation_fraction,
        random_state=args.split_seed, stratify=[rows[i]['label'] for i in development])
    for ids, split in [(train, 'train'), (val, 'validation')]:
        for i in ids:
            rows[i]['split'] = split
    with Path(args.output).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'label', 'split', 'sha256'])
        writer.writeheader()
        writer.writerows(rows)
    print('Saved manifest. Exact byte duplicates checked; speaker/source overlap still needs review.')


def prepare(args):
    import librosa
    with Path(args.manifest).open() as handle:
        rows = list(csv.DictReader(handle))
    required_splits = {'train', 'validation'} if getattr(args, 'pilot', False) else {'train', 'validation', 'test'}
    if set(r['split'] for r in rows) != required_splits:
        raise ValueError(f'Manifest must contain exactly {required_splits}.')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    total = np.zeros(args.n_mfcc, dtype=np.float64)
    squared = total.copy()
    count = 0
    retained, excluded = [], []
    save_json(out / 'excluded_audio.json', excluded)
    for i, row in enumerate(rows):
        if digest(row['path']) != row['sha256']:
            raise ValueError(f"File changed: {row['path']}")
        audio, _ = librosa.load(row['path'], sr=args.sample_rate, mono=True,
                                duration=args.duration)
        if len(audio) == 0 or not np.isfinite(audio).all():
            reason = 'empty_audio' if len(audio) == 0 else 'non_finite_audio'
            excluded.append(dict(path=row['path'], label=int(row['label']),
                split=row['split'], sha256=row['sha256'], reason=reason,
                samples=int(audio.size),
                non_finite_samples=int(np.count_nonzero(~np.isfinite(audio)))))
            save_json(out / 'excluded_audio.json', excluded)
            print(f"Skipping {reason}: {row['path']}", flush=True)
            continue
        # Extract before batch padding: short clips retain their actual length.
        x = librosa.feature.mfcc(y=audio, sr=args.sample_rate, n_mfcc=args.n_mfcc,
            n_fft=args.n_fft, hop_length=args.hop_length, center=True).T.astype('float32')
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite MFCC: {row['path']}")
        row['feature'] = f'{i:07d}.npy'
        row['label'] = int(row['label'])
        row['frames'] = len(x)
        np.save(out / row['feature'], x)
        retained.append(row)
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            print(f'Prepared {i + 1}/{len(rows)} clips', flush=True)
        if row['split'] == 'train':
            total += x.astype('float64').sum(0)
            squared += np.square(x.astype('float64')).sum(0)
            count += len(x)

    def split_counts(items):
        return {split: {str(label): sum(r['split'] == split and int(r['label']) == label
                                        for r in items) for label in [0, 1]}
                for split in sorted(required_splits)}
    counts = split_counts(retained)
    report = dict(input_clips=len(rows), retained_clips=len(retained),
        excluded_clips=len(excluded), retained_counts=counts,
        excluded_counts=split_counts(excluded), labels={'0': 'real', '1': 'fake'},
        policy='Exclude empty or non-finite loaded audio; preserve original split assignments')
    save_json(out / 'preprocessing_report.json', report)
    with (out / 'retained_manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'label', 'split', 'sha256', 'feature', 'frames'])
        writer.writeheader()
        writer.writerows(retained)
    print(f'Preprocessing complete: {len(retained)} retained, {len(excluded)} excluded. Counts: {counts}', flush=True)
    if any(n == 0 for classes in counts.values() for n in classes.values()):
        raise ValueError('A split has lost an entire class; see preprocessing_report.json. No training should proceed.')
    mean = total / count
    scale = np.sqrt(np.maximum(squared / count - mean ** 2, 0))
    scale[scale < 1e-8] = 1
    metadata = dict(rows=retained, mean=mean.tolist(), scale=scale.tolist(),
        preprocessing_report=report,
        settings={k: getattr(args, k) for k in
                  ['sample_rate', 'duration', 'n_mfcc', 'n_fft', 'hop_length']},
        manifest_sha256=digest(args.manifest), split_seed_note='See saved manifest')
    save_json(out / 'metadata.json', metadata)


class Clips(Dataset):
    def __init__(self, cache, split):
        self.cache = Path(cache)
        self.meta = json.loads((self.cache / 'metadata.json').read_text())
        self.rows = [r for r in self.meta['rows'] if r['split'] == split]
        self.mean = np.array(self.meta['mean'], dtype='float32')
        self.scale = np.array(self.meta['scale'], dtype='float32')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = (np.load(self.cache / r['feature']) - self.mean) / self.scale
        return torch.from_numpy(x), r['label'], r['sha256']


def collate(batch):
    x, y, ids = zip(*batch)
    return (nn.utils.rnn.pad_sequence(x, batch_first=True),
            torch.tensor([len(v) for v in x]), torch.tensor(y), ids)


class TemporalANN(nn.Module):
    def __init__(self, features, hidden, beta=0.9):
        super().__init__()
        self.rnn = nn.GRU(features, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 2)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(),
            batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        return self.output(h[-1]), {}


class AudioSNN(nn.Module):
    def __init__(self, features, hidden, beta=0.9):
        super().__init__()
        import snntorch as snn
        from snntorch import surrogate
        self.fc1, self.fc2 = nn.Linear(features, hidden), nn.Linear(hidden, 2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(slope=25))
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(slope=25))

    def forward(self, x, lengths):
        m1, m2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        counts = x.new_zeros((len(x), 2))
        hidden_counts = x.new_zeros((len(x), self.fc1.out_features))
        for t in range(x.shape[1]):
            valid = (t < lengths.to(x.device)).unsqueeze(1)
            s1, m1 = self.lif1(self.fc1(x[:, t]), m1)
            s2, m2 = self.lif2(self.fc2(s1), m2)
            counts = counts + s2 * valid
            hidden_counts = hidden_counts + s1.detach() * valid
        return counts, {'hidden_spikes': hidden_counts.sum(1),
                        'silent_hidden': (hidden_counts == 0).float().mean(1),
                        'output_spikes': counts.detach().sum(1)}


def metrics(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    difference = fpr - fnr
    right = int(np.flatnonzero(difference >= 0)[0])
    if right == 0:
        eer = float(fpr[0])
    else:
        left = right - 1
        weight = -difference[left] / (difference[right] - difference[left])
        eer = float(fpr[left] + weight * (fpr[right] - fpr[left]))
    predicted = (np.asarray(scores) > 0.5).astype(int)  # ties choose class 0
    return dict(accuracy=float(accuracy_score(labels, predicted)), eer=eer,
        confusion_matrix=confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        predicted_counts=np.bincount(predicted, minlength=2).tolist())


def epoch(model, loader, device, optimizer=None, shuffle_seed=None):
    model.train(optimizer is not None)
    labels, scores, records, activity = [], [], [], {}
    loss_sum = 0
    rng = torch.Generator().manual_seed(shuffle_seed or 0)
    with torch.set_grad_enabled(optimizer is not None):
        for x, lengths, y, ids in loader:
            if shuffle_seed is not None:
                for i, length in enumerate(lengths.tolist()):
                    x[i, :length] = x[i, :length][torch.randperm(length, generator=rng)]
            x, y = x.to(device), y.to(device)
            logits, stats = model(x, lengths)
            loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss')
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            probs = logits.detach().softmax(1)[:, 1].cpu().tolist()
            truth = y.cpu().tolist()
            labels.extend(truth)
            scores.extend(probs)
            records.extend(dict(id=i, label=l, score=s) for i, l, s in zip(ids, truth, probs))
            loss_sum += loss.item() * len(y)
            for key, value in stats.items():
                activity[key] = activity.get(key, 0) + value.sum().item()
    result = metrics(labels, scores)
    result['loss'] = loss_sum / len(labels)
    result.update({f'mean_{k}_per_clip': v / len(labels) for k, v in activity.items()})
    return result, records


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    meta_path = Path(args.cache) / 'metadata.json'
    config['cache_metadata_sha256'] = digest(meta_path)
    config['versions'] = {p: importlib.metadata.version(p) for p in
                          ['numpy', 'torch', 'librosa', 'scikit-learn', 'snntorch']}
    model = (AudioSNN if args.model == 'snn' else TemporalANN)(
        json.loads(meta_path.read_text())['settings']['n_mfcc'], args.hidden, args.beta).to(device)
    config['parameter_count'] = sum(p.numel() for p in model.parameters())
    save_json(out / 'config.json', config)
    def loader(split, training=False):
        return DataLoader(Clips(args.cache, split), batch_size=args.batch_size,
            shuffle=training, collate_fn=collate,
            generator=torch.Generator().manual_seed(args.seed))
    if args.command == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        train_loader, val_loader = loader('train', True), loader('validation')
        best, history = float('inf'), []
        for number in range(1, args.epochs + 1):
            start = time.perf_counter()
            train, _ = epoch(model, train_loader, device, optimizer)
            validation, _ = epoch(model, val_loader, device)
            history.append(dict(epoch=number, train=train, validation=validation,
                                seconds=time.perf_counter() - start))
            save_json(out / 'history.json', history)
            if validation['eer'] < best:
                best = validation['eer']
                torch.save(dict(state=model.state_dict(), config=config, epoch=number),
                           out / 'best.pt')
            print(json.dumps(history[-1]), flush=True)
    else:
        saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
        for key in ['model', 'hidden', 'beta', 'cache_metadata_sha256']:
            if saved['config'][key] != config[key]:
                raise ValueError(f'Checkpoint mismatch: {key}')
        model.load_state_dict(saved['state'])
        result, records = epoch(model, loader('test'), device,
                                shuffle_seed=args.shuffle_seed)
        save_json(out / 'metrics.json', result)
        save_json(out / 'predictions.json', records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('manifest')
    for name in ['train-real', 'train-fake', 'test-real', 'test-fake', 'output']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--validation-fraction', type=float, default=0.15)
    p = commands.add_parser('prepare')
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--duration', type=float, required=True)
    for name, value in [('sample-rate', 16000), ('n-mfcc', 13), ('n-fft', 2048), ('hop-length', 512)]:
        p.add_argument('--' + name, type=int, default=value)
    for command in ['train', 'evaluate']:
        p = commands.add_parser(command)
        for name in ['cache', 'output']:
            p.add_argument('--' + name, required=True)
        p.add_argument('--model', choices=['snn', 'ann'], required=True)
        p.add_argument('--hidden', type=int, required=True)
        p.add_argument('--beta', type=float, default=0.9)
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--batch-size', type=int, default=32)
        p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
        p.add_argument('--threads', type=int, default=2)
        if command == 'train':
            p.add_argument('--epochs', type=int, required=True)
            p.add_argument('--learning-rate', type=float, default=0.001)
        else:
            p.add_argument('--checkpoint', required=True)
            p.add_argument('--shuffle-seed', type=int)
    args = parser.parse_args()
    if args.command == 'manifest':
        manifest(args)
    elif args.command == 'prepare':
        if args.duration <= 0:
            parser.error('duration must be positive')
        prepare(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
