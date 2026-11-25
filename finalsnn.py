import argparse
import os
import random

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import librosa
import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

#all clips will be 4 seconds (might wnat to change to be 2 seconds)
def extract_mfcc_sequence(audio_path, sample_rate=16_000, duration=4.0, n_mfcc=13, n_fft=2048, hop_length=512):
    #Return one fixed-size MFCC sequence with shape [frames, coefficients].
    target_samples = int(sample_rate * duration)
    audio, sr = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True,
        duration=duration,
    )

    if audio.size == 0:
        raise ValueError("audio file is empty")

    # Every batch item needs the same number of time steps. Short clips are
    # zero-padded; long clips were already limited by librosa's duration option.
    audio = librosa.util.fix_length(audio, size=target_samples)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # librosa returns [coefficients, frames], while the SNN expects time first.
    mfcc = mfcc.T.astype(np.float32)
    return np.nan_to_num(mfcc, nan=0.0, posinf=0.0, neginf=0.0)


def iter_audio_files(folder):
    for root, _, files in os.walk(folder):
        for filename in sorted(files):
            if filename.lower().endswith(AUDIO_EXTENSIONS):
                yield os.path.join(root, filename)


def load_audio_dataset(real_folder, fake_folder, **mfcc_kwargs):
    #Load clips as [samples, frames, coefficients] and binary labels.
    sequences = []
    labels = []
    skipped = 0

    for folder, label in ((real_folder, 0), (fake_folder, 1)):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Could not find folder: {folder}")

        for path in iter_audio_files(folder):
            try:
                sequences.append(extract_mfcc_sequence(path, **mfcc_kwargs))
                labels.append(label)
            except Exception as error:
                skipped += 1
                print(f"Skipping {os.path.basename(path)}: {error}")

    if not sequences:
        raise ValueError(f"No supported audio files found ({', '.join(AUDIO_EXTENSIONS)})")

    class_counts = np.bincount(labels, minlength=2)
    if np.any(class_counts < 2):
        raise ValueError(
            "Need at least two readable files from each class for a stratified split; "
            f"found real={class_counts[0]}, fake={class_counts[1]}."
        )

    features = np.stack(sequences).astype(np.float32)
    targets = np.asarray(labels, dtype=np.int64)
    print(f"Loaded {len(features)} clips with shape {features.shape}; skipped {skipped}.")
    return features, targets


def split_and_standardize(features, targets, test_size=0.2):
    #Split first, then standardize each MFCC coefficient without leakage.
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=SEED,
        stratify=targets,
    )

    scaler = StandardScaler()
    n_coefficients = train_x.shape[-1]
    scaler.fit(train_x.reshape(-1, n_coefficients))
    train_x = scaler.transform(train_x.reshape(-1, n_coefficients)).reshape(train_x.shape)
    test_x = scaler.transform(test_x.reshape(-1, n_coefficients)).reshape(test_x.shape)
    return train_x.astype(np.float32), test_x.astype(np.float32), train_y, test_y, scaler


def make_loader(features, targets, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(targets),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class AudioSNN(nn.Module):
    def __init__(self, n_mfcc=13, hidden_size=200, beta=0.95):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(n_mfcc, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, mfcc_sequence):
        #Process input shaped [batch, frames, MFCC coefficients].
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        output_spikes = []

        for frame in range(mfcc_sequence.size(1)):
            current1 = self.fc1(mfcc_sequence[:, frame, :])
            spikes1, mem1 = self.lif1(current1, mem1)
            current2 = self.fc2(spikes1)
            spikes2, mem2 = self.lif2(current2, mem2)
            output_spikes.append(spikes2)

        return torch.stack(output_spikes, dim=0)


def run_epoch(model, loader, device, loss_function, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            spikes = model(sequences)
            spike_counts = spikes.sum(dim=0)
            loss = loss_function(spike_counts, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (spike_counts.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_audio_snn(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, targets = load_audio_dataset(
        args.real_folder,
        args.fake_folder,
        sample_rate=args.sample_rate,
        duration=args.duration,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    train_x, test_x, train_y, test_y, scaler = split_and_standardize(
        features, targets, test_size=args.test_size
    )
    train_loader = make_loader(train_x, train_y, args.batch_size, shuffle=True)
    test_loader = make_loader(test_x, test_y, args.batch_size, shuffle=False)

    model = AudioSNN(args.n_mfcc, args.hidden_size, args.beta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    print(
        f"Training on {len(train_y)} clips, testing on {len(test_y)} clips, "
        f"using {features.shape[1]} SNN time steps on {device}."
    )
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, device, loss_function, optimizer
        )
        test_loss, test_accuracy = run_epoch(model, test_loader, device, loss_function)
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train loss={train_loss:.4f}, train accuracy={train_accuracy:.1%}, "
            f"test loss={test_loss:.4f}, test accuracy={test_accuracy:.1%}"
        )

    if args.save_path:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "audio_settings": {
                    "sample_rate": args.sample_rate,
                    "duration": args.duration,
                    "n_mfcc": args.n_mfcc,
                    "n_fft": args.n_fft,
                    "hop_length": args.hop_length,
                },
            },
            args.save_path,
        )
        print(f"Saved model and preprocessing settings to {args.save_path}")

    return model, scaler


def parse_args():
    parser = argparse.ArgumentParser(description="Train an SNN to classify real and fake audio.")
    parser.add_argument("--real-folder", required=True)
    parser.add_argument("--fake-folder", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--n-mfcc", type=int, default=13)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--save-path", default="audio_snn.pt")
    return parser.parse_args()


if __name__ == "__main__":
    train_audio_snn(parse_args())
