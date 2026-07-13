import argparse
import os
import random

os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable


device = torch.device('cpu')


class NeuralNetwork(nn.Module):

    def __init__(self, inodes, hnodes, onodes, learning_rate):
        super().__init__()

        # dimensions
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes

        # learning rate
        self.lr = learning_rate

        # define the layers and their sizes, turn off bias to match Rashid's simple ANN
        self.linear_ih = nn.Linear(inodes, hnodes, bias=False)
        self.linear_ho = nn.Linear(hnodes, onodes, bias=False)

        # define activation function
        self.activation = nn.Sigmoid()

        # create error function
        self.error_function = torch.nn.MSELoss(reduction='sum')

        # create optimiser, using simple stochastic gradient descent
        self.optimiser = torch.optim.SGD(self.parameters(), self.lr)

    def forward(self, inputs_list):
        # convert list to a 2-D FloatTensor on same device as model
        inputs = Variable(torch.tensor(inputs_list, dtype=torch.float32, device=device).view(1, self.inodes))

        # combine input layer signals into hidden layer
        hidden_inputs = self.linear_ih(inputs)
        # apply sigmoid activation function
        hidden_outputs = self.activation(hidden_inputs)

        # combine hidden layer signals into output layer
        final_inputs = self.linear_ho(hidden_outputs)
        # apply sigmoid activation function
        final_outputs = self.activation(final_inputs)

        return final_outputs

    def train_record(self, inputs_list, targets_list):
        # calculate the output of the network
        output = self.forward(inputs_list)

        # create a Variable out of the target vector, doesn't need gradients calculated
        target_variable = Variable(
            torch.tensor(targets_list, dtype=torch.float32, device=device).view(1, self.onodes),
            requires_grad=False
        )

        # calculate error
        loss = self.error_function(output, target_variable)

        # zero gradients, perform a backward pass, and update the weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item()


#AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')


def extract_features(audio_path, sample_rate=16000, n_mfcc=13, min_samples=2048, min_mfcc_frames=2):
    """Turn one audio file into one fixed-length vector of MFCC statistics."""
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    if len(y) < min_samples:
        raise ValueError(f'too short: {len(y)} samples')

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < min_mfcc_frames:
        raise ValueError(f'MFCC shape too short: {mfcc.shape}')

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features = np.concatenate([mfcc_mean, mfcc_std])
    return np.nan_to_num(features)


def iter_audio_files(folder):
    for root, _, files in os.walk(folder):
        for filename in sorted(files):
            if filename.lower().endswith('.wav'):
                yield os.path.join(root, filename)


def load_audio_dataset(real_folder, fake_folder):
    """Load real and fake audio folders into X feature vectors and y labels."""
    X = []
    y = []
    skipped = 0

    folder_info = [
        (real_folder, 0),  # real
        (fake_folder, 1),  # fake / AI generated
    ]

    for folder, label in folder_info:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f'Could not find folder: {folder}')

        for path in iter_audio_files(folder):
            filename = os.path.basename(path)
            try:
                features = extract_features(path)
                X.append(features)
                y.append(label)
            except Exception as e:
                skipped += 1
                print(f'Skipping {filename}: {e}')

    if len(X) == 0:
        raise ValueError('No audio files found. Expected files ending in: ' + ', '.join('.wav'))

    print(f'Loaded {len(X)} audio files. Skipped {skipped}.')
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def make_targets(label, output_nodes):
    # all 0.01, except the desired label which is 0.99
    targets = np.zeros(output_nodes) + 0.01
    targets[label] = 0.99
    return targets


def evaluate_network(network, X_test, y_test):
    scorecard = []

    print('\nTesting neural network...')
    for inputs, correct_label in zip(X_test, y_test):
        outputs = network.forward(inputs)
        _, label = outputs.max(1)

        predicted_label = label.item()
        scorecard.append(1 if predicted_label == correct_label else 0)
        print(f'correct label: {correct_label}, network answer: {predicted_label}')

    scorecard_array = np.asarray(scorecard)
    performance = scorecard_array.sum() / scorecard_array.size
    print('performance = ', performance)
    return performance


def run_audio_classification(real_folder, fake_folder, epochs=20, hidden_nodes=200, learning_rate=0.1):
    print('Loading audio files and extracting MFCC features...')
    X, y = load_audio_dataset(real_folder, fake_folder)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Fit the scaler on training data only to avoid data leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    input_nodes = X_train.shape[1]  # 13 MFCC means + 13 MFCC standard deviations = 26
    output_nodes = 2  # 0 = real, 1 = fake

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate).to(device)

    print(f'Training with {len(X_train)} samples, testing with {len(X_test)} samples')
    print(f'input_nodes = {input_nodes}, hidden_nodes = {hidden_nodes}, output_nodes = {output_nodes}')

    for e in range(epochs):
        training_records = list(zip(X_train, y_train))
        random.shuffle(training_records)

        total_loss = 0.0
        for inputs, label in training_records:
            targets = make_targets(label, output_nodes)
            total_loss += n.train_record(inputs, targets)

        average_loss = total_loss / len(training_records)
        print(f'Epoch {e + 1}/{epochs}, average loss = {average_loss:.4f}')

    return evaluate_network(n, X_test, y_test)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a simple feedforward network on real/fake audio MFCC features.')
    parser.add_argument('--real-folder', required=True, help='Folder containing real human audio files')
    parser.add_argument('--fake-folder', required=True, help='Folder containing fake or AI-generated audio files')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--hidden-nodes', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_audio_classification(
        real_folder=args.real_folder,
        fake_folder=args.fake_folder,
        epochs=args.epochs,
        hidden_nodes=args.hidden_nodes,
        learning_rate=args.learning_rate
    )

