import os, random, csv, humanize, sys

from tqdm import tqdm
import pandas as pd
import librosa
import torchaudio
from augments import AudioOfflineTransforms

from externals.pytorch_balanced_sampler.sampler import SamplerFactory

from model import nime2025

import torch
from torch.utils.data import TensorDataset, DataLoader

torchaudio.set_audio_backend("sox_io")

# -------------------------------------------------------
# Dataset preparation  
# -------------------------------------------------------
class DatasetSplitter:
    @staticmethod
    def split_train_validation(csv_path, train_path, test_path, val_path, val_split='train', val_ratio=0.2):
        """Splits the dataset into training, validation, and test sets, and saves the information in a CSV file."""

        if val_path is not None:
            val_path = val_path
            print('Validation path provided:', val_path, 'No validation split required.')
        else:
            val_path = None

        if val_path is None:
            if val_split not in ['train', 'test']:
                raise ValueError("val_split must be either 'train' or 'test'.")

        # Function to process files and write to CSV
        def process_files(files, label, set_type, writer):
            for file in files:
                writer.writerow([file, label, set_type])

        # Process training and validation sets
        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['file_path', 'label', 'set'])

            # Process train directory
            for root, dirs, files in tqdm(os.walk(train_path), desc='Process training audio files.'):
                label = os.path.basename(root)
                all_train_files = [os.path.join(root, f) for f in files if f.lower().endswith(('.wav', '.aiff', '.aif', '.mp3'))]

                if val_split == 'train' and val_path is None:
                    # Split into train and validation sets
                    num_files = len(all_train_files)
                    num_val = int(num_files * val_ratio)
                    val_files = random.sample(all_train_files, num_val)
                    train_files = list(set(all_train_files) - set(val_files))

                    process_files(train_files, label, 'train', writer)
                    process_files(val_files, label, 'val', writer)
                else:
                    process_files(all_train_files, label, 'train', writer)

            # Process test directory
            for root, dirs, files in tqdm(os.walk(test_path), desc='Process test audio files.'):
                label = os.path.basename(root)
                all_test_files = [os.path.join(root, f) for f in files if f.lower().endswith(('.wav', '.aiff', '.aif', '.mp3'))]

                if val_split == 'test' and val_path is None:
                    # Split into test and validation sets
                    num_files = len(all_test_files)
                    num_val = int(num_files * val_ratio)
                    val_files = random.sample(all_test_files, num_val)
                    test_files = list(set(all_test_files) - set(val_files))

                    process_files(test_files, label, 'test', writer)
                    process_files(val_files, label, 'val', writer)
                else:
                    process_files(all_test_files, label, 'test', writer)

            # Process validation directory if provided
            if val_path is not None:
                for root, dirs, files in tqdm(os.walk(val_path), desc='Process validation audio files.'):
                    label = os.path.basename(root)
                    all_val_files = [os.path.join(root, f) for f in files if f.lower().endswith(('.wav', '.aiff', '.aif', '.mp3'))]
                    process_files(all_val_files, label, 'val', writer)

        print(f"CSV dataset file created successfully.")


class DatasetValidator:
    @staticmethod
    def validate_labels(csv_file):
        """Validates that the train, test, and val sets have the same unique labels."""
        data = pd.read_csv(csv_file)

        train_labels = set(data[data['set'] == 'train']['label'].unique())
        test_labels = set(data[data['set'] == 'test']['label'].unique())
        val_labels = set(data[data['set'] == 'val']['label'].unique())

        missing_in_train = test_labels.union(val_labels) - train_labels
        missing_in_test = train_labels.union(val_labels) - test_labels
        missing_in_val = train_labels.union(test_labels) - val_labels

        if missing_in_train or missing_in_test or missing_in_val:
            print("Labels mismatch found:")
            if missing_in_train:
                print(f"Missing in train: {missing_in_train}")
                print("Details of missing classes in train:")
                print(data[data['label'].isin(missing_in_train)])
            if missing_in_test:
                print(f"Missing in test: {missing_in_test}")
                print("Details of missing classes in test:")
                print(data[data['label'].isin(missing_in_test)])
            if missing_in_val:
                print(f"Missing in val: {missing_in_val}")
                print("Details of missing classes in val:")
                print(data[data['label'].isin(missing_in_val)])
            raise ValueError("Mismatch in labels between train, test, and val sets.")
        
        print("Label validation passed: All sets have the same labels.")

    @staticmethod
    def get_num_classes_from_csv(csv_file):
        data = pd.read_csv(csv_file)
        return len(data['label'].unique())
    
    @staticmethod
    def get_classnames_from_csv(csv_file):
        data = pd.read_csv(csv_file)
        return sorted(data['label'].unique())

# -------------------------------------------------------
# Data preparation  
# -------------------------------------------------------
class PrepareData:
    """Prepare datasets in processing the audio samples from train, val, test dirs."""
    def __init__(self,  csv_file_path, device, target_sr, batch_size, augment):
        self.device = device
        self.csv = csv_file_path
        self.segment_length = 14 * 512 # 14 frames * 512 samples = 7168 samples = 0.896s
        self.target_sr = target_sr
        self.batch_size = batch_size
        self.augment = augment

    def prepare(self):
        num_classes = DatasetValidator.get_num_classes_from_csv(self.csv)
        classnames = DatasetValidator.get_classnames_from_csv(self.csv)
        train_dataset = ProcessDataset('train', self.csv, self.device, self.target_sr, self.segment_length, self.augment)
        test_dataset = ProcessDataset('test', self.csv, self.device, self.target_sr, self.segment_length, False)
        val_dataset = ProcessDataset('val', self.csv, self.device, self.target_sr, self.segment_length, False)

        train_loader = BalancedDataLoader(train_dataset.get_data(), self.batch_size).get_dataloader()
        test_loader = DataLoader(test_dataset.get_data(), self.batch_size)
        val_loader = DataLoader(val_dataset.get_data(), self.batch_size)
        
        print('Data successfully loaded into DataLoaders.')

        return train_loader, test_loader, val_loader, num_classes, classnames, self.segment_length

class ProcessDataset:
    def __init__(self, set_type, csv_path, device, target_sr, segment_length, augment):
        self.set_type = set_type
        self.csv_path = csv_path
        self.target_sr = target_sr
        self.segment_length = segment_length
        self.segment_overlap = False
        self.padding = 'minimal'
        self.device = device
        self.offline_aug = augment

        self.data = pd.read_csv(self.csv_path)
        
        self.data = self.data[self.data['set'] == self.set_type]
        
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.data['label'].unique()))}

        self.X = []
        self.y = []

        self.process_all_files()

    def remove_silence(self, waveform):
        """Remove silence from the audio waveform."""
        wav = waveform.detach().cpu().numpy()
        wav = librosa.effects.trim(wav)
        return torch.tensor(wav[0])
    
    def pad_waveform(self, waveform, target_length):
        """Add silence to the waveform to match the target length."""
        extra_length = target_length - waveform.size(1)
        if extra_length > 0:
            silence = torch.zeros((waveform.size(0), extra_length))
            waveform = torch.cat((waveform, silence), dim=1)
        return waveform
    
    def process_segment(self, waveform):
        """Process the waveform by dividing it into segments with or without overlap."""
        segments = []
        num_samples = waveform.size(1)

        for i in range(0, num_samples, self.segment_length if not self.segment_overlap else self.segment_length // 2):
            if i + self.segment_length <= num_samples:
                segment = waveform[:, i:i + self.segment_length]
            else:
                if self.padding == 'full':
                    valid_length = num_samples - i
                    segment = torch.zeros((waveform.size(0), self.segment_length))
                    segment[:, :valid_length] = waveform[:, i:i + valid_length]
            segments.append(segment)

        return segments
    
    def process_all_files(self):
        """Process all audio files and store them in X and y."""
        augmenter = AudioOfflineTransforms(self.target_sr, self.device) if self.offline_aug else None
        if self.offline_aug and self.set_type == 'train':
            print(f'self offline augmentations: {self.offline_aug}')  

        for _, row in tqdm(self.data.iterrows()):
            file_path, label_name = row['file_path'], row['label']
            label = self.label_map[label_name]

            waveform, original_sr = torchaudio.load(file_path)

            if original_sr != self.target_sr:
                waveform = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=self.target_sr)(waveform)

            if label_name != 'silence':
                waveform = self.remove_silence(waveform)

            if waveform.shape[0] == 2:
                waveform = waveform[0, :].unsqueeze(0)

            if self.padding == 'minimal' and waveform.size(1) < self.segment_length:
                waveform = self.pad_waveform(waveform, self.segment_length)

            segments = self.process_segment(waveform)

            for segment in segments:
                if augmenter and self.set_type == 'train':
                    aug1, aug2, aug3 = augmenter(segment)
                    self.X.extend([aug1, aug2, aug3])
                    self.y.extend([label] * 3)

                self.X.append(segment)
                self.y.append(label)

        self.X = torch.stack(self.X)
        self.y = torch.tensor(self.y)

    def get_data(self):
        return TensorDataset(self.X, self.y)

class BalancedDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.num_classes = self.get_num_classes()
        self.batch_size = batch_size

        all_targets = [dataset[i][1].unsqueeze(0) if dataset[i][1].dim() == 0 else dataset[i][1] for i in range(len(dataset))]
        all_targets = torch.cat(all_targets)

        class_idxs = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            indexes = torch.nonzero(all_targets == i, as_tuple=True)
            if indexes[0].numel() > 0:
                class_idxs[i] = indexes[0].tolist()
            else:
                print(f"Class {i} has no indices")

        total_samples = len(self.dataset)
        n_batches = total_samples // self.batch_size

        class_counts = [0] * self.num_classes
        for i in range(len(self.dataset)):
            _, label = self.dataset[i]
            if label.dim() == 0:
                label = label.item()
            else:
                label = label.argmax().item()
            if 0 <= label < self.num_classes:
                class_counts[label] += 1

        print(f"Class distribution: {class_counts}")

        self.batch_sampler = SamplerFactory().get(
            class_idxs=class_idxs,
            batch_size=self.batch_size,
            n_batches=n_batches,
            alpha=1,
            kind='fixed'
        )

    def get_num_classes(self):
        """ Determines the number of unique classes in the dataset. """
        all_labels = [label.item() for label in self.dataset.tensors[1]]
        unique_classes = set(all_labels)
        num_classes = len(unique_classes)
        print(f"Unique classes detected: {unique_classes}")
        return num_classes
    
    def get_dataloader(self):
        """ Returns a DataLoader with the balanced batch sampler. """
        return DataLoader(
            self.dataset,
            batch_sampler=self.batch_sampler,
            # collate_fn=self.custom_collate_fn,
        )

# -------------------------------------------------------
# Model preparation  
# -------------------------------------------------------
class PrepareModel:
    def __init__(self, device, num_classes, segment_length, sr, classnames):
        self.num_classes = num_classes
        self.segment_length = segment_length
        self.device = device
        self.sr = sr
        self.classnames = classnames
        self.config = 'nime2025'

    def prepare(self):
        model = LoadModel().get_model(self.config, self.num_classes, self.sr, self.classnames, self.segment_length).to(self.device)

        tester = ModelTester(model, input_shape=(1, 1, self.segment_length), device=self.device)
        output = tester.test()
        if output.size(1) != self.num_classes:
            print("Error: Output dimension does not match the number of classes.")
            sys.exit(1)
    
        summary = ModelSummary(model, self.num_classes, self.config)
        summary.print_summary()

        model = ModelInit(model).initialize()
        return model
    
class LoadModel:
    def __init__(self):
        self.models = {
            'nime2025': nime2025,
        }
    
    def get_model(self, model_name, num_classes, sr, classnames, segment_length):
        if model_name in self.models:
            return self.models[model_name](num_classes, sr, classnames, segment_length)

class ModelSummary:
    def __init__(self, model, num_labels, config):
        self.model = model
        self.num_labels = num_labels
        self.config = config

    def get_total_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def print_summary(self):
        total_params = self.get_total_parameters()
        formatted_params = humanize.intcomma(total_params)

        print('-----------------------------------------------')
        print(f"Number of labels: {self.num_labels}")
        print(f"Total number of parameters: {formatted_params}")
        print('-----------------------------------------------')

class ModelTester:
    def __init__(self, model, input_shape, device='cpu'):
        self.model = model
        self.input_shape = input_shape
        self.device = device

    def test(self):
        """Tests the model with a fixed-length random input tensor."""
        self.model.to(self.device)
        self.model.eval()

        random_input = torch.randn(self.input_shape).to(self.device)
        
        with torch.no_grad():
            output = self.model(random_input)
        
        return output
    
class ModelInit:
    def __init__(self, model):
        self.model = model

    def initialize(self):
        """Apply weight initialization to the model layers."""
        init_method = torch.nn.init.xavier_normal_

        for layer in self.model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.Conv1d)):
                init_method(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

        return self.model