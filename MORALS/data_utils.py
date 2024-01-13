from MORALS.systems.utils import get_system

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from MORALS.systems.utils import get_system
import torch
import os

class SequenceDataset(Dataset):
    def __init__(self, config):
        raw_sequences = []
        transformed_sequences = []
        self.labels = []
        self.true_sequence_length = []

        # step = config['step']
        subsample = config['subsample']
        system = get_system(config['system'], config['high_dims'])
        print("Getting data for: ",system.name)

        labels = np.loadtxt(config['labels_fname'], delimiter=',', dtype=str)
        success_files = []
        failure_files = []
        labels_dict = {}
        for i in range(len(labels)):
            labels_dict[labels[i,0]] = int(labels[i,1])
            if int(labels[i,1]) == 1:
                success_files.append(labels[i,0])
            else:
                failure_files.append(labels[i,0])

        print(len(failure_files), len(success_files))

        first_ten = []
        raw_sequence_files = os.listdir(config['data_dir'])
        for_norm = len(raw_sequence_files) 
        for f in tqdm(raw_sequence_files):
            data = np.loadtxt(os.path.join(config['data_dir'], f), delimiter=',')
            
            indices = np.arange(data.shape[0])
            subsampled_indices = indices % subsample == 0
            subsampled_data_untransformed = data[subsampled_indices]
            self.true_sequence_length.append(subsampled_data_untransformed.shape[0])
            # all_max_val.append(subsampled_data_untransformed)
            # all_min_val.append(np.min(subsampled_data_untransformed, axis=0))
            transformed_sequences.append(system.transform(subsampled_data_untransformed))
            first_ten.append(system.transform(subsampled_data_untransformed))
            try:
                self.labels.append(labels_dict[f])
            except KeyError:
                print("No label found for ", f)
                self.labels.append(0)
        
        first_ten = np.concatenate(first_ten, axis=0)
        max_val = np.max(first_ten, axis=0)
        min_val = np.min(first_ten, axis=0)
        padded_transformed_sequences = []
        max_sequence_len = max(self.true_sequence_length)
        for seq in tqdm(transformed_sequences):
            # new_seq = seq
            new_seq = ((seq - min_val)/(max_val-min_val))
            if len(seq) < max_sequence_len:
                padding_len = max_sequence_len - len(seq)
                new_seq = np.concatenate([new_seq, np.zeros((padding_len, seq.shape[-1]))], axis=0)
            padded_transformed_sequences.append(new_seq)

        self.sequences = np.stack(padded_transformed_sequences)
        self.mask_ratio = config['mask_ratio']

        # If model_dir does nto exist, create it
        if not os.path.exists(config['model_dir']):
            os.makedirs(config['model_dir'])

        # Write the normalization parameters to a file
        np.savetxt(os.path.join(config['model_dir'], 'min_val.txt'), min_val, delimiter=',')
        np.savetxt(os.path.join(config['model_dir'], 'max_val.txt'), max_val, delimiter=',')

        # # Convert to torch tensors
        self.sequences = torch.from_numpy(self.sequences).float()
        # self.raw_sequences = torch.from_numpy(self.raw_sequences).float()
        self.max_sequence_length = max_sequence_len
        self.true_sequence_length = np.array(self.true_sequence_length)
        self.true_sequence_length = torch.from_numpy(self.true_sequence_length).long()
        self.labels = np.array(self.labels)
        self.labels = torch.from_numpy(self.labels).long()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        label =  self.labels[idx]
        sequence = self.sequences[idx] # 67-dim
        mask = torch.ones(*sequence.shape[:-1], 1) # trajectory_len, 1
        num_tokens_to_mask = int(self.true_sequence_length[idx] * self.mask_ratio)
        mask_indices = np.random.choice(self.true_sequence_length[idx], num_tokens_to_mask, replace=False)
        mask[mask_indices] = 0.
        in_sequence = sequence.clone()*mask
        out_sequence = sequence.clone()*(1-mask)
        return in_sequence, out_sequence, (1-mask).squeeze(-1), sequence, self.true_sequence_length[idx], label

class DynamicsDataset(Dataset):
    def __init__(self, config):
        Xt = []
        Xnext = []

        step = config['step']
        subsample = config['subsample']
        system = get_system(config['system'], config['high_dims'])
        print("Getting data for: ",system.name)

        for f in tqdm(os.listdir(config['data_dir'])):
            data = np.loadtxt(os.path.join(config['data_dir'], f), delimiter=',')
            indices = np.arange(data.shape[0])
            subsampled_indices = indices % subsample == 0
            subsampled_data_untransformed = data[subsampled_indices]
            subsampled_data = system.transform(subsampled_data_untransformed)
            Xt.append(subsampled_data[:-step])
            Xnext.append(subsampled_data[step:])
            # for i in range(subsampled_data.shape[0] - step):
            #     Xt.append(system.transform(subsampled_data[i]))
            #     Xnext.append(system.transform(subsampled_data[i + step]))

        self.Xt = np.vstack(Xt)
        self.Xnext = np.vstack(Xnext)
        assert len(self.Xt) == len(self.Xnext), "Xt and Xnext must have the same length"

        self.X_min = np.loadtxt(os.path.join(config['model_dir'], 'min_val.txt'))
        self.X_max = np.loadtxt(os.path.join(config['model_dir'], 'max_val.txt'))

        for i in range(self.X_min.shape[0]):
            if np.abs(self.X_min[i] - self.X_max[i]) < 1e-6:
                print("Warning: X_min and X_max are the same for dimension ", i)
                self.X_min[i] -= 1
                self.X_max[i] += 1
        
        self.Xt = (self.Xt - self.X_min) / (self.X_max - self.X_min)
        self.Xnext = (self.Xnext - self.X_min) / (self.X_max - self.X_min)

        # # If model_dir does nto exist, create it
        # if not os.path.exists(config['model_dir']):
        #     os.makedirs(config['model_dir'])

        # # Write the normalization parameters to a file
        # np.savetxt(os.path.join(config['model_dir'], 'X_min.txt'), self.X_min, delimiter=',')
        # np.savetxt(os.path.join(config['model_dir'], 'X_max.txt'), self.X_max, delimiter=',')

        # Convert to torch tensors
        self.Xt = torch.from_numpy(self.Xt).float()
        self.Xnext = torch.from_numpy(self.Xnext).float()

    def __len__(self):
        return len(self.Xt)

    def __getitem__(self, idx):
        return self.Xt[idx], self.Xnext[idx]

class LabelsDataset(Dataset):
    def __init__(self,config):
        labels = np.loadtxt(config['labels_fname'], delimiter=',', dtype=str)
        labels_dict = {}
        for i in range(len(labels)):
            labels_dict[labels[i,0]] = int(labels[i,1])

        self.final_points = []
        self.labels = []

        for f in tqdm(os.listdir(config['data_dir'])):
            data = np.loadtxt(os.path.join(config['data_dir'], f), delimiter=',')
            self.final_points.append(data[-1])
            try:
                self.labels.append(labels_dict[f])
            except KeyError:
                print("No label found for ", f)
                self.labels.append(0)

        self.final_points = np.array(self.final_points)
        system = get_system(config['system'], config['high_dims'])
        self.final_points = system.transform(self.final_points)
        X_max = np.loadtxt(os.path.join(config['model_dir'], 'X_max.txt'), delimiter=',')
        X_min = np.loadtxt(os.path.join(config['model_dir'], 'X_min.txt'), delimiter=',')
        self.final_points = (self.final_points - X_min) / (X_max - X_min)
        self.final_points = torch.from_numpy(self.final_points).float()

        self.labels = np.array(self.labels)
        self.labels = torch.from_numpy(self.labels).long()

        self.generate_opposite_pairs()

    def __len__(self):
        return len(self.opposite_pairs)

    def __getitem__(self, idx):
        return self.opposite_pairs[idx]

    def generate_opposite_pairs(self):
        """
        Generates the opposite pairs where each pair has the form (success, failure)
        """

        # Gathering the indices of the success and failure labels
        success_indices = torch.where(self.labels == 1)[0]
        failure_indices = torch.where(self.labels == 0)[0]

        # Generating cartesian product of success and failure indices
        self.opposite_pairs = torch.stack(torch.meshgrid(success_indices, failure_indices)).T.reshape(-1,2)

        print('Number of pairs: ', len(self.opposite_pairs))


    def collate_fn(self, batch):
        """
        Processes the batch of pairs to create a batch of points present in the pairs and a new batch of pairs is created with the updated indices of the points
        :param batch: batch of pairs of the form [success, failure]
        :return: updated batch of pairs and the batch of points
        """
        pairs = np.array([pair.to('cpu').numpy() for pair in batch])

        # Getting the points present in the batch of pairs
        unique_indices = np.unique(pairs)
        x_batch = self.final_points[unique_indices]
        orig_indices_vs_new_indices = {orig_idx: new_idx for new_idx, orig_idx in enumerate(unique_indices)}

        # Updating the indices of the triplets to match the indices of the points in the batch
        updated_success_indices = []
        updated_failure_indices = []

        for success_idx, failure_idx in pairs:
            updated_success_indices.append(orig_indices_vs_new_indices[success_idx])
            updated_failure_indices.append(orig_indices_vs_new_indices[failure_idx])

        return {"successes": updated_success_indices, "failures": updated_failure_indices}, x_batch

class TrajectoryDataset:
    # Useful for plotting
    def __init__(self, config):
        self.trajs = []
        subsample = config['subsample']

        system = get_system(config['system'], config['high_dims'])
        print("Getting data for: ",system.name)

        self.labels_dict = {}
        self.labels = []

        labels_fname = config['labels_fname']

        if labels_fname is not None:
            labels = np.loadtxt(labels_fname, delimiter=',', dtype=str)
            # Create dict with key as fname and value as label
            for i in range(len(labels)):
                self.labels_dict[labels[i,0]] = int(labels[i,1])

        for f in tqdm(os.listdir(config['data_dir'])):
            raw_data = np.loadtxt(os.path.join(config['data_dir'], f), delimiter=',')
            if len(raw_data) == 0: continue
            indices = np.arange(raw_data.shape[0])
            subsampled_indices = indices % subsample == 0
            subsampled_data = raw_data[subsampled_indices]
            # Transform each state in the trajectory
            self.trajs.append(system.transform(subsampled_data))
            # data = []
            # for i in range(subsampled_data.shape[0]):
            #     data.append(system.transform(subsampled_data[i]))
            # data = np.array(data)
            # self.trajs.append(data)
            if labels_fname is not None:
                try:
                    self.labels.append(self.labels_dict[f])
                except KeyError:
                    print("No label found for ", f)
                    self.labels.append(-1)

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return self.trajs[idx]

    def get_label(self,index):
        return self.labels[index]

    def get_successful_initial_conditions(self):
        assert len(self.trajs) == len(self.labels)
        initial_points = []
        for i in range(len(self.trajs)):
            if self.labels[i] == 1:
                initial_points.append(self.trajs[i][0])
        return np.array(initial_points)

    def get_unsuccessful_initial_conditions(self):
        assert len(self.trajs) == len(self.labels)
        initial_points = []
        for i in range(len(self.trajs)):
            if self.labels[i] == 0:
                initial_points.append(self.trajs[i][0])
        return np.array(initial_points)

    def get_successful_final_conditions(self):
        assert len(self.trajs) == len(self.labels)
        final_points = []
        for i in range(len(self.trajs)):
            if self.labels[i] == 1:
                final_points.append(self.trajs[i][-1])
        return np.array(final_points)

    def get_unsuccessful_final_conditions(self):
        assert len(self.trajs) == len(self.labels)
        final_points = []
        for i in range(len(self.trajs)):
            if self.labels[i] == 0:
                final_points.append(self.trajs[i][-1])
        return np.array(final_points)
