import torch
import os
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from MORALS.models import *

class TrainingConfig:
    def __init__(self, weights_str):
        self.weights_str = weights_str
        self.parse_config()
    
    def parse_config(self):
        ids = self.weights_str.split('_')
        self.weights = []
        for _, id in enumerate(ids):
            self.weights.append([float(e) for e in id.split('x')[:-1]])
            if len(self.weights[-1]) != 4:
                print("Expected 4 values per training config, got ", len(self.weights[-1]))
                raise ValueError
    
    def __getitem__(self, key):
        return self.weights[key]
    
    def __len__(self):
        return len(self.weights)

class LabelsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(LabelsLoss, self).__init__()
        self.reduction = reduction
        self.scale = 100.0

    def forward(self, x, y):
        pairwise_distance = torch.linalg.vector_norm(x - y, ord=2, dim=1)
        loss = torch.sigmoid(-self.scale * pairwise_distance)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError("Invalid reduction type")

class Training:
    def __init__(self, config, loaders, verbose):
        self.encoder = Encoder(config)
        self.dynamics = LatentDynamics(config)
        self.decoder = Decoder(config)

        self.verbose = bool(verbose)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

        self.encoder.to(self.device)
        self.dynamics.to(self.device)
        self.decoder.to(self.device)

        self.dynamics_train_loader = loaders['train_dynamics']
        self.dynamics_test_loader = loaders['test_dynamics']
        self.labels_train_loader = loaders['train_labels']
        self.labels_test_loader = loaders['test_labels']

        self.reset_losses()

        self.dynamics_criterion = nn.MSELoss(reduction='mean')
        self.labels_criterion = LabelsLoss(reduction='mean')

        self.lr = config["learning_rate"]

        self.model_dir = config["model_dir"]
        self.log_dir = config["log_dir"]

    def save_models(self):
        torch.save(self.encoder, os.path.join(self.model_dir, 'encoder.pt'))
        torch.save(self.dynamics, os.path.join(self.model_dir, 'dynamics.pt'))
        torch.save(self.decoder, os.path.join(self.model_dir, 'decoder.pt'))
    
    def save_logs(self, suffix):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        with open(os.path.join(self.log_dir, 'train_losses_' + suffix + '.pkl'), 'wb') as f:
            pickle.dump(self.train_losses, f)
        
        with open(os.path.join(self.log_dir, 'test_losses_' + suffix + '.pkl'), 'wb') as f:
            pickle.dump(self.test_losses, f)
            
    def reset_losses(self):
        self.train_losses = {'loss_ae1': [], 'loss_ae2': [], 'loss_dyn': [], 'loss_contrastive': [], 'loss_total': []}
        self.test_losses = {'loss_ae1': [], 'loss_ae2': [], 'loss_dyn': [], 'loss_contrastive': [], 'loss_total': []}
    
    def forward(self, x_t, x_tau):
        x_t = x_t.to(self.device)
        x_tau = x_tau.to(self.device)

        z_t = self.encoder(x_t)
        x_t_pred = self.decoder(z_t)

        z_tau = self.encoder(x_tau)
        x_tau_pred = self.decoder(z_tau)

        z_tau_pred = self.dynamics(z_t)
        x_tau_pred_dyn = self.decoder(z_tau_pred)

        return (x_t, x_tau, x_t_pred, z_tau, z_tau_pred, x_tau_pred_dyn)

    def dynamics_losses(self, forward_pass, weight):
        x_t, x_tau, x_t_pred, z_tau, z_tau_pred, x_tau_pred_dyn = forward_pass

        loss_ae1 = self.dynamics_criterion(x_t, x_t_pred)
        loss_ae2 = self.dynamics_criterion(x_tau, x_tau_pred_dyn)
        loss_dyn = self.dynamics_criterion(z_tau_pred, z_tau)
        loss_total = loss_ae1 * weight[0] + loss_ae2 * weight[1] + loss_dyn * weight[2]
        return loss_ae1, loss_ae2, loss_dyn, loss_total

    def labels_losses(self, encodings, pairs, weight):
        return self.labels_criterion(encodings[pairs['successes']], encodings[pairs['failures']]) * weight


    def train(self, epochs=1000, patience=50, weight=[1,1,1,0]):
        '''
        Function that trains all the models with all the losses and weight.
        It will stop if the test loss does not improve for "patience" epochs.
        '''
        weight_bool = [bool(i) for i in weight]
        list_parameters = (weight_bool[0] or weight_bool[1] or weight_bool[2]) * (list(self.encoder.parameters()) + list(self.decoder.parameters()))
        list_parameters += (weight_bool[1] or weight_bool[2]) * list(self.dynamics.parameters())
        optimizer = torch.optim.Adam(list_parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001, patience=patience, verbose=True)
        for epoch in tqdm(range(epochs)):
            loss_ae1_train = 0
            loss_ae2_train = 0
            loss_dyn_train = 0
            loss_contrastive_train = 0

            epoch_train_loss = 0
            epoch_test_loss  = 0


            if weight_bool[0] or weight_bool[1] or weight_bool[2]: 
                self.encoder.train() 
                self.decoder.train() 
            if weight_bool[1] or weight_bool[2]: 
                self.dynamics.train()

            num_batches = min(len(self.dynamics_train_loader), len(self.labels_train_loader))
            for (x_t, x_tau), (pairs, x_final) in zip(self.dynamics_train_loader, self.labels_train_loader):
                optimizer.zero_grad()

                # Forward pass
                forward_pass = self.forward(x_t, x_tau)
                # Compute losses
                loss_ae1, loss_ae2, loss_dyn, loss_total = self.dynamics_losses(forward_pass, weight)
                loss_con = 0
                if weight[3] != 0:
                    x_final = x_final.to(self.device)
                    z_final = self.encoder(x_final)
                    loss_con = self.labels_losses(z_final, pairs, weight[3])
                    loss_total += loss_con
                    loss_contrastive_train += loss_con.item()
                # Backward pass
                loss_total.backward()
                optimizer.step()

                loss_ae1_train += loss_ae1.item()
                loss_ae2_train += loss_ae2.item()
                loss_dyn_train += loss_dyn.item()
                epoch_train_loss += loss_total.item()

            epoch_train_loss /= num_batches

            self.train_losses['loss_ae1'].append(loss_ae1_train / num_batches)
            self.train_losses['loss_ae2'].append(loss_ae2_train / num_batches)
            self.train_losses['loss_dyn'].append(loss_dyn_train / num_batches)
            self.train_losses['loss_contrastive'].append(loss_contrastive_train / num_batches)
            self.train_losses['loss_total'].append(epoch_train_loss)

            with torch.no_grad():
                loss_ae1_test = 0
                loss_ae2_test = 0
                loss_dyn_test = 0
                loss_contrastive_test = 0

                if weight_bool[0] or weight_bool[1] or weight_bool[2]:  
                    self.encoder.eval() 
                    self.decoder.eval() 
                if weight_bool[1] or weight_bool[2]: 
                    self.dynamics.eval()

                num_batches = min(len(self.dynamics_test_loader), len(self.labels_test_loader))
                for (x_t, x_tau), (pairs, x_final) in zip(self.dynamics_test_loader, self.labels_test_loader):
                    # Forward pass
                    forward_pass = self.forward(x_t, x_tau)
                    # Compute losses
                    loss_ae1, loss_ae2, loss_dyn, loss_total = self.dynamics_losses(forward_pass, weight)

                    loss_ae1_test += loss_ae1.item() 
                    loss_ae2_test += loss_ae2.item() 
                    loss_dyn_test += loss_dyn.item() 
                    epoch_test_loss += loss_total.item()

                    if weight[3] != 0:
                        x_final = x_final.to(self.device)
                        z_final = self.encoder(x_final)
                        loss_con = self.labels_losses(z_final, pairs, weight[3])
                        loss_contrastive_test += loss_con.item()

                epoch_test_loss /= num_batches

                self.test_losses['loss_ae1'].append(loss_ae1_test / num_batches)
                self.test_losses['loss_ae2'].append(loss_ae2_test / num_batches)
                self.test_losses['loss_dyn'].append(loss_dyn_test / num_batches)
                self.test_losses['loss_contrastive'].append(loss_contrastive_test / num_batches)
                self.test_losses['loss_total'].append(epoch_test_loss)

            scheduler.step(epoch_test_loss)
            
            if epoch >= patience:
                if np.mean(self.test_losses['loss_total'][-patience:]) > np.mean(self.test_losses['loss_total'][-patience-1:-1]):
                    if self.verbose:
                        print("Early stopping")
                    break
            
            if self.verbose:
                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, epochs, epoch_train_loss, epoch_test_loss))


class SequenceTraining:
    def __init__(self, config, loaders, verbose):
        self.transformer = MaskedTransformer(config)
        self.dynamics_model = None
        self.verbose = bool(verbose)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

        self.transformer.to(self.device)

        self.criterion = nn.MSELoss(reduction='none')

        self.lr = config["learning_rate"]

        self.model_dir = config["model_dir"]
        self.load_seq_model_file = config.get('load_seq_model', None)
        self.load_dyn_model_file = config.get('load_dyn_model', None)
        self.log_dir = config["log_dir"]

        self.train_loader = loaders['train_sequence']
        self.test_loader = loaders['test_sequence']
        self.dynamics_train_loader = loaders['dynamics_train']
        self.dynamics_test_loader = loaders['dynamics_test']

        self.reset_losses()

    def save_models(self):
        torch.save(self.transformer, os.path.join(self.model_dir, 'encoder.pt'))
        if self.dynamics_model is not None:
            torch.save(self.dynamics_model, os.path.join(self.model_dir, 'dynamics_model.pt'))
    
    def save_logs(self, suffix):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        with open(os.path.join(self.log_dir, 'train_losses_' + suffix + '.pkl'), 'wb') as f:
            pickle.dump(self.train_losses, f)
        
        with open(os.path.join(self.log_dir, 'test_losses_' + suffix + '.pkl'), 'wb') as f:
            pickle.dump(self.test_losses, f)

    def save_plots(self, fig, suffix):
        plot_log_dir = os.path.join(self.log_dir, 'plots')
        if not os.path.exists(plot_log_dir):
            os.makedirs(plot_log_dir)
        fig.savefig(os.path.join(plot_log_dir, 'plot_' + suffix + '.png'))
        plt.close(fig)
    
    def reset_losses(self):
        self.train_losses = {'loss': [], 'loss_total': []}
        self.test_losses = {'loss': [], 'loss_total': []}

    def load_seq_model(self):
        model_file = os.path.join(self.model_dir, self.load_seq_model_file)
        if self.load_seq_model_file is not None and os.path.exists(model_file):
            self.transformer = torch.load(model_file)
            self.transformer.to(self.device)

    def load_dyn_model(self):
        model_file = os.path.join(self.model_dir, self.load_dyn_model_file)
        if self.load_dyn_model_file is not None and os.path.exists(model_file):
            self.dynamics_model = torch.load(model_file)
            self.dynamics_model.to(self.device)

    def train(self, epochs=1000, patience=50, weight=[1,1,1,0]):
        '''
        Function that trains all the models with all the losses and weight.
        It will stop if the test loss does not improve for "patience" epochs.
        '''
        
        optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001, patience=patience, verbose=True)
        # dataloader = 
        for epoch in tqdm(range(epochs)):
            loss_train = 0
            epoch_train_loss = 0
            epoch_test_loss  = 0

            self.transformer.train()

            num_batches = len(self.train_loader)
            for inp_seq, out_seq, mask, _, _, seq_len in self.train_loader:
                optimizer.zero_grad()
                inp_seq = inp_seq.to(self.device)
                mask = mask.to(self.device)

                out_seq = self.transformer(inp_seq) # target
                loss = torch.sum(self.criterion(inp_seq, out_seq), axis=-1)
                loss = torch.sum(loss * mask) / torch.sum(mask)
                loss.backward()
                optimizer.step()

                loss_train += loss.item()
                epoch_train_loss += loss.item()

            epoch_train_loss /= num_batches

            self.train_losses['loss'].append(loss_train / num_batches)
            self.train_losses['loss_total'].append(epoch_train_loss)

            with torch.no_grad():
                loss_test = 0
                self.transformer.eval()

                num_batches = 1 # len(self.test_loader)
                for idx, (inp_seq, out_seq, mask, true_seq, seq_len, label) in enumerate(self.test_loader):
                    optimizer.zero_grad()
                    inp_seq = inp_seq.to(self.device)
                    true_seq = true_seq.to(self.device)
                    mask = mask.to(self.device)

                    out_seq = self.transformer(inp_seq)

                    # if idx == 0 and (epoch % 5 == 0):
                        
                    #     # seq = torch.zeros_like(true_seq)
                    #     # latent = torch.zeros((true_seq.shape[0], true_seq.shape[1], 2))
                    #     # seq[:, 0, :] = true_seq[:, 0, :]
                    #     # for seq_idx in tqdm(range(seq.shape[1]-1)):
                    #     #     seq[:, seq_idx, :] = true_seq[:, seq_idx, :]
                    #     #     latent[:, seq_idx+1, :] = self.transformer.forward_transformer_enc(seq)[:, seq_idx+1, :]
                    #     latent = self.transformer.get_latent_embeddings(true_seq).cpu()
                    #     plot_idxs = np.random.choice(label.shape[0], 20) # random 20 plots
                    #     fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                    #     ax = axes.flatten()
                    #     failure_idxs = plot_idxs[label[plot_idxs] == 0]
                    #     success_idxs = plot_idxs[label[plot_idxs] == 1]
                    #     failure = latent[plot_idxs][label[plot_idxs] == 0]
                    #     success = latent[plot_idxs][label[plot_idxs] == 1]

                    #     for i in range(min(failure.shape[0], 5)):
                    #         ax[0].plot(failure[i, :seq_len[failure_idxs][i], 0], failure[i, :seq_len[failure_idxs][i], 1], label='failure', color='black')
                    #         ax[0].scatter(failure[i, seq_len[failure_idxs][i]-1, 0], failure[i, seq_len[failure_idxs][i]-1, 1], color='red', s=100)

                            
                    #         # ax[3].plot(failure[i, :, 0], failure[i, :, 1], label='failure', color='red')
                    #     for i in range(min(success.shape[0], 5)):
                    #         ax[0].plot(success[i, :seq_len[success_idxs][i], 0], success[i, :seq_len[success_idxs][i], 1], label='success', color='black')
                    #         ax[0].scatter(success[i, seq_len[success_idxs][i]-1, 0], success[i, seq_len[success_idxs][i]-1, 1], color='green', s=100)

                    #         ax[1].plot(success[i, :seq_len[success_idxs][i], 0], success[i, :seq_len[success_idxs][i], 1], label='success', color='black')
                    #         ax[1].scatter(success[i, seq_len[success_idxs][i]-1, 0], success[i, seq_len[success_idxs][i]-1, 1], color='green', s=100)
                    #     ctr_neg = 0
                    #     ctr_pos = 0
                    #     num_points = 20
                    #     for i in range(latent.shape[0]):
                    #         if ctr_neg == num_points and ctr_pos == num_points:
                    #             break
                    #         if (label[i] == 0 and ctr_neg == num_points )or (label[i] == 1 and ctr_pos == num_points):
                    #             continue
                            
                    #         if label[i] == 0:
                    #             ctr_neg += 1
                    #         else:
                    #             ctr_pos += 1
                    #         ax[2].scatter(latent[i, 0, 0], latent[i, 0, 1], color='black' if label[i] == 0 else 'yellow')
                    #         ax[2].scatter(latent[i, seq_len[i]-1, 0], latent[i, seq_len[i]-1, 1], color='red' if label[i] == 0 else 'green', marker=(5, 1) if label[i] == 1 else (5, 0), s=100 if label[i] == 1 else 50)
                            
                    #         # ax[3].plot(latent[i, :seq_len[i], 0], latent[i, :seq_len[i], 1], color='black' if label[i] == 0 else 'yellow')
                    #         ax[3].scatter(latent[i, seq_len[i]-1, 0], latent[i, seq_len[i]-1, 1], color='red' if label[i] == 0 else 'green', marker=(5, 1) if label[i] == 1 else (5, 0), s=100 if label[i] == 1 else 50)
                            
                    #     # ax[0].set_title("failure")
                    #     # ax[0].set(ylim=(-1,1), xlim=(-1.0, 1.0))
                    #     # ax[1].set_title("success")
                    #     # ax[1].set(ylim=(-1,1), xlim=(-1.0, 1.0))

                    #     # ax[2].set(ylim=(-1,1), xlim=(-1.0, 1.0))
                    #     # ax[3].set(ylim=(-1,1), xlim=(-1.0, 1.0))

                    #     for a in ax:
                    #         a.grid()
                    #     # # ax.legend()
                    #     self.save_plots(fig, str(epoch))
                        

                    loss = torch.sum(self.criterion(inp_seq, out_seq), axis=-1)
                    loss = torch.sum(loss * mask) / torch.sum(mask)

                    loss_test += loss.item()
                    epoch_test_loss += loss.item()

                epoch_test_loss /= num_batches

                self.test_losses['loss'].append(loss_test / num_batches)
                self.test_losses['loss_total'].append(epoch_test_loss)
                
            scheduler.step(epoch_test_loss)
            
            if epoch >= patience:
                if np.mean(self.test_losses['loss_total'][-patience:]) > np.mean(self.test_losses['loss_total'][-patience-1:-1]):
                    if self.verbose:
                        print("Early stopping")
                    break
            
            if self.verbose:
                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, epochs, epoch_train_loss, epoch_test_loss))


    def train_dynamics_model(self, config, epochs=3, patience=50):
        
        self.dynamics_model = LatentDynamics(config)
        self.dynamics_model = self.dynamics_model.to(self.device)
        optimizer = torch.optim.Adam(self.dynamics_model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001, patience=patience, verbose=True)
        self.dynamics_criterion = nn.MSELoss(reduction='mean')
        self.transformer.eval()

        for epoch in tqdm(range(epochs)):
            loss_train = 0
            epoch_train_loss = 0
            epoch_test_loss  = 0
    
            self.dynamics_model.train()

            num_batches = len(self.dynamics_train_loader)
            for x_t, x_tau in self.dynamics_train_loader:
                optimizer.zero_grad()
                x_t = x_t.to(self.device)
                x_tau = x_tau.to(self.device)
                with torch.no_grad():
                    z_t = self.transformer.get_latent_embeddings(x_t)
                    z_tau = self.transformer.get_latent_embeddings(x_tau)
                
                # print(z_t.shape, z_tau.shape)

                z_tau_pred = self.dynamics_model.forward(z_t)
                loss = self.dynamics_criterion(z_tau, z_tau_pred) * 10
                loss.backward()
                optimizer.step()

                loss_train += loss.item()
                epoch_train_loss += loss.item()

            epoch_train_loss /= num_batches

            self.train_losses['loss'].append(loss_train / num_batches)
            self.train_losses['loss_total'].append(epoch_train_loss)

            with torch.no_grad():
                loss_test = 0
                self.dynamics_model.eval()
                num_batches = len(self.dynamics_test_loader)
                for x_t, x_tau in self.dynamics_test_loader:
                    x_t = x_t.to(self.device)
                    x_tau = x_tau.to(self.device)
                    z_t = self.transformer.get_latent_embeddings(x_t)
                    z_tau = self.transformer.get_latent_embeddings(x_tau)

                    z_tau_pred = self.dynamics_model.forward(z_t)
                    loss = self.dynamics_criterion(z_tau, z_tau_pred) * 10
                    loss_test += loss.item()
                    epoch_test_loss += loss.item()

                epoch_test_loss /= num_batches

                self.test_losses['loss'].append(loss_test / num_batches)
                self.test_losses['loss_total'].append(epoch_test_loss)
                
            scheduler.step(epoch_test_loss)
            
            if epoch >= patience:
                if np.mean(self.test_losses['loss_total'][-patience:]) > np.mean(self.test_losses['loss_total'][-patience-1:-1]):
                    if self.verbose:
                        print("Early stopping")
                    break
            
            if self.verbose:
                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, epochs, epoch_train_loss, epoch_test_loss))
    

    def make_plots(self):
        # sequence plots
        self.transformer.eval()
        with torch.no_grad():
            inp_seq, out_seq, mask, true_seq, seq_len, label = next(iter(self.test_loader))
            true_seq = true_seq.to(self.device)
            raw_latent = self.transformer.get_latent_embeddings(true_seq)
            latent = raw_latent.clone().cpu().numpy()

            failure_idxs = (label == 0).nonzero().squeeze(-1).numpy()
            sampled_failure_idxs = np.random.choice(failure_idxs, 25)

            fig = plt.figure(figsize=(8, 8))
            ax = fig.gca()
            for i in sampled_failure_idxs:
                ax.plot(latent[i, :, 0], latent[i, :, 1], color='black')

            # for i in sampled_failure_idxs:
            #     if i == 0:
            #         ax.scatter(latent[i, 0, 0], latent[i, 0, 1], color='blue', label='initial state')
            #     else:
            #         ax.scatter(latent[i, 0, 0], latent[i, 0, 1], color='blue')

            for i in sampled_failure_idxs:
                if i == 0:
                    ax.scatter(latent[i, seq_len[i]-1, 0], latent[i, seq_len[i]-1, 1], color='red', s=300, marker=(5, 1), label='final state')
                else:
                    ax.scatter(latent[i, seq_len[i]-1, 0], latent[i, seq_len[i]-1, 1], color='red', s=300, marker=(5, 1))

            # ax.legend()
            plt.grid()
            self.save_plots(fig, 'failure')


            success_idxs = (label == 1).nonzero().squeeze(-1).numpy()
            sampled_success_idxs = np.random.choice(success_idxs, 25)
            print(sampled_success_idxs.shape, latent[sampled_success_idxs].shape)
            fig = plt.figure(figsize=(8, 8))
            ax = fig.gca()
            for i in sampled_success_idxs:
                ax.plot(latent[i, :, 0], latent[i, :, 1], color='black')

            # for i in sampled_success_idxs:
            #     ax.scatter(latent[i, 0, 0], latent[i, 0, 1], color='blue')
                
            for i in sampled_success_idxs:
                ax.scatter(latent[i, seq_len[i]-1, 0], latent[i, seq_len[i]-1, 1], color='green', s=300, marker=(5, 1))
            plt.grid()
            self.save_plots(fig, 'success')


            fig = plt.figure(figsize=(8, 8))
            ax = fig.gca()
            for i in sampled_failure_idxs:
                ax.plot(latent[i, :, 0], latent[i, :, 1], color='black')
            for i in sampled_success_idxs:
                ax.plot(latent[i, :, 0], latent[i, :, 1], color='black')

            # for i in sampled_failure_idxs:
            #     ax.scatter(latent[i, 0, 0], latent[i, 0, 1], color='red')
            # for i in sampled_success_idxs:
            #     ax.scatter(latent[i, 0, 0], latent[i, 0, 1], color='green')
                
            for i in sampled_failure_idxs:
                ax.scatter(latent[i, seq_len[i]-1, 0], latent[i, seq_len[i]-1, 1], color='red', s=300, marker=(5, 0))
            for i in sampled_success_idxs:
                ax.scatter(latent[i, seq_len[i]-1, 0], latent[i, seq_len[i]-1, 1], color='green', s=300, marker=(5, 1))
            plt.grid()
            self.save_plots(fig, 'combined')

            
            # # dynamics plots
            # print(raw_latent)
            # dynamics_latent = []
            # latent = raw_latent.clone()[:, :1, :]
            # for i in range(raw_latent.shape[1]):
            #     latent_tau = self.dynamics_model.forward(latent)
            #     dynamics_latent.append(latent_tau.clone().unsqueeze(1).cpu())
            #     latent = latent_tau.clone()
            
            # latent = torch.concatenate(dynamics_latent, dim=1).squeeze(2)

            # fig = plt.figure(figsize=(8, 8))
            # ax = fig.gca()
            # for i in sampled_failure_idxs:
            #     ax.plot(latent[i, :, 0], latent[i, :, 1], color='black')


            # for i in sampled_failure_idxs:
            #     if i == 0:
            #         ax.scatter(latent[i, seq_len[i]-1, 0], latent[i, seq_len[i]-1, 1], color='red', s=300, marker=(5, 1), label='final state')
            #     else:
            #         ax.scatter(latent[i, seq_len[i]-1, 0], latent[i, seq_len[i]-1, 1], color='red', s=300, marker=(5, 1))

            # # ax.legend()
            # plt.grid()
            # self.save_plots(fig, 'latent_failure')

            # # print(latent_tau.shape)
            # # self.dynamics_model.forward()

            # success_idxs = (label == 1).nonzero().squeeze(-1).numpy()
            # sampled_success_idxs = np.random.choice(success_idxs, 25)
            # print(sampled_success_idxs.shape, latent[sampled_success_idxs].shape)
            # fig = plt.figure(figsize=(8, 8))
            # ax = fig.gca()
            # for i in sampled_success_idxs:
            #     ax.plot(latent[i, :, 0], latent[i, :, 1], color='black')
                
            # for i in sampled_success_idxs:
            #     ax.scatter(latent[i, seq_len[i]-1, 0], latent[i, seq_len[i]-1, 1], color='green', s=300, marker=(5, 1))
            # plt.grid()
            # self.save_plots(fig, 'latent_success')



        


        