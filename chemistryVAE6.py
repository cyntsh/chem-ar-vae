#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#VERSION 6: Adds an optional regulation loss term to enforce logP to a single dimension so that there is a monotonic relationship
#           Script is probably no longer functional because the 'import selfies' cannot find the correct selfies module; must import selfies0, which is updated in the next version        
        
import os, sys, time
import numpy as np
import torch
import pandas as pd
import selfies0
import yaml
import matplotlib.pyplot as plt
from torch import nn
from random import shuffle

sys.path.append('VAE_dependencies')
from data_loader0 import selfies_to_hot, multiple_smile_to_hot, multiple_selfies_to_hot, len_selfie, split_selfie, hot_to_selfies, logp
from rdkit.Chem import MolFromSmiles
from rdkit import rdBase
from selfies0 import decoder, encoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
rdBase.DisableLog('rdApp.error')


def _make_dir(directory):
    os.makedirs(directory)


class VAE_encode(nn.Module):
    
    def __init__(self, layer_1d, layer_2d, layer_3d, latent_dimension):
        """
        Fully Connected layers to encode molecule to latent space
        """
        super(VAE_encode, self).__init__()
        
        # Reduce dimension upto second last layer of Encoder
        self.encode_nn = nn.Sequential(
            nn.Linear(len_max_molec1Hot, layer_1d),
            nn.ReLU(),
            nn.Linear(layer_1d, layer_2d),
            nn.ReLU(),
            nn.Linear(layer_2d, layer_3d),
			nn.ReLU()
        )
        
        # Latent space mean
        self.encode_mu = nn.Linear(layer_3d, latent_dimension) 
        
        # Latent space variance 
        self.encode_log_var = nn.Linear(layer_3d, latent_dimension)
        
        
    def reparameterize(self, mu, log_var):
        """
        This trick is explained well here:
            https://stats.stackexchange.com/a/16338
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) 
        return eps.mul(std).add_(mu)
    
    
    def forward(self, x):
        """
        Pass throught the Encoder
        """
        # Get results of encoder network
        h1 = self.encode_nn(x)
         
        # latent space
        mu = self.encode_mu(h1)
        log_var = self.encode_log_var(h1)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
        


class VAE_decode(nn.Module):
    
    def __init__(self, latent_dimension, gru_stack_size, gru_neurons_num):
        """
        Through Decoder
        """
        super(VAE_decode, self).__init__()
        self.gru_stack_size = gru_stack_size
        self.gru_neurons_num = gru_neurons_num

        # Simple Decoder
        self.decode_RNN  = nn.GRU(
                input_size  = latent_dimension, 
                hidden_size = gru_neurons_num,
                num_layers  = gru_stack_size,
                batch_first = False)                
        
        self.decode_FC = nn.Sequential(
            nn.Linear(gru_neurons_num, len_alphabet),
        )
    

    def init_hidden(self, batch_size = 1):
        weight = next(self.parameters())
        return weight.new_zeros(self.gru_stack_size, batch_size, self.gru_neurons_num)
                 
                       
    def forward(self, z, hidden):
        """
        A forward pass throught the entire model.
        """
        # Decode
        l1, hidden = self.decode_RNN(z, hidden)    
        decoded = self.decode_FC(l1)        # fully connected layer

        return decoded, hidden



def is_correct_smiles(smiles):    
    """
    Using RDKit to calculate whether molecule is syntactically and semantically valid.
    """
    if smiles == "":
        return 0
    try:
        MolFromSmiles(smiles, sanitize=True)
        return 1
    except Exception:
        return 0


def sample_latent_space(latent_dimension, total_samples): 
    model_encode.eval()
    model_decode.eval()
    
    fancy_latent_point=torch.normal(torch.zeros(latent_dimension),torch.ones(latent_dimension))
    hidden = model_decode.init_hidden() 
    gathered_atoms = []
    for ii in range(len_max_molec):                 # runs over letters from molecules (len=size of largest molecule)
        fancy_latent_point = fancy_latent_point.reshape(1, 1, latent_dimension) 
        fancy_latent_point=fancy_latent_point.to(device)
        decoded_one_hot, hidden = model_decode(fancy_latent_point, hidden)
        decoded_one_hot = decoded_one_hot.flatten()
        decoded_one_hot = decoded_one_hot.detach()
        soft = nn.Softmax(0)
        decoded_one_hot = soft(decoded_one_hot)
        _,max_index=decoded_one_hot.max(0)
        gathered_atoms.append(max_index.data.cpu().numpy().tolist())
            
    model_encode.train()
    model_decode.train()
    
    #test molecules visually
    if total_samples <= 5:
        print('Sample #', total_samples, decoder(hot_to_selfies(gathered_atoms, encoding_alphabet)))
    
    return gathered_atoms




def latent_space_quality(latent_dimension, encoding_alphabet, sample_num):
    total_correct = 0
    all_correct_molecules = set()
    print(f"latent_space_quality:"
          f" Take {sample_num} samples from the latent space")
    
    for sample_i in range(1, sample_num + 1):
        molecule_pre = ''
        for ii in sample_latent_space(latent_dimension, sample_i):
            molecule_pre += encoding_alphabet[ii]
        molecule = molecule_pre.replace(' ', '')

        if type_of_encoding == 1:  # if SELFIES, decode to SMILES
            molecule = selfies0.decoder(molecule)

        if is_correct_smiles(molecule):
            total_correct += 1
            all_correct_molecules.add(molecule)

    return total_correct, len(all_correct_molecules)


def quality_in_validation_set(data_valid):    
    x = [i for i in range(len(data_valid))]  # random shuffle input
    shuffle(x)
    data_valid = data_valid[x]
    
    for batch_iteration in range(min(25,num_batches_valid)):  # batch iterator
        
        current_smiles_start, current_smiles_stop = batch_iteration * batch_size, (batch_iteration + 1) * batch_size
        inp_smile_hot = data_valid[current_smiles_start : current_smiles_stop]
    
        inp_smile_encode = inp_smile_hot.reshape(inp_smile_hot.shape[0], inp_smile_hot.shape[1] * inp_smile_hot.shape[2])
        latent_points, mus, log_vars = model_encode(inp_smile_encode)
        latent_points = latent_points.reshape(1, batch_size, latent_points.shape[1])
    
        hidden = model_decode.init_hidden(batch_size = batch_size)
        decoded_one_hot = torch.zeros(batch_size, inp_smile_hot.shape[1], inp_smile_hot.shape[2]).to(device)
        for seq_index in range(inp_smile_hot.shape[1]):
            decoded_one_hot_line, hidden  = model_decode(latent_points, hidden)            
            decoded_one_hot[:, seq_index, :] = decoded_one_hot_line[0]
    
        decoded_one_hot = decoded_one_hot.reshape(batch_size, inp_smile_hot.shape[1], inp_smile_hot.shape[2])
        _, label_atoms  = inp_smile_hot.max(2)   
          
        # assess reconstruction quality
        _, label_atoms_decoded = decoded_one_hot.max(2)
        
        """
        # print a few decoded molecules to visually test reconstruction
        print('Validation set decoded molecules:')
        print(label_atoms_decoded[:2])
        """
        
        flat_decoded = label_atoms_decoded.flatten()
        flat_input = label_atoms.flatten()
        equal_position_count = 0
        num_position = 0
        for mol in range(len(flat_decoded)):
            if flat_decoded[mol] == flat_input[mol]:
                equal_position_count += 1
            num_position += 1
           
        quality = equal_position_count / num_position *100
    return(quality)

# plugged in from this resource: https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels
 
def split_mol(mol, encoding_alphabet):
    """
    Molecule should be in selfies format. This function splits the molecule
    into its molecular string and property string
    """
    mol = hot_to_selfies(mol, encoding_alphabet)
    prop = ''
    for i in range(settings['properties']['num_categories']+2):
        if '['+str(i)+']' in mol:
            mol = mol.replace('['+str(i)+']', '')
            prop += '['+str(i)+']'
    return mol, prop

def split_prop(prop):
    """
    Prop is a string of three bracketed numbers. This function splits it into
    individual bracketed numbers. (If there are more than three, then all
    others are discarded.)
    """
    lst = ['', '', '']
    pos = 0
    for i in prop:
        if i.isnumeric() or i == '[' or i == ']':
            lst[pos] = lst[pos] + str(i)
        if i == ']':
            if pos == 2:
                break
            pos += 1      
    return lst[0], lst[1], lst[2]

def num_in_brackets(bracket):
    """
    Returns the number that is enclosed within brackets in num
    """
    num = ''
    for i in range(1,len(bracket)-1):
        num += bracket[i]
    return int(num)

def compute_logp(x, encoding_alphabet):
    """
    the items are in one-hot format
    """
    logp_lst = torch.zeros(x.shape[0])
    for i in range(len(logp_lst)):
        mol = x[i]
        prop = logp(hot_to_selfies(mol, encoding_alphabet))
        logp_lst[i] = prop
    return logp_lst
    
def compute_reg_loss(z, labels, reg_dim=0, gamma=1.0,factor=1.0):
    latent_dim = z[:, reg_dim]
    
    # compute latent distance matrix at regularized dimension
    latent_dim = latent_dim.view(-1, 1).repeat(1, latent_dim.shape[0])
    latent_dist_mat = (latent_dim - latent_dim.transpose(1,0)).view(-1, 1)
    
    # compute attribute distance matrix
    attribute = labels.view(-1, 1).repeat(1, labels.shape[0])
    attribute_dist_mat = (attribute - attribute.transpose(1,0)).view(-1, 1)
   
    #compute regularization loss
    criterion = torch.nn.L1Loss()
    latent_tanh = torch.tanh(latent_dist_mat*factor)
    
    attribute_sign = torch.sign(attribute_dist_mat)
    reg_loss = criterion(latent_tanh, attribute_sign.float())
    return gamma * reg_loss


def train_model(data_train, data_valid, num_epochs, latent_dimension, lr_enc, lr_dec, KLD_alpha, sample_num, use_reg_loss, encoding_alphabet):
    """
    Train the Variational Auto-Encoder
    """
    print('num_epochs: ',num_epochs)
    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam(model_encode.parameters(), lr=lr_enc)
    optimizer_decoder = torch.optim.Adam(model_decode.parameters(), lr=lr_dec)

    data_train = data_train.clone().detach()
    data_train=data_train.to(device)

    #print(data)
    quality_valid_list=[0,0,0,0];
    quality_valid_diff=[]
    num_deviates = 0
    if settings['evaluate']['evaluate_model']:
        num_epochs = 1
    for epoch in range(num_epochs):
        x = [i for i in range(len(data_train))]  # random shuffle input
        shuffle(x)

        data_train  = data_train[x]
        latent_points_combined = []
        output_mol_combined = [] # decoded molecules in string format
        output_mol_prop_combined = [] # attached properties
        input_mol_combined = [] # input molecules in string format
        input_mol_prop_combined = [] # attached properties
        start = time.time()
        for batch_iteration in range(num_batches_train):  # batch iterator
            
            loss, recon_loss, kld = 0., 0., 0.

            # manual batch iterations
            current_smiles_start, current_smiles_stop = batch_iteration * batch_size, (batch_iteration + 1) * batch_size 
            inp_smile_hot = data_train[current_smiles_start : current_smiles_stop]\
            
            # reshaping for efficient parallelization
            inp_smile_encode = inp_smile_hot.reshape(inp_smile_hot.shape[0], inp_smile_hot.shape[1] * inp_smile_hot.shape[2]) 
            latent_points, mus, log_vars = model_encode(inp_smile_encode)
            z = latent_points.clone()
            latent_points_combined.extend(z.detach().numpy())
            latent_points = latent_points.reshape(1, batch_size, latent_points.shape[1])
            # standard Kullback–Leibler divergence
            kld += -0.5 * torch.mean(1. + log_vars - mus.pow(2) - log_vars.exp()) 
            # initialization hidden internal state of RNN (RNN has two inputs and two outputs:)
            #    input: latent space & hidden state
            #    output: onehot encoding of one character of molecule & hidden state
            #    the hidden state acts as the internal memory
            hidden = model_decode.init_hidden(batch_size = batch_size)
                                                                       
            # decoding from RNN N times, where N is the length of the largest molecule (all molecules are padded)
            decoded_one_hot = torch.zeros(batch_size, inp_smile_hot.shape[1], inp_smile_hot.shape[2]).to(device) 
            
            for seq_index in range(inp_smile_hot.shape[1]):
                decoded_one_hot_line, hidden  = model_decode(latent_points, hidden)
                decoded_one_hot[:, seq_index, :] = decoded_one_hot_line[0]
                
            test_decoded_one_hot = decoded_one_hot.reshape(batch_size, inp_smile_hot.shape[1], inp_smile_hot.shape[2])
            decoded_one_hot = decoded_one_hot.reshape(batch_size * inp_smile_hot.shape[1], inp_smile_hot.shape[2])
            _, label_atoms  = inp_smile_hot.max(2)
            test_label_atoms = label_atoms.clone()
            
            if settings['evaluate']['evaluate_model']:
                # add the new batch of decoded molecules, in string format, to memory
                _, decoded_mol = test_decoded_one_hot.max(2)
                output_mol = []
                output_mol_prop = []
                for mol in decoded_mol:
                    mol, prop = split_mol(mol, encoding_alphabet)
                    mol = decoder(mol)
                    output_mol.append(mol)
                    output_mol_prop.append(prop)
                #print(output_mol[0], output_mol[1])
                output_mol_combined.extend(output_mol)
                output_mol_prop_combined.extend(output_mol_prop)
                
                # add the input molecules in string format to memory
                input_mol = []
                input_mol_prop = []
                for mol in test_label_atoms:
                    mol, prop = split_mol(mol, encoding_alphabet)
                    mol = decoder(mol)
                    input_mol.append(mol)
                    input_mol_prop.append(prop)
                
                #print(test_mol1, test_mol2)
                input_mol_combined.extend(input_mol)
                input_mol_prop_combined.extend(input_mol_prop)
                
            else:
            
                label_atoms     = label_atoms.reshape(batch_size * inp_smile_hot.shape[1])
                
                # we use cross entropy of expected symbols and decoded one-hot
                criterion   = torch.nn.CrossEntropyLoss()
                recon_loss += criterion(decoded_one_hot, label_atoms)
                loss += recon_loss + KLD_alpha * kld 
                
                if use_reg_loss:
                    labels = compute_logp(test_label_atoms, encoding_alphabet)
                    reg_loss = compute_reg_loss(z, labels)
                    loss += reg_loss                
                    #print(float(recon_loss.detach()),  float(reg_loss.detach()))
    
                # perform back propogation
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(model_decode.parameters(), 0.5)
                optimizer_encoder.step()
                optimizer_decoder.step()
    
                if batch_iteration % 30 == 0:     
                    end = time.time()     
                    
                    _, label_atoms_decoded = test_decoded_one_hot.max(2)
                    """
                    # print a few decoded molecules to visually test reconstruction
                    print('Training set decoded molecules:')
                    print(label_atoms_decoded[:2])
                    """
                    
                    flat_decoded = label_atoms_decoded.flatten()
                    flat_input = test_label_atoms.flatten()
                    equal_position_count = 0
                    num_position = 0
                    for mol in range(len(flat_decoded)):
                        if flat_decoded[mol] == flat_input[mol]:
                            equal_position_count += 1
                        num_position += 1
                    quality = equal_position_count / num_position *100
    
                    qualityValid=quality_in_validation_set(data_valid)
                    new_line = 'Epoch: %d,  Batch: %d / %d,\t(loss: %.4f\t| quality: %.4f | quality_valid: %.4f)\tELAPSED TIME: %.5f' % (epoch, batch_iteration, num_batches_train, loss.item(), quality, qualityValid, end - start)
                    print(new_line)
                    start = time.time()
        
        if settings['evaluate']['evaluate_model']:
            print(len(input_mol_combined), input_mol_combined[0])
            print(len(output_mol_combined), output_mol_combined[0])
            print(len(latent_points_combined), latent_points_combined[0])
                
        if settings['plot']['plot_latent']:
            latent_points_combined = np.array(latent_points_combined)
            print('Finding clusters...')
            # perform k-means clustering
            centers, labels = find_clusters(latent_points_combined, settings['kmeans']['num_clusters'])
            #print(len(labels), labels[:20])
            #print(len(centers), centers[0])
            print('Clusters found.')
            
            if settings['training_VAE']['latent_dimension'] == 2:
                df = pd.DataFrame(np.transpose((latent_points_combined[:,0],latent_points_combined[:,1])))
                df.columns = ['x', 'y']
                colormap = labels
                if settings['properties']['num_properties'] > 0:
                    mol_prop = np.array([])
                    for mol in input_mol_combined:
                        mol_prop = np.append(mol_prop, logp(encoder(mol)))
                    colormap = mol_prop
                
                print('Plotting projection...')
                plt.scatter(x=df['x'], y=df['y'], c=colormap,
                            cmap= 'viridis', marker='.',
                            s=10,alpha=0.5, edgecolors='none')
                plt.savefig('Results/{}_dimensions/VAE_{}-dataset_{}-clusters_{}-properties_{}-epochs_{}-version_{}-dimensions_plot_fig'.format(settings['training_VAE']['latent_dimension'], settings['data']['file_description'], settings['kmeans']['num_clusters'], settings['properties']['num_properties'], settings['training_VAE']['num_epochs'], version, settings['training_VAE']['latent_dimension']))   
                plt.show()
            elif settings['training_VAE']['latent_dimension'] == 3:
                df = pd.DataFrame(np.transpose((latent_points_combined[:,0],latent_points_combined[:,1], latent_points_combined[:,2])))
                df.columns = ['x', 'y', 'z']
                colormap = labels
                if settings['properties']['num_properties'] > 0:
                    mol_prop = np.array([])
                    for mol in input_mol_combined:
                        mol_prop = np.append(mol_prop, logp(encoder(mol)))
                    colormap = mol_prop
                
                print('Plotting projection...')
                fig=plt.figure()
                plot = fig.add_subplot(111, projection='3d')
                plot.scatter(df['x'], df['y'], df['z'], c=colormap, marker='.')
                plot.figure.savefig('Results/{}_dimensions/VAE_{}-dataset_{}-clusters_{}-properties_{}-epochs_{}-version_{}-dimensions_plot_fig'.format(settings['training_VAE']['latent_dimension'], settings['data']['file_description'], settings['kmeans']['num_clusters'], settings['properties']['num_properties'], settings['training_VAE']['num_epochs'], version, settings['training_VAE']['latent_dimension']))
            
            elif settings['training_VAE']['latent_dimension'] >= 4:
                df = pd.DataFrame(np.transpose((latent_points_combined[:,0],latent_points_combined[:,1], latent_points_combined[:,2], latent_points_combined[:,3])))
                df.columns = ['x', 'y', 'z', 'w']
                
                print('Plotting projection...')
                fig=plt.figure()
                plot = fig.add_subplot(111, projection='3d')
                plot.scatter(df['x'], df['y'], df['z'], c=df['w'], marker='.')
                plot.figure.savefig('Results/{}_dimensions/VAE_{}-dataset_{}-clusters_{}-properties_{}-epochs_{}-version_{}-dimensions_plot_fig'.format(settings['training_VAE']['latent_dimension'], settings['data']['file_description'], settings['kmeans']['num_clusters'], settings['properties']['num_properties'], settings['training_VAE']['num_epochs'], version, settings['training_VAE']['latent_dimension']))
            cluster_to_mol = {}
            for i in range(len(labels)):
                label = labels[i]
                input_mol = input_mol_combined[i]
                input_mol_prop = input_mol_prop_combined[i]
                output_mol = output_mol_combined[i]
                output_mol_prop = output_mol_prop_combined[i]
                
                if label not in cluster_to_mol:
                    cluster_to_mol[label] = [centers[label], [], [], [], []]
                    
                cluster_to_mol[label][1].append(input_mol)
                cluster_to_mol[label][2].append(output_mol)
                cluster_to_mol[label][3].append(input_mol_prop)
                cluster_to_mol[label][4].append(output_mol_prop)
                
            save = input("Enter 'y' to save plot and results to file.")
            if save == 'y':
                f = open('Results/{}_dimensions/VAE_{}-dataset_{}-clusters_{}-properties_{}-epochs_{}-version_{}-dimensions_plot_fig_results'.format(settings['training_VAE']['latent_dimension'], settings['data']['file_description'], settings['kmeans']['num_clusters'], settings['properties']['num_properties'], settings['training_VAE']['num_epochs'], version, settings['training_VAE']['latent_dimension']), "w+")
            for label in cluster_to_mol:
                print('LABEL: '+str(label))
                print('CENTER: '+str(cluster_to_mol[label][0]))
                print('INPUT MOLECULES:')
                for i in range(len(cluster_to_mol[label][1])):
                    mol = cluster_to_mol[label][1][i]
                    prop = split_prop(cluster_to_mol[label][3][i])[1]
                    print('\t'+mol+prop)
                print('OUTPUT MOLECULES:')
                for i in range(len(cluster_to_mol[label][2])):
                    mol = cluster_to_mol[label][2][i]
                    prop = split_prop(cluster_to_mol[label][4][i])[1]
                    print('\t'+mol+prop)
                print('\n\n')
                if save == 'y':
                    f.write('LABEL: '+str(label)+'\n')
                    f.write('CENTER: '+str(cluster_to_mol[label][0])+'\n')
                    f.write('INPUT MOLECULES:\n')
                    for i in range(len(cluster_to_mol[label][1])):
                        mol = cluster_to_mol[label][1][i]
                        prop = split_prop(cluster_to_mol[label][3][i])[1]
                        f.write('\t'+ mol+prop+'\n')
                    f.write('OUTPUT MOLECULES:\n')
                    for i in range(len(cluster_to_mol[label][2])):
                        mol = cluster_to_mol[label][2][i]
                        prop = split_prop(cluster_to_mol[label][4][i])[1]
                        f.write('\t'+ mol+prop+'\n')
                    f.write('\n\n')
            if save == 'y':
                f.close()    
        
        if settings['plot']['plot_PCA']:
            if epoch  % 10 == 0:
                print('PCA Projection:')
                Z_pca = PCA(n_components=2).fit_transform(latent_points_combined)
                
                #print(len(Z_pca), Z_pca[0])
                
                # preprocessing step to standardize the data by scaling features
                # to lie between 0 and 1 (allows robustness to small standard
                # deviations of features)
                #Z_pca = MinMaxScaler().fit_transform(Z_pca)
                
                df = pd.DataFrame(np.transpose((Z_pca[:,0],Z_pca[:,1])))
                df.columns = ['x','y']
                
                if settings['properties']['num_properties'] > 0:
                    mol_prop = np.array([])
                    for mol in input_mol_combined:
                        mol_prop = np.append(mol_prop, logp(encoder(mol)))
                    labels = mol_prop
                    print(labels)
                    plt.scatter(x=df['x'], y=df['y'], c=labels,
                                cmap= plt.cm.RdYlGn, marker='.',
                                s=10,alpha=0.5, edgecolors='none')
                    plt.savefig('Results/{}_dimensions/VAE_{}-dataset_{}-properties_{}-epochs_{}-version_{}_dimensions_pca_fig'.format(settings['training_VAE']['latent_dimension'], settings['data']['file_description'], settings['properties']['num_properties'], settings['training_VAE']['num_epochs'], version, settings['training_VAE']['latent_dimension']))  
                    plt.show()
                
                else:
                    plt.scatter(x=df['x'], y=df['y'],
                                cmap= 'viridis', marker='.',
                                s=10,alpha=0.5, edgecolors='none')
                    plt.savefig('Results/{}_dimensions/VAE_{}-dataset_{}-properties_{}-epochs_{}-version_{}_dimensions_pca_fig'.format(settings['training_VAE']['latent_dimension'], settings['data']['file_description'], settings['properties']['num_properties'], settings['training_VAE']['num_epochs'], version, settings['training_VAE']['latent_dimension']))  
                    plt.show()
                
        if settings['plot']['plot_tSNE']:
            if epoch % 20 == 0:
                print('tSNE projection:')
                n_comp = settings['tSNE']['n_components']
                perplexity = settings['tSNE']['perplexity']
                # see https://distill.pub/2016/misread-tsne/ for hyperparameter tuning tips
                Z_tsne = TSNE(n_components=n_comp, perplexity=perplexity).fit_transform(latent_points_combined)
                print('Projection finished.')
                
                #print(len(Z_tsne), Z_tsne[0])
                
                # preprocessing step to standardize the data by scaling features
                # to lie between 0 and 1 (allows robustness to small standard
                # deviations of features)
                #Z_tsne = MinMaxScaler().fit_transform(Z_tsne)
                if not settings['evaluate']['evaluate_model']:
                    if n_comp == 2:
                        df = pd.DataFrame(np.transpose((Z_tsne[:,0],Z_tsne[:,1])))
                        df.columns = ['x','y']
                        
                        print('Plotting projection...')
                        plt.scatter(x=df['x'], y=df['y'],
                                    cmap= plt.cm.RdYlGn, marker='.',
                                    s=10,alpha=0.5, edgecolors='none')
                        plt.show()
                        
                    elif n_comp == 3:
                        df = pd.DataFrame(np.transpose((Z_tsne[:,0],Z_tsne[:,1],Z_tsne[:,2])))
                        df.columns = ['x', 'y', 'z']
                        
                        print('Plotting projection...')
                        fig=plt.figure()
                        plot = fig.add_subplot(111, projection='3d')
                        plot.scatter(df['x'], df['y'], df['z'], marker='.')
                else:
                    num_clusters = settings['kmeans']['num_clusters']
                    if n_comp == 2:
                        df = pd.DataFrame(np.transpose((Z_tsne[:,0],Z_tsne[:,1])))
                        df.columns = ['x','y']
                        
                        print('Finding clusters...')
                        # perform k-means clustering
                        centers, labels = find_clusters(Z_tsne, num_clusters)
                        
                        #print(len(labels), labels[:20])
                        #print(len(centers), centers[0])
                        print('Clusters found.')
                        colormap = labels
                        if settings['properties']['num_properties'] > 0:
                            mol_prop = np.array([])
                            for mol in input_mol_combined:
                                mol_prop = np.append(mol_prop, logp(encoder(mol)))
                            colormap = mol_prop
                        
                        print('Plotting projection...')
                        plt.scatter(x=df['x'], y=df['y'], c=colormap,
                                    cmap= plt.cm.RdYlGn, marker='.',
                                    s=10,alpha=0.5, edgecolors='none')
                        plt.savefig('Results/{}_dimensions/VAE_{}-dataset_{}-comp_{}-perplexity_{}-clusters_{}-properties_{}-epochs_{}-version_{}-dimensions_tsne_fig'.format(settings['training_VAE']['latent_dimension'], settings['data']['file_description'], n_comp, perplexity, num_clusters, settings['properties']['num_properties'], settings['training_VAE']['num_epochs'], version, settings['training_VAE']['latent_dimension']))   
                        plt.show()
                    
                    elif n_comp == 3:
                        df = pd.DataFrame(np.transpose((Z_tsne[:,0],Z_tsne[:,1],Z_tsne[:,2])))
                        df.columns = ['x', 'y', 'z']
                        
                        print('Finding clusters...')
                        # perform k-means clustering
                        centers, labels = find_clusters(Z_tsne, num_clusters)
                        
                        #print(len(labels), labels[:20])
                        #print(len(centers), centers[0])
                        print('Clusters found.')
                        colormap = labels
                        if settings['properties']['num_properties'] > 0:
                            mol_prop = np.array([])
                            for mol in input_mol_combined:
                                mol_prop = np.append(mol_prop, logp(encoder(mol)))
                            colormap = mol_prop
                        
                        print('Plotting projection...')
                        fig=plt.figure()
                        plot = fig.add_subplot(111, projection='3d')
                        plot.scatter(df['x'], df['y'], df['z'], c=colormap, marker='.')
                        plot.figure.savefig('Results/{}_dimensions/VAE_{}-dataset_{}-comp_{}-perplexity_{}-clusters_{}-properties_{}-epochs_{}-version_{}-dimensions_tsne_fig'.format(settings['training_VAE']['latent_dimension'], settings['data']['file_description'], n_comp, perplexity, num_clusters, settings['properties']['num_properties'], settings['training_VAE']['num_epochs'], version, settings['training_VAE']['latent_dimension']))
                    
                    cluster_to_mol = {}
                    for i in range(len(labels)):
                        label = labels[i]
                        input_mol = input_mol_combined[i]
                        input_mol_prop = input_mol_prop_combined[i]
                        output_mol = output_mol_combined[i]
                        output_mol_prop = output_mol_prop_combined[i]
                        
                        if label not in cluster_to_mol:
                            cluster_to_mol[label] = [centers[label], [], [], [], []]
                            
                        cluster_to_mol[label][1].append(input_mol)
                        cluster_to_mol[label][2].append(output_mol)
                        cluster_to_mol[label][3].append(input_mol_prop)
                        cluster_to_mol[label][4].append(output_mol_prop)
                        
                    save = input("Enter 'y' to save tSNE proj and results to file.")
                    if save == 'y':
                        f = open('Results/{}_dimensions/VAE_{}-dataset_{}-comp_{}-perplexity_{}-clusters_{}-properties_{}-epochs_{}-version_{}-dimensions_tsne_results'.format(settings['training_VAE']['latent_dimension'], settings['data']['file_description'], n_comp, perplexity, num_clusters, settings['properties']['num_properties'], settings['training_VAE']['num_epochs'], version, settings['training_VAE']['latent_dimension']), "w+")
                    for label in cluster_to_mol:
                        print('LABEL: '+str(label))
                        print('CENTER: '+str(cluster_to_mol[label][0]))
                        print('INPUT MOLECULES:')
                        for i in range(len(cluster_to_mol[label][1])):
                            mol = cluster_to_mol[label][1][i]
                            prop = split_prop(cluster_to_mol[label][3][i])[1]
                            print('\t'+mol+prop)
                        print('OUTPUT MOLECULES:')
                        for i in range(len(cluster_to_mol[label][2])):
                            mol = cluster_to_mol[label][2][i]
                            prop = split_prop(cluster_to_mol[label][4][i])[1]
                            print('\t'+mol+prop)
                        print('\n\n')
                        if save == 'y':
                            f.write('LABEL: '+str(label)+'\n')
                            f.write('CENTER: '+str(cluster_to_mol[label][0])+'\n')
                            f.write('INPUT MOLECULES:\n')
                            for i in range(len(cluster_to_mol[label][1])):
                                mol = cluster_to_mol[label][1][i]
                                prop = split_prop(cluster_to_mol[label][3][i])[1]
                                f.write('\t'+ mol+prop+'\n')
                            f.write('OUTPUT MOLECULES:\n')
                            for i in range(len(cluster_to_mol[label][2])):
                                mol = cluster_to_mol[label][2][i]
                                prop = split_prop(cluster_to_mol[label][4][i])[1]
                                f.write('\t'+ mol+prop+'\n')
                            f.write('\n\n')
                    if save == 'y':
                        f.close()
                        
        if settings['evaluate']['evaluate_interp']:
        #some attributes to try:
        # - add # or add a beginning # (index second last) or ending # (second index)
        # - add =NO at the end, or ON= at the beginning
        # - add ()
        # - add both # and ()
        # - add F or N or both F and N
        # - add =
            
            attribute_to_dimension(latent_points_combined, input_mol_combined, 
                                   output_mol_combined, 
                                   "'#' in mol", "'#' not in mol", 
                                   "'#' in mol", "'#' not in mol",
                                   'right-hand triple bond', 
                                   'Presence of triple bonds',
                                   'right-hand triple bond',
                                   'CC1=NNN=N1', 5, 30)
            attribute_to_dimension(latent_points_combined, input_mol_combined, 
                                   output_mol_combined, 
                                   "mol.count('C') > 4", "mol.count('C') <= 4", 
                                   "mol.count('C') > 4", "mol.count('C') <= 4",
                                   'number of carbons', 
                                   'Number of carbons',
                                   'carbons',
                                   'CC1=NNN=N1', 5, 30)
            attribute_to_dimension(latent_points_combined, input_mol_combined, 
                                   output_mol_combined, 
                                   "'2' in mol or '3' in mol", "'1' in mol", 
                                   "'1' in mol", "'1' not in mol",
                                   'ring structure complexity', 
                                   'Ring structure complexity', 
                                   'ring structure complexity',
                                   'O=CC12CC(CC1)O2', 5, 1)
            
            attribute_to_dimension(latent_points_combined, input_mol_combined, 
                                   output_mol_combined, 
                                   "len(mol)>3 and mol[3] == '=' and (mol[-4]== '=' or mol[-3]== '=') and mol[2]== '1' and (mol[-1]=='1' or mol[-2]=='1')", 
                                   "'1' not in mol and '=' not in mol", 
                                   "'1' in mol and '=' in mol", 
                                   "'1' not in mol or '=' not in mol", 
                                   'double-bond and ring structure complexity', 
                                   'Ring, double-bond', 
                                   'double-bond and ring structure complexity',
                                   'CCCCCCCCCC', 5, 1)
        
        if settings['evaluate']['evaluate_dim_v1']:
            num_dim = settings['training_VAE']['latent_dimension']
            num_mol=1000
            rank = 1 # Rank 1 means that only the highest dimension is considered; rank 2 means that the highest and second-highest are both considered, etc
            high_dim_to_mol = {}
            low_dim_to_mol = {}
            for i in range(num_mol):
                input_mol = input_mol_combined[i]
                max_dim = int(remove_trivial_dim(latent_points_combined[i], num_dim)[1])
                min_dim = int(remove_trivial_dim(latent_points_combined[i], num_dim)[2])
                if max_dim in high_dim_to_mol:
                    high_dim_to_mol[max_dim].append(input_mol)
                else:
                    high_dim_to_mol[max_dim] = [input_mol]
                if min_dim in low_dim_to_mol:
                    low_dim_to_mol[min_dim].append(input_mol)
                else:
                    low_dim_to_mol[min_dim] = [input_mol]
                if rank >= 2:
                    latent2 = np.zeros([num_dim-1])
                    for j in range(num_dim):
                        if j < max_dim:
                            latent2[j] = latent_points_combined[i][j]
                        if j > max_dim:
                            latent2[j-1] = latent_points_combined[i][j]
                            
                    max_dim = int(remove_trivial_dim(latent2, num_dim-1)[1])
                    if max_dim in high_dim_to_mol:
                        high_dim_to_mol[max_dim].append(input_mol)
                    else:
                        high_dim_to_mol[max_dim] = [input_mol]
                if rank >= 3:
                    latent3 = np.zeros([num_dim-2])
                    for j in range(num_dim-1):
                        if j < max_dim:
                            latent3[j] = latent2[j]
                        if j > max_dim:
                            latent3[j-1] = latent2[j]
                    max_dim = int(remove_trivial_dim(latent3, num_dim-2)[1])
                    if max_dim in high_dim_to_mol:
                        high_dim_to_mol[max_dim].append(input_mol)
                    else:
                        high_dim_to_mol[max_dim] = [input_mol]
            
            save = input("Enter 'y' to save highest dimension analysis results to file.")
            if save == 'y':
                f = open('Results/{}_dimensions/VAE_{}-dataset_{}-epochs_{}-version_{}-properties_{}-molecules_{}-rank_{}-dimensions_highest_results'.format(settings['training_VAE']['latent_dimension'], settings['data']['file_description'], settings['training_VAE']['num_epochs'], version, settings['properties']['num_properties'], num_mol, rank, settings['training_VAE']['latent_dimension']), "w+")
                f.write('There are ' + str(len(high_dim_to_mol)) + ' dimensions described in this file.\n\n')
            for dim in high_dim_to_mol:
                print('DIMENSION: '+str(dim))
                print('INPUT MOLECULES HIGHEST IN THIS DIMENSION:')
                for mol in high_dim_to_mol[dim]:
                    print('\t'+mol)
                """
                print('INPUT MOLECULES LOWEST IN THIS DIMENSION:')
                for mol in low_dim_to_mol[dim]:
                    print('\t'+mol)
                """
                print('\n\n')
                if save =='y':
                    f.write('DIMENSION: '+str(dim)+'\n')
                    f.write('INPUT MOLECULES HIGHEST IN THIS DIMENSION:\n')
                    for mol in high_dim_to_mol[dim]:
                        f.write('\t'+mol+'\n')
                    """
                    f.write('INPUT MOLECULES LOWEST IN THIS DIMENSION:\n')
                    for mol in low_dim_to_mol[dim]:
                        f.write('\t'+mol+'\n')
                    """
                    f.write('\n\n')
            if save=='y':
                f.close()
            save = input("Enter 'y' to save lowest dimension analysis results to file.")
            if save == 'y':
                f = open('Results/{}_dimensions/VAE_{}-dataset_{}-epochs_{}-version_{}-properties_{}-molecules_{}-rank_{}-dimensions_lowest_results'.format(settings['training_VAE']['latent_dimension'], settings['data']['file_description'], settings['training_VAE']['num_epochs'], version, settings['properties']['num_properties'], num_mol, rank, settings['training_VAE']['latent_dimension']), "w+")
                f.write('There are ' + str(len(low_dim_to_mol)) + ' dimensions described in this file.\n\n')
            for dim in low_dim_to_mol:
                print('DIMENSION: ' +str(dim))
                print('INPUT MOLECULES LOWEST IN THIS DIMENSION:')
                for mol in low_dim_to_mol[dim]:
                    print('\t'+mol)   
                print('\n\n')
                if save =='y':
                    f.write('DIMENSION: '+str(dim)+'\n')
                    f.write('INPUT MOLECULES LOWEST IN THIS DIMENSION:\n')
                    for mol in low_dim_to_mol[dim]:
                        f.write('\t'+mol+'\n')
                    f.write('\n\n')
            if save=='y':
                f.close()
            
            dim_to_attributeVec = {}
            for dim in high_dim_to_mol:
                if dim in low_dim_to_mol:
                    high_mol = high_dim_to_mol[dim]
                    low_mol = low_dim_to_mol[dim]
                    num_high = len(high_mol)
                    num_low = len(low_mol)
                    sum_high = np.zeros([settings['training_VAE']['latent_dimension']])
                    sum_low = np.zeros([settings['training_VAE']['latent_dimension']])
                    for mol in high_mol:
                        sum_high += latent_points_combined[input_mol_combined.index(mol)]
                    for mol in low_mol:
                        sum_low += latent_points_combined[input_mol_combined.index(mol)]
                    
                    dim_to_attributeVec[dim] = sum_high/num_high - sum_low/num_low
            
            test_mol = 'CCCCCC'
            steps = 5
            weight = 30
            save = input("Enter 'y' to save dimension to attribute vector conversion results to file.")
            if save == 'y':
                f = open('Results/{}_dimensions/VAE_{}-dataset_{}-epochs_{}-version_{}-properties_{}-molecules_{}-rank_{}-dimensions_{}-step_{}-weight_attribute_vector_{}_results'.format(settings['training_VAE']['latent_dimension'], settings['data']['file_description'], settings['training_VAE']['num_epochs'], version, settings['properties']['num_properties'], num_mol, rank, settings['training_VAE']['latent_dimension'], steps, weight, test_mol), "w+")
                f.write('There are ' + str(len(dim_to_attributeVec)) + ' dimensions described in this file.\n')
                f.write('For each dimension, attribute vectors are computed as the difference between the averages of the latent vectors for molecules highest in the dimension and the latent vectors for molecules lowest in the dimension.\n')
                f.write('Molecule to test computed attribute vector: '+test_mol+'\n\n')
            for dim in dim_to_attributeVec:
                attribute_vector = dim_to_attributeVec[dim]
                print('DIMENSION: '+str(dim))
                print('Attribute vector: '+ str(attribute_vector))
                latent, max_index, min_index = remove_trivial_dim(attribute_vector, len(attribute_vector))
                print('Attribute vector maximum dimension: '+ str(int(max_index)))
                print('Attribute vector minimum dimension: '+str(int(min_index)))
                print('Reduced attribute vector: '+str(latent))
                print('Attribute added to '+test_mol+': '+str(add_attribute(encoder(test_mol), attribute_vector, steps, weight)))
                print('\n\n')
                if save == 'y':
                    f.write('DIMENSION: '+str(dim)+'\n')
                    f.write('Attribute vector: '+ str(attribute_vector)+'\n')
                    f.write('Attribute vector maximum dimension: '+ str(max_index)+'\n')
                    f.write('Attribute vector minimum dimension: '+str(min_index)+'\n')
                    f.write('Reduced attribute vector: '+str(latent)+'\n')
                    f.write('Attribute added to '+test_mol+': '+str(add_attribute(encoder(test_mol), attribute_vector, steps, weight))+'\n')
                    f.write('\n\n')
            if save=='y':
                f.close()
                    
        
        if settings['evaluate']['evaluate_dim_v2']:
            dim_hierarchy = []
            dim_correlation = []
            num_mol = 30
            test_mol1 = 'CCC1CC1O'
            test_mol2 = 'NC1=NON=N1'
            test_mol3 = 'CC(C)(O)CCO'
            test_mol4 = 'OC1CC2CC12O'
            test_mol5 = 'CCCCCCC'
            test_mol6 = 'O=C1OCC11CO1'
            test_mol7 = 'CC12C3C4C(C13)N24'
            test_mol8 = 'COC1CCC1=NO'
            steps = 10
            weight_attribute = 1
            weight_dim = 10
        
            for i in range(settings['training_VAE']['latent_dimension']):
                dim_lst = ([], i)
                for j in range(len(latent_points_combined)):
                    latent = latent_points_combined[j]
                    dim_lst[0].append((latent[i], j))
                dim_lst[0].sort(reverse=True)
                dim_hierarchy.append(dim_lst)
            
            if settings['properties']['num_properties'] > 0:
                lst = dim_hierarchy[0][0]
                prop_lst = []
                for item in lst:
                    prop_lst.append(logp(encoder(input_mol_combined[item[1]])))
                lst = np.array(lst)[:,0]
                prop_lst = np.array(prop_lst)
                corr = stats.pearsonr(lst, prop_lst)[0]
                plt.scatter(x=lst, y=prop_lst, marker='.',
                            s=10,alpha=0.5, edgecolors='none')
                plt.savefig('Results/{}_dimensions/VAE_{}-dataset_{}-properties_{}-epochs_{}-version_{}-dimensions_property-dimension-correlation_fig'.format(settings['training_VAE']['latent_dimension'], settings['data']['file_description'], settings['properties']['num_properties'], settings['training_VAE']['num_epochs'], version, settings['training_VAE']['latent_dimension']))   
                plt.show()

                    
            
            save = input("Enter 'y' to save hierarchical dimension analysis results to file.")
            if save == 'y':
                f = open('Results/{}_dimensions/VAE_{}-dataset_{}-epochs_{}-version_{}-properties_{}-molecules_{}-dimensions_{}-step_{}-weight(attribute)_{}-weight(dimension)_hierarchical_{}_results'.format(settings['training_VAE']['latent_dimension'], settings['data']['file_description'], settings['training_VAE']['num_epochs'], version, settings['properties']['num_properties'], num_mol, settings['training_VAE']['latent_dimension'], steps, weight_attribute, weight_dim, test_mol1), "w+")
            
            for i in range(len(dim_hierarchy)):
                lst = dim_hierarchy[i][0]
                dim = dim_hierarchy[i][1]
                print('DIMENSION: '+ str(dim))
                print('THE '+ str(num_mol) +' HIGHEST MOLECULES IN THIS DIMENSION:')
                sum_high = np.zeros([settings['training_VAE']['latent_dimension']])
                sum_low = np.zeros([settings['training_VAE']['latent_dimension']])
                for j in range(num_mol):
                    print('\t'+input_mol_combined[lst[j][1]])
                    sum_high += latent_points_combined[lst[j][1]]
                print('THE '+str(num_mol) +' LOWEST MOLECULES IN THIS DIMENSION:')
                for j in range(len(lst)-1, len(lst)-num_mol-1, -1):
                    print('\t'+input_mol_combined[lst[j][1]])
                    sum_low += latent_points_combined[lst[j][1]]
                attribute_vector = sum_high/num_mol - sum_low/num_mol
                dimension_vector = np.zeros([settings['training_VAE']['latent_dimension']])
                dimension_vector[i] = attribute_vector[i]
                #print('ATTRIBUTE VECTOR: '+str(attribute_vector))
                print_vector_analysis(test_mol1, attribute_vector, dimension_vector, steps, weight_attribute, weight_dim)
                print_vector_analysis(test_mol2, attribute_vector, dimension_vector, steps, weight_attribute, weight_dim)
                print_vector_analysis(test_mol3, attribute_vector, dimension_vector, steps, weight_attribute, weight_dim)
                print_vector_analysis(test_mol4, attribute_vector, dimension_vector, steps, weight_attribute, weight_dim)
                print_vector_analysis(test_mol5, attribute_vector, dimension_vector, steps, weight_attribute, weight_dim)
                print_vector_analysis(test_mol6, attribute_vector, dimension_vector, steps, weight_attribute, weight_dim)
                print_vector_analysis(test_mol7, attribute_vector, dimension_vector, steps, weight_attribute, weight_dim)
                print_vector_analysis(test_mol8, attribute_vector, dimension_vector, steps, weight_attribute, weight_dim)
    
                print('\n\n')
                if save == 'y':
                    f.write('DIMENSION: '+ str(dim)+'\n')
                    f.write('THE '+ str(num_mol) +' HIGHEST MOLECULES IN THIS DIMENSION:\n')
                    for j in range(num_mol):
                        f.write('\t'+input_mol_combined[lst[j][1]]+'\n')
                    f.write('THE '+str(num_mol) +' LOWEST MOLECULES IN THIS DIMENSION:\n')
                    for j in range(len(lst)-1, len(lst)-num_mol-1, -1):
                        f.write('\t'+input_mol_combined[lst[j][1]]+'\n')
                    #f.write('ATTRIBUTE VECTOR: '+str(attribute_vector)+'\n')
                    write_vector_analysis(f, test_mol1, attribute_vector, dimension_vector, steps, weight_attribute, weight_dim)
                    write_vector_analysis(f, test_mol2, attribute_vector, dimension_vector, steps, weight_attribute, weight_dim)
                    write_vector_analysis(f, test_mol3, attribute_vector, dimension_vector, steps, weight_attribute, weight_dim)
                    write_vector_analysis(f, test_mol4, attribute_vector, dimension_vector, steps, weight_attribute, weight_dim)
                    write_vector_analysis(f, test_mol5, attribute_vector, dimension_vector, steps, weight_attribute, weight_dim)
                    write_vector_analysis(f, test_mol6, attribute_vector, dimension_vector, steps, weight_attribute, weight_dim)
                    write_vector_analysis(f, test_mol7, attribute_vector, dimension_vector, steps, weight_attribute, weight_dim)
                    write_vector_analysis(f, test_mol8, attribute_vector, dimension_vector, steps, weight_attribute, weight_dim)
                    f.write('\n\n')
            """
            if settings['properties']['num_properties']>0:
                
                print('DIMENSIONS RANKED BY THEIR CORRELATION WITH LOGP (with format DIMENSION, CORR COEFFICIENT):')
                for item in dim_prop_corr:
                    dim = item[1]
                    corr = item[0]
                    print('\t'+str(dim)+', '+str(corr))
                if save == 'y':
                    f.write('DIMENSIONS RANKED BY THEIR CORRELATION WITH LOGP (with format DIMENSION, CORR COEFFICIENT):\n')
                    for item in dim_prop_corr:
                        dim = item[1]
                        corr = item[0]
                        f.write('\t'+str(dim)+', '+str(corr)+'\n')
            """
            if save=='y':
                f.close()
                
        if settings['evaluate']['evaluate_dim_corr']:
            dim_correlation = []
            num_mol = 10
            num_over = 0.0
            sum_corr = 0.0
            num_corr = 0
            for i in range(settings['training_VAE']['latent_dimension']):
                corr_lst = ([], i)
                dim_lst_i = np.array(latent_points_combined)[:, i]
                for j in range(settings['training_VAE']['latent_dimension']-1):
                    if j != i:
                        dim_lst_j = np.array(latent_points_combined)[:, j]
                        corr = stats.pearsonr(dim_lst_i, dim_lst_j)[0] 
                        corr_lst[0].append((corr, j))
                        if j > i:
                            sum_corr += abs(corr)
                            num_corr += 1
                            if abs(corr) > 0.7:
                                num_over += 1
                    
                corr_lst[0].sort(reverse=True)
                dim_correlation.append(corr_lst)
            if num_corr == 0:
                avg_corr = 'NA'
                percent_over = 'NA'
            else:
                avg_corr = sum_corr / num_corr
                percent_over = num_over/num_corr*100
            
            groups = []
            grouped = []
            for i in range(len(dim_correlation)):
                print(groups)
                group = [dim_correlation[i][1]]
                for corr in dim_correlation[i][0]:
                    if abs(corr[0]) > 0.7:
                        group.append(corr[1])
                        if not corr[1] in grouped:
                            grouped.append(corr[1])
                if len(group) > 1:
                    grouped.append(dim_correlation[i][1])
                    groups.append(group)
            ungrouped = []
            for i in range(len(dim_correlation)):
                if not i in grouped:
                    ungrouped.append(i)
            
            save = input("Enter 'y' to save dimension correlation analysis results to file.")
            if save == 'y':
                f = open('Results/{}_dimensions/VAE_{}-dataset_{}-epochs_{}-version_{}-properties_{}-molecules_{}-dimensions_correlation_results'.format(settings['training_VAE']['latent_dimension'], settings['data']['file_description'], settings['training_VAE']['num_epochs'], version, settings['properties']['num_properties'], num_mol, settings['training_VAE']['latent_dimension']), "w+")
                f.write('Average of the absolute value of all computed coefficients: '+str(avg_corr)+'\n')
                f.write('Percentage of computed coefficients over 0.7: %'+str(percent_over)+'\n')
                f.write('Format: Input molecule, correlation coefficient\n')
            for i in range(len(dim_correlation)):
                lst = dim_correlation[i][0]
                dim = dim_correlation[i][1]
                
                print('DIMENSION: '+ str(dim))
                print('THE '+str(num_mol)+' MOST POSITIVELY CORRELATED DIMENSIONS:')
                for j in range(num_mol):
                    print('\t'+str(lst[j][1])+', '+str(lst[j][0]))
                print('THE '+str(num_mol)+' MOST NEGATIVELY CORRELATED DIMENSIONS:')
                for j in range(len(lst)-1, len(lst)-num_mol-1, -1):
                    print('\t'+str(lst[j][1])+', '+str(lst[j][0]))
                centre = 0
                while centre<len(lst) and lst[centre][0]>=0.0:
                    centre+=1
                left = 0
                right = len(lst)
                if left < (centre - num_mol//2):
                    left = centre - num_mol//2
                if right > (centre + num_mol//2):
                    right = centre +num_mol//2
                    
                print('THE '+str(num_mol)+' LEAST CORRELATED DIMENSIONS:')
                for j in range(left, right):
                    print('\t'+str(lst[j][1])+', '+str(lst[j][0]))
                
                print('\n\n')
                if save == 'y':
                    f.write('DIMENSION: '+ str(dim)+'\n')
                    f.write('THE '+str(num_mol)+' MOST POSITIVELY CORRELATED DIMENSIONS:\n')
                    for j in range(num_mol):
                        f.write('\t'+str(lst[j][1])+', '+str(lst[j][0])+'\n')
                    f.write('THE '+str(num_mol)+' MOST NEGATIVELY CORRELATED DIMENSIONS:\n')
                    for j in range(len(lst)-1, len(lst)-num_mol-1, -1):
                        f.write('\t'+str(lst[j][1])+', '+str(lst[j][0])+'\n')
                    f.write('THE '+str(num_mol)+' LEAST CORRELATED DIMENSIONS:\n')
                    for j in range(left, right):
                        f.write('\t'+str(lst[j][1])+', '+str(lst[j][0]) +'\n')
                    f.write('\n\n')
                
            print('\n\n')
            print('Dimensions grouped by correlation: ')
            for i in range(len(groups)):
                group = groups[i]
                print('Group '+str(i+1)+': ')
                for dim in group:
                    print(dim)
                print(str(len(group))+' in total')
                print('\n\n')
            print('No group / Low correlation with other dimensions:')
            for dim in ungrouped:
                print(dim)
            print(str(len(ungrouped))+' in total')
            
            if save=='y':
                f.write('\n\n')
                f.write('Dimensions grouped by correlation:\n')
                for i in range(len(groups)):
                    group = groups[i]
                    f.write('Group '+str(i+1)+':\n')
                    for dim in group:
                        f.write(str(dim)+'\n')
                    f.write(str(len(group))+' in total\n')
                    f.write('\n\n')
                f.write('No group / Low correlation with other dimensions:\n')
                for dim in ungrouped:
                    f.write(str(dim)+'\n')
                f.write(str(len(ungrouped))+' in total\n')
                f.close()
            
        if not settings['evaluate']['evaluate_model'] and settings['plot']['plot_quality']:
            recons_quality_valid.append(qualityValid)
            recons_quality_train.append(quality)
            
        if not settings['evaluate']['evaluate_model'] and settings['plot']['plot_loss']:
            recons_loss.append(loss.item())
        
        if not settings['evaluate']['evaluate_model'] and settings['evaluate']['evaluate_metrics']:
            qualityValid = quality_in_validation_set(data_valid)
            
            if len(quality_valid_diff) > 30:
                quality_valid_diff.pop(0)
            quality_valid_diff.append(qualityValid - quality_valid_list[len(quality_valid_list)-1])
            quality_valid_list.append(qualityValid)
    
            # only measure validity of reconstruction improved
            quality_increase = len(quality_valid_list) - np.argmax(quality_valid_list)
            if quality_increase == 1 and quality_valid_list[-1] > 50.:
                corr, unique = latent_space_quality(latent_dimension,sample_num = sample_num, encoding_alphabet=encoding_alphabet)
            else:
                corr, unique = -1., -1.
    
            new_line = 'Validity: %.5f %% | Diversity: %.5f %% | Reconstruction: %.5f %%' % (corr * 100. / sample_num, unique * 100. / sample_num, qualityValid)
    
            print(new_line)
            with open('results.dat', 'a') as content:
                content.write(new_line + '\n')
    
            if quality_valid_list[-1] < 70. and epoch > 200:
                break
            
            if qualityValid < quality - 5:
                num_deviates += 1
                
            if num_deviates == 10:
                print('Early stopping criteria: validation set quality deviates from training set quality')
                break


# resource: http://krasserm.github.io/2018/07/27/dfc-vae/
def linear_interpolation(x_from, x_to, steps):
    n = steps + 1
    hot = multiple_selfies_to_hot([x_from], largest_molecule_len, encoding_alphabet)
    x_from = torch.tensor(hot, dtype=torch.float).to(device)
    input_shape1 = x_from.shape[1]
    input_shape2 = x_from.shape[2]
    x_from = x_from.reshape(x_from.shape[0], x_from.shape[1] * x_from.shape[2])
    _, hot = selfies_to_hot(x_to, largest_molecule_len, encoding_alphabet)
    x_to = torch.tensor([hot], dtype=torch.float).to(device)
    x_to = x_to.reshape(x_to.shape[0], x_to.shape[1] * x_to.shape[2])
    t_from = model_encode(x_from)[0]
    t_from = t_from.reshape(1, 1, t_from.shape[1])
    t_to = model_encode(x_to)[0]
    t_to = t_to.reshape(1, 1, t_to.shape[1])
    diff = t_to[0][0] - t_from[0][0]
    inter = torch.zeros((1, n, t_to.shape[2]))
    for i in range(n):
        inter[0][i] = t_from[0][0] + i / steps * diff
        
    hidden = model_decode.init_hidden(batch_size=n)
    
    decoded_one_hot = torch.zeros(n, input_shape1, input_shape2).to(device)
    for seq_index in range(input_shape1):
        decoded_one_hot_line, hidden  = model_decode(inter, hidden)
        decoded_one_hot[:, seq_index, :] = decoded_one_hot_line[0]
    
    decoded_one_hot = decoded_one_hot.reshape(n, input_shape1, input_shape2)
    output_mol = []
    _, decoded_mol = decoded_one_hot.max(2)
    for mol in decoded_mol:
        output_mol.append(decoder(hot_to_selfies(mol, encoding_alphabet)))
    
    return output_mol



def attribute_to_dimension(latent_points_combined, input_mol_combined, 
                           output_mol_combined, condition_with, 
                           condition_without, plot_with, plot_without, 
                           description, capital_description, add_description, 
                           test_mol, steps, weight):
    print(latent_points_combined[0])
    shape = latent_points_combined[0].shape
    sum_with_attribute = torch.zeros(shape)
    num_with_attribute = 0
    sum_without_attribute = torch.zeros(shape)
    num_without_attribute = 0
    print('Estimating the attribute vector for '+description+'...') 
    for i in range(len(input_mol_combined)):
        mol = input_mol_combined[i]
        if eval(condition_with):
            num_with_attribute += 1
            sum_with_attribute += latent_points_combined[i]
        elif eval(condition_without):
            num_without_attribute += 1
            sum_without_attribute += latent_points_combined[i]
    attribute_vector1 = sum_with_attribute / num_with_attribute - sum_without_attribute / num_without_attribute
    print(attribute_vector1)
    num_dim = settings['training_VAE']['latent_dimension']
    attribute_vector2, max_index, min_index = remove_trivial_dim(attribute_vector1, num_dim)
    print(attribute_vector2)
    #attribute_vector2 = latent(encoder('CC(O)CC#N'))[1] - latent(encoder('CC(O)CC=O'))[1]
    print('Adding '+add_description+' to the molecule '+ test_mol+':')
    interp = add_attribute(encoder(test_mol), attribute_vector1, steps, weight)
    print(interp)
    print('Plotting the relationship between dimension '+str(max_index)+' and dimension '+str(min_index)+':')
    x=np.array([])
    y=np.array([])
    labels=np.array([])
    for i in range(len(latent_points_combined)):
        x = np.append(x, latent_points_combined[i][max_index])
        y = np.append(y, latent_points_combined[i][min_index])
        mol = output_mol_combined[i]
        if eval(plot_with):
            labels = np.append(labels, 1)
        elif eval(plot_without):
            labels = np.append(labels, 0)
    print(labels[:100])
    plot = plt.scatter(x=x, y=y, c=labels,
                cmap=plt.cm.RdYlGn, marker='.',
                s=10,alpha=0.5, edgecolors='none')
    plt.legend(*plot.legend_elements(num=1),
            loc="upper left", title=capital_description)
    plt.xlabel('Dimension '+str(int(max_index)))
    plt.ylabel('Dimension '+str(int(min_index)))
    plt.savefig('Results/{}_dimensions/VAE_{}-dataset_{}-clusters_{}-properties_{}-epochs_{}-version_{}-dimensions_attribute_analysis_plot_{}_fig'.format(settings['training_VAE']['latent_dimension'], settings['data']['file_description'], settings['kmeans']['num_clusters'], settings['properties']['num_properties'], settings['training_VAE']['num_epochs'], version, settings['training_VAE']['latent_dimension'], description)) 
    plt.show()
    
    label_with = np.array([])
    label_without = np.array([])
    for i in range(len(labels)):
        label = labels[i]
        if label == 1:
            label_with = np.append(label_with, x[i])
        else:
            label_without = np.append(label_without, x[i])
            
    hist_with, bin_edges_with = np.histogram(label_with)
    hist_without, bin_edges_without = np.histogram(label_without)
    
    plt.figure(figsize=[10,8])
    plt.bar(bin_edges_with[:-1], hist_with, width = 0.5, alpha=0.7)
    plt.xlim(min(bin_edges_with), max(bin_edges_with))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Dimension '+str(int(max_index)),fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    title = 'Label 1 Histogram for '+capital_description
    plt.title(title,fontsize=15)
    plt.savefig('Results/{}_dimensions/VAE_{}-dataset_{}-clusters_{}-properties_{}-epochs_{}-version_{}-dimensions_attribute_analysis_histogram_label1_{}_fig'.format(settings['training_VAE']['latent_dimension'], settings['data']['file_description'], settings['kmeans']['num_clusters'], settings['properties']['num_properties'], settings['training_VAE']['num_epochs'], version, settings['training_VAE']['latent_dimension'], description)) 
    plt.show()
    
    plt.figure(figsize=[10,8])
    plt.bar(bin_edges_without[:-1], hist_without, width = 0.5, alpha=0.7)
    plt.xlim(min(bin_edges_without), max(bin_edges_without))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Dimension '+str(int(max_index)),fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    title = 'Label 0 Histogram for '+capital_description
    plt.title(title,fontsize=15)
    plt.savefig('Results/{}_dimensions/VAE_{}-dataset_{}-clusters_{}-properties_{}-epochs_{}-version_{}-dimensions_attribute_analysis_histogram_label0_{}_fig'.format(settings['training_VAE']['latent_dimension'], settings['data']['file_description'], settings['kmeans']['num_clusters'], settings['properties']['num_properties'], settings['training_VAE']['num_epochs'], version, settings['training_VAE']['latent_dimension'], description))     
    plt.show()

def remove_trivial_dim(vec, num_dim, dist=0.2):
    """
    The absolute value of any number stored in the latent point vector that is smaller
    than the maximum absolute value of all numbers minus dist
    """
    latent = torch.zeros([num_dim])
    vector = torch.tensor(vec, dtype=torch.float).to(device)
    for i in range(len(vector)):
        latent[i]=vector[i]
    max_abs_num, max_index = np.absolute(latent).max(0)
    _, min_index = np.absolute(latent).min(0)
    for i in range(len(latent)):
        if float(abs(latent[i])) <= float(max_abs_num) - dist:
            latent[i] = 0
    return latent, max_index, min_index

def add_attribute(x_from, attribute_vector, steps, weight):
    n = steps + 1
    attribute_vector = torch.tensor(attribute_vector, dtype=torch.float).to(device)
    hot = multiple_selfies_to_hot([x_from], largest_molecule_len, encoding_alphabet)
    x_from = torch.tensor(hot, dtype=torch.float).to(device)
    input_shape1 = x_from.shape[1]
    input_shape2 = x_from.shape[2]
    x_from = x_from.reshape(x_from.shape[0], x_from.shape[1] * x_from.shape[2])
    t_from = model_encode(x_from)[0]
    t_from = t_from.reshape(1, 1, t_from.shape[1])
    inter = torch.zeros((1, n, t_from.shape[2]))
    for i in range(n):
        inter[0][i] = t_from[0][0] + i / steps * attribute_vector * weight
        
    hidden = model_decode.init_hidden(batch_size=n)
    
    decoded_one_hot = torch.zeros(n, input_shape1, input_shape2).to(device)
    for seq_index in range(input_shape1):
        decoded_one_hot_line, hidden  = model_decode(inter, hidden)
        decoded_one_hot[:, seq_index, :] = decoded_one_hot_line[0]
    
    decoded_one_hot = decoded_one_hot.reshape(n, input_shape1, input_shape2)
    output_mol = []
    _, decoded_mol = decoded_one_hot.max(2)
    for mol in decoded_mol:
        output_mol.append(decoder(hot_to_selfies(mol, encoding_alphabet)))
    
    return output_mol

def latent(x):
    hot = multiple_selfies_to_hot([x], largest_molecule_len, encoding_alphabet)
    x = torch.tensor(hot, dtype=torch.float).to(device)
    input_shape1 = x.shape[1]
    input_shape2 = x.shape[2]
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    t = model_encode(x)[0]
    latent = t
    t = t.reshape(1, 1, t.shape[1])
        
    hidden = model_decode.init_hidden(batch_size=1)
    
    decoded_one_hot = torch.zeros(1, input_shape1, input_shape2).to(device)
    for seq_index in range(input_shape1):
        decoded_one_hot_line, hidden  = model_decode(t, hidden)
        decoded_one_hot[:, seq_index, :] = decoded_one_hot_line[0]
    
    decoded_one_hot = decoded_one_hot.reshape(1, input_shape1, input_shape2)
    output_mol = []
    _, decoded_mol = decoded_one_hot.max(2)
    for mol in decoded_mol:
        output_mol.append(decoder(hot_to_selfies(mol, encoding_alphabet)))
    
    return output_mol[0], latent

def extract_difference(x_from, x_to, extra, steps):
    """
    Extract the most influential dimension that contributes to the difference
    between x_from and x_to, then sample from x_from along this dimension
    """
    n = steps + 1
    
    hot = multiple_selfies_to_hot([x_from], largest_molecule_len, encoding_alphabet)
    x_from = torch.tensor(hot, dtype=torch.float).to(device)
    input_shape1 = x_from.shape[1]
    input_shape2 = x_from.shape[2]
    x_from = x_from.reshape(x_from.shape[0], x_from.shape[1] * x_from.shape[2])
    
    _, hot = selfies_to_hot(x_to, largest_molecule_len, encoding_alphabet)
    x_to = torch.tensor([hot], dtype=torch.float).to(device)
    x_to = x_to.reshape(x_to.shape[0], x_to.shape[1] * x_to.shape[2])
    
    _, hot = selfies_to_hot(extra, largest_molecule_len, encoding_alphabet)
    extra = torch.tensor([hot], dtype=torch.float).to(device)
    extra = extra.reshape(extra.shape[0], extra.shape[1] * extra.shape[2])
    
    t_from = model_encode(x_from)[0]
    t_from = t_from.reshape(1, 1, t_from.shape[1])
    t_to = model_encode(x_to)[0]
    t_to = t_to.reshape(1, 1, t_to.shape[1])
    extra = model_encode(extra)[0]
    extra = extra.reshape(1, 1, extra.shape[1])
    
    diff = t_to[0][0] - t_from[0][0]
    diff = np.array(diff.detach().numpy())
    max_abs_num = np.absolute(diff).max(0)
    max_num = diff[np.where(np.absolute(diff)==max_abs_num)]
    diff = np.where(np.absolute(diff)==max_abs_num, max_num, 0)
    dimension = np.where(np.absolute(diff)==max_abs_num)
    diff = torch.tensor(diff, dtype=torch.float).to(device)
    inter_from = torch.zeros((1, n, t_to.shape[2]))
    inter_to = torch.zeros((1, n, t_to.shape[2]))
    inter_extra = torch.zeros((1, n, t_to.shape[2]))
    weight = 10
    
    #from t_from:
    for i in range(n):
        inter_from[0][i] = t_from[0][0] + i/steps * diff * weight
        
    hidden = model_decode.init_hidden(batch_size=n)
    
    decoded_one_hot = torch.zeros(n, input_shape1, input_shape2).to(device)
    for seq_index in range(input_shape1):
        decoded_one_hot_line, hidden  = model_decode(inter_from, hidden)
        decoded_one_hot[:, seq_index, :] = decoded_one_hot_line[0]
    
    decoded_one_hot = decoded_one_hot.reshape(n, input_shape1, input_shape2)
    output_mol_from = []
    _, decoded_mol = decoded_one_hot.max(2)
    for mol in decoded_mol:
        output_mol_from.append(decoder(hot_to_selfies(mol, encoding_alphabet)))
    
    #from t_to:
    for i in range(n):
        inter_to[0][i] = t_to[0][0] + i/steps * diff * weight
        
    hidden = model_decode.init_hidden(batch_size=n)
    
    decoded_one_hot = torch.zeros(n, input_shape1, input_shape2).to(device)
    for seq_index in range(input_shape1):
        decoded_one_hot_line, hidden  = model_decode(inter_to, hidden)
        decoded_one_hot[:, seq_index, :] = decoded_one_hot_line[0]
    
    decoded_one_hot = decoded_one_hot.reshape(n, input_shape1, input_shape2)
    output_mol_to = []
    _, decoded_mol = decoded_one_hot.max(2)
    for mol in decoded_mol:
        output_mol_to.append(decoder(hot_to_selfies(mol, encoding_alphabet)))
        
    #from extra:
    for i in range(n):
        inter_extra[0][i] = extra[0][0] + i/steps * diff * weight
        
    hidden = model_decode.init_hidden(batch_size=n)
    
    decoded_one_hot = torch.zeros(n, input_shape1, input_shape2).to(device)
    for seq_index in range(input_shape1):
        decoded_one_hot_line, hidden  = model_decode(inter_extra, hidden)
        decoded_one_hot[:, seq_index, :] = decoded_one_hot_line[0]
    
    decoded_one_hot = decoded_one_hot.reshape(n, input_shape1, input_shape2)
    output_mol_extra = []
    _, decoded_mol = decoded_one_hot.max(2)
    for mol in decoded_mol:
        output_mol_extra.append(decoder(hot_to_selfies(mol, encoding_alphabet)))
    
    return output_mol_from, output_mol_to, output_mol_extra, dimension

def print_vector_analysis(test_mol, attribute_vector, dimension_vector, steps, weight_attribute, weight_dim):
    print('ATTRIBUTE VECTOR ADDED TO '+test_mol+':\n'+ str(add_attribute(encoder(test_mol), attribute_vector, steps, weight_attribute)))
    print('ATTRIBUTE VECTOR SUBTRACTED FROM '+test_mol+':\n'+str(add_attribute(encoder(test_mol), attribute_vector/-1, steps, weight_attribute)))
    print('DIMENSION ADDED TO '+test_mol+':\n'+ str(add_attribute(encoder(test_mol), dimension_vector, steps, weight_dim)))
    print('DIMENSION SUBTRACTED FROM '+test_mol+':\n'+ str(add_attribute(encoder(test_mol), dimension_vector/-1, steps, weight_dim)))
    
    
def write_vector_analysis(f, test_mol, attribute_vector, dimension_vector, steps, weight_attribute, weight_dim):
    f.write('ATTRIBUTE VECTOR ADDED TO '+test_mol+':\n'+ str(add_attribute(encoder(test_mol), attribute_vector, steps, weight_attribute))+'\n')
    f.write('ATTRIBUTE VECTOR SUBTRACTED FROM '+test_mol+':\n'+str(add_attribute(encoder(test_mol), attribute_vector/-1, steps, weight_attribute))+ '\n')
    f.write('DIMENSION ADDED TO '+test_mol+':\n'+ str(add_attribute(encoder(test_mol), dimension_vector, steps, weight_dim))+'\n')
    f.write('DIMENSION SUBTRACTED FROM '+test_mol+':\n'+ str(add_attribute(encoder(test_mol), dimension_vector/-1, steps, weight_dim))+'\n')


def get_selfie_and_smiles_encodings_for_dataset(filename_data_set_file_smiles):
    """
    Returns encoding, alphabet and length of largest molecule in SMILES and SELFIES, given a file containing SMILES molecules.
    input:
        csv file with molecules. Column's name must be 'smiles'.
    output:
        - selfies encoding
        - selfies alphabet
        - longest selfies string
        - smiles encoding (equivalent to file content)
        - smiles alphabet (character based)
        - longest smiles string
    """

    df = pd.read_csv(filename_data_set_file_smiles)
    smiles_list = np.asanyarray(df.smiles)
    smiles_alphabet = list(set(''.join(smiles_list)))
    largest_smiles_len = len(max(smiles_list, key=len))
    print('--> Translating SMILES to SELFIES...')
    selfies_list = list(map(selfies0.encoder, smiles_list))
    largest_selfies_len = max(len_selfie(s) for s in selfies_list)

    all_selfies_chars = split_selfie(''.join(selfies_list))
    all_selfies_chars.append('[epsilon]')
    selfies_alphabet = list(set(all_selfies_chars))
    print('Finished translating SMILES to SELFIES.')
    return(selfies_list, selfies_alphabet, largest_selfies_len, smiles_list, smiles_alphabet, largest_smiles_len)
    
    
if __name__ == '__main__':   
    try:
        version = 6
        content = open('logfile.dat', 'w')
        content.close()
        content = open('results.dat', 'w') 
        content.close()

        if os.path.exists("settings.yml"):        
            user_settings=yaml.safe_load(open("settings.yml","r"))
            settings = user_settings
        else:
            print("Expected a file settings.yml but didn't find it.")
            print()
            exit()
       
        
        print('--> Acquiring data...')        
        type_of_encoding = settings['data']['type_of_encoding']
        file_name_smiles = settings['data']['smiles_file']
        
        selfies_list, selfies_alphabet, largest_selfies_len, smiles_list, smiles_alphabet, largest_smiles_len=get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)
        print('Finished acquiring data.')
        
        print(selfies_alphabet)
        print(selfies_list[0], selfies_list[1])
        if type_of_encoding == 0:
            print('Representation: SMILES')            
            encoding_alphabet=smiles_alphabet
            encoding_alphabet.append(' ') # for padding
            encoding_list=smiles_list
            largest_molecule_len = largest_smiles_len
            print('--> Creating one-hot encoding...')
            data = multiple_smile_to_hot(smiles_list, largest_molecule_len, encoding_alphabet)
            print('Finished creating one-hot encoding.')
        elif type_of_encoding == 1:
            print('Representation: SELFIES')            
            
            encoding_alphabet=selfies_alphabet
            encoding_list=selfies_list
            largest_molecule_len=largest_selfies_len
            
            print('--> Creating one-hot encoding...')
            data = multiple_selfies_to_hot(encoding_list, largest_molecule_len, encoding_alphabet)
            print('Finished creating one-hot encoding.')

        len_max_molec = data.shape[1]
        len_alphabet = data.shape[2]
        len_max_molec1Hot = len_max_molec * len_alphabet
        print(' ')
        print('Alphabet has ', len_alphabet, ' letters, largest molecule is ', len_max_molec, ' letters.')
         
        data_parameters = settings['data']
        batch_size = data_parameters['batch_size']
 
        encoder_parameter = settings['encoder'] 
        decoder_parameter = settings['decoder']
        training_parameters = settings['training_VAE']
  
        model_encode = VAE_encode(**encoder_parameter)
        model_decode = VAE_decode(**decoder_parameter)
       
        model_encode.train()
        model_decode.train()
           
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('*'*15, ': -->', device)        
  
        data = torch.tensor(data, dtype=torch.float).to(device)
  
        train_valid_test_size=[0.5, 0.5, 0.0]    
        x = [i for i in range(len(data))]  # random shuffle input
        shuffle(x)
        data = data[x]
        idx_traintest=int(len(data)*train_valid_test_size[0])
        idx_trainvalid=idx_traintest+int(len(data)*train_valid_test_size[1])    
        data_train=data[0:idx_traintest]
        data_valid=data[idx_traintest:idx_trainvalid]
        data_test=data[idx_trainvalid:]
         
        num_batches_train = int(len(data_train) / batch_size)
        num_batches_valid = int(len(data_valid) / batch_size)
      
        model_encode = VAE_encode(**encoder_parameter).to(device)
        model_decode = VAE_decode(**decoder_parameter).to(device)
        
        if settings['plot']['plot_quality'] and not settings['evaluate']['evaluate_model']:
            recons_quality_valid = []
            recons_quality_train = []
        if settings['plot']['plot_loss'] and not settings['evaluate']['evaluate_model']:
            recons_loss = []
        
        if settings['evaluate']['evaluate_model']:
            if os.path.exists('VAE_dependencies/Saved_models/VAE_{}-dataset_{}-properties_{}-dimensions_encode_epoch_{}'.format(settings['data']['file_description'], settings['properties']['num_properties'], settings['training_VAE']['latent_dimension'], settings['training_VAE']['num_epochs'])):
                model_encode.load_state_dict(torch.load('VAE_dependencies/Saved_models/VAE_{}-dataset_{}-properties_{}-dimensions_encode_epoch_{}'.format(settings['data']['file_description'], settings['properties']['num_properties'], settings['training_VAE']['latent_dimension'], settings['training_VAE']['num_epochs'])))
                model_decode.load_state_dict(torch.load('VAE_dependencies/Saved_models/VAE_{}-dataset_{}-properties_{}-dimensions_decode_epoch_{}'.format(settings['data']['file_description'], settings['properties']['num_properties'], settings['training_VAE']['latent_dimension'], settings['training_VAE']['num_epochs'])))
                model_encode.eval()
                model_decode.eval()
            else: 
                print('No models saved in file with ' + str(settings['training_VAE']['num_epochs']) + ' epochs')
                
        print("start training")
        
        
        train_model(data_train=data_train, data_valid=data_valid, **training_parameters, encoding_alphabet=encoding_alphabet)
        
        if not settings['evaluate']['evaluate_model']:
            torch.save(model_encode.state_dict(), 'VAE_dependencies/Saved_models/VAE_{}-dataset_{}-properties_{}-dimensions_encode_epoch_{}'.format(settings['data']['file_description'], settings['properties']['num_properties'], settings['training_VAE']['latent_dimension'], settings['training_VAE']['num_epochs']))
            torch.save(model_decode.state_dict(), 'VAE_dependencies/Saved_models/VAE_{}-dataset_{}-properties_{}-dimensions_decode_epoch_{}'.format(settings['data']['file_description'], settings['properties']['num_properties'], settings['training_VAE']['latent_dimension'], settings['training_VAE']['num_epochs']))
            #plot epoch vs reconstruction loss / quality
            print(recons_quality_valid, recons_quality_train, recons_loss)
            if settings['plot']['plot_quality']:
                line1, = plt.plot(recons_quality_valid, label='Validation set')
                line2, = plt.plot(recons_quality_train, label='Training set')
                plt.xlabel('Epochs')
                plt.ylabel('Reconstruction Quality (%)')
                plt.legend(handles=[line1, line2])
                plt.show()
            if settings['plot']['plot_loss']:
                plt.plot(recons_loss)
                plt.xlabel('Epochs')
                plt.ylabel('Reconstruction Loss')
                plt.show()
        else:
            test_mol1 = 'C'
            test_mol2 = 'CCCCC'
            test_mol3 = 'N=C1OCC=C1'
            steps = 10
            f = open('Results/{}_dimensions/VAE_{}-dataset_{}-epochs_{}-version_{}-properties_{}-dimensions_{}-step_interpolation_{}_{}_{}_results'.format(settings['training_VAE']['latent_dimension'], settings['data']['file_description'], settings['training_VAE']['num_epochs'], version, settings['properties']['num_properties'], settings['training_VAE']['latent_dimension'], steps, test_mol1, test_mol2, test_mol3), "w+")
            f.write('Linear interpolation of '+str(steps)+' steps between '+test_mol1 +' and '+test_mol2+':')
            print('Linear interpolation of '+str(steps)+' steps between '+test_mol1 +' and '+test_mol2+':\n')
            mol_inter = linear_interpolation(encoder(test_mol1), encoder(test_mol2), steps)
            f.write(str(mol_inter)+'\n')
            print(mol_inter, logp(encoder(test_mol1)))
            f.write('Interpolating '+str(steps)+' steps along the most distinguishing dimension between '+test_mol1 +' and '+test_mol2+' starting from '+test_mol1+':\n')
            print('Interpolating '+str(steps)+' steps along the most distinguishing dimension between '+test_mol1 +' and '+test_mol2+' starting from '+test_mol1+':')
            mol_inter_from, mol_inter_to, mol_inter_extra, dim = extract_difference(encoder(test_mol1), encoder(test_mol2), encoder(test_mol3), steps)
            f.write(str(mol_inter_from)+'\n')
            print(mol_inter_from)
            f.write('Interpolating '+str(steps)+' steps along the most distinguishing dimension between '+test_mol1 +' and '+test_mol2+' starting from '+test_mol2+':\n')
            print('Interpolating '+str(steps)+' steps along the most distinguishing dimension between '+test_mol1 +' and '+test_mol2+' starting from '+test_mol2+':')
            f.write(str(mol_inter_to)+'\n')
            print(mol_inter_to)
            f.write('Interpolating '+str(steps)+' steps along the most distinguishing dimension between '+test_mol1 +' and '+test_mol2+' starting from '+test_mol3+':\n')
            print('Interpolating '+str(steps)+' steps along the most distinguishing dimension between '+test_mol1 +' and '+test_mol2+' starting from '+test_mol3+':')
            f.write(str(mol_inter_extra)+'\n')
            print(mol_inter_extra)
            print('Dimension: '+str(dim))
            
            if settings['properties']['num_properties']>0:
                f.write('\nInterpolating along regularized dimension for logP:\n')
                print('\nInterpolating along regularized dimension for logP:')
                dim_vector = [0.1]
                for i in range(settings['training_VAE']['latent_dimension']-1):
                    dim_vector.append(0)
                dim_vector = np.array(dim_vector)
                interp = add_attribute(encoder('CC#CCCC(C)O'), dim_vector/-1, 50, 5000)
                f.write(str(interp)+'\n')
                print(str(interp))
                f.write('Corresponding logP values for each interpolated molecule along this regularized dimension:\n')
                print('Corresponding logP values for each interpolated molecule along this regularized dimension:')
                logP_lst = []
                for mol in interp:
                    logP_lst.append(logp(encoder(mol)))
                f.write(str(logP_lst)+'\n')
                print(logP_lst)
            
            f.close()
            
        
        with open('COMPLETED', 'w') as content:
            content.write('exit code: 0')


    except AttributeError:
        _, error_message,_ = sys.exc_info()
        print(error_message)