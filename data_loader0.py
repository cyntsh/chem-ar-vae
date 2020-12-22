#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:47:01 2020

@author: CS
"""

"""
This file is to encode SMILES and SELFIES into one-hot encodings
"""
import re
import numpy as np
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import MolFromSmiles
#from sklearn.preprocessing import MinMaxScaler
import selfies0

def smile_to_hot(smile, largest_smile_len, alphabet):
    """
    Go from a single smile string to a one-hot encoding.
    """
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # integer encode input smile
    for _ in range(largest_smile_len-len(smile)):
        smile+=' '
        
    atoms = []
    for i in range(len(smile)):
        atoms.append(smile[i])        
        
    integer_encoded = [char_to_int[char] for char in atoms]

    # one hot-encode input smile
    onehot_encoded = list()
    for value in integer_encoded:
    	letter = [0 for _ in range(len(alphabet))]
    	letter[value] = 1
    	onehot_encoded.append(letter)
    return integer_encoded, np.array(onehot_encoded)
    

def multiple_smile_to_hot(smiles_list, largest_molecule_len, alphabet):
    """
    Convert a list of smile strings to a one-hot encoding
    
    Returned shape (num_smiles x len_of_largest_smile x len_smile_encoding)
    """
    hot_list = []
    for smile in smiles_list:
        _, onehot_encoded = smile_to_hot(smile, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)
        
def len_selfie(molecule):
    """Returns the length of selfies <molecule>, in other words, the
     number of characters in the sequence."""
    return molecule.count('[') + molecule.count('.')


def split_selfie(molecule):
    """Splits the selfies <molecule> into a list of character strings.
    """
    return re.findall(r'\[.*?\]|\.', molecule)

def selfies_to_hot(molecule, largest_selfie_len, alphabet):
    """
    Go from a single selfies string to a one-hot encoding.
    """
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # pad with [epsilon]
    molecule += '[epsilon]' * (largest_selfie_len - len_selfie(molecule))
    # integer encode
    char_list = split_selfie(molecule)
    integer_encoded = [char_to_int[char] for char in char_list]
    # one hot-encode the integer encoded selfie
    onehot_encoded = list()
    for index in integer_encoded:
        letter = [0] * len(alphabet)
        letter[index] = 1
        onehot_encoded.append(letter)
    return integer_encoded, np.array(onehot_encoded)

def logp(molecule):
    """
    Calculate the logP of the selfies string
    """
    m = MolFromSmiles(selfies0.decoder(molecule))
    return rdMolDescriptors.CalcCrippenDescriptors(m)[0]

def logp_of_smiles(molecule):
    """
    Calculate the logP of the smiles string
    """
    m = MolFromSmiles(molecule)
    return rdMolDescriptors.CalcCrippenDescriptors(m)[0]


def add_logp(selfies_list, num_categories=10): # version 4, 5
    """
    Attach another string represeting ClogP to the end of each selfies molecule
    Since logP is a continuous value, it is first normalized into
    values between 0 and 1 and then split into num_categories categories.
    """
    selfies_lst = selfies_list[:]
    logp_lst = np.array([])
    add_to_alph = np.array([])
    for i in range(num_categories+2):
        add_to_alph = np.append(add_to_alph, '['+str(int(i))+']')
    for sf in selfies_lst:
        logp_lst = np.append(logp_lst, np.array(logp(sf)))
    print('MAX', max(logp_lst)) #delete
    print('MIN', min(logp_lst)) #delete
    logp_lst = logp_lst.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(logp_lst)
    logp_lst = scaler.transform(logp_lst)
    for s in range(len(selfies_lst)):
        num = 0
        for i in range(num_categories):
            if logp_lst[s] >= i * 1/num_categories:
                num+=1
        selfies_lst[s] += '['+str(int(num-1))+']['+str(int(num))+']['+str(int(num+1))+']'
        if num > 20:
            print(selfies_lst[s])
    return selfies_lst, add_to_alph, scaler


def hot_to_selfies(molecule, alphabet):
    """
    Go from the one-hot encoding to the corresponding selfies string
    """
    one_hot = ''
    for _ in molecule:
        'pass'
        #print(_)
        one_hot += alphabet[_]
    return one_hot
    

def multiple_selfies_to_hot(selfies_list, largest_molecule_len, alphabet):
    """
    Convert a list of selfies strings to a one-hot encoding
    """
    hot_list = []
    for selfiesI in selfies_list:
        _, onehot_encoded = selfies_to_hot(selfiesI, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)

