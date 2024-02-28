from torch.utils.data import Dataset
import pickle
import random
import numpy as np
import torch
import argparse

class CipherDataset(Dataset):
    def __init__(self, initial_vocab_path, reviews_path):
        with open(initial_vocab_path, 'rb') as f:
            original_vocab = pickle.load(f)

        index_to_word = {index : word for word, index in original_vocab.items()}
        next_index = len(original_vocab)
        
        self.indexes_to_be_flipped = random.sample(range(2, next_index), next_index//4)
        self.vocab = original_vocab.copy()
        self.index_to_flipped = {i : i for i in range(0, next_index)}
    
        for index in self.indexes_to_be_flipped:
            if (index_to_word[index][::-1] in self.vocab):
                continue
            self.vocab[index_to_word[index][::-1]] = next_index
            self.index_to_flipped[index] = next_index
            next_index += 1
        
        self.index_to_word = {index : word for word, index in self.vocab.items()}
        self.reviews = np.loadtxt(reviews_path)

    def __len__(self):
        # returns the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        y = torch.from_numpy(self.reviews[idx]).to(torch.long)
        x= torch.tensor([self.index_to_flipped[int(i)] for i in y], dtype=torch.long)
        return x, y
    
"""
Code under here is strictly for your debugging purposes; feel free to modify
as desired.
"""
if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('dataset_type', help="Type of dataset to sample from."
            "Options: cipher",
            choices=["cipher"])
    args = argp.parse_args()

    if args.dataset_type == 'cipher':
        ciper_dataset = CipherDataset('./data/initial_vocab.pkl', './data/sentiment_dataset_reviews.txt')
        for _, example in zip(range(4), ciper_dataset):
            x, y = example
            print('x:', ' '.join([ciper_dataset.index_to_word[int(c)] for c in x]))
            print('y:', ' '.join([ciper_dataset.index_to_word[int(c)] for c in y]))
    else:
        raise ValueError("Unknown dataset type in command line args: {}"
                .format(args.dataset_type))

