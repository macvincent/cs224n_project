from torch.utils.data import Dataset
import pickle
import random
import numpy as np
import torch
import argparse
import json
from tqdm import tqdm

class CipherDataset(Dataset):
    def __init__(self, initial_vocab_path, reviews_path):
        with open(initial_vocab_path, 'rb') as f:
            original_vocab = pickle.load(f)

        index_to_word = {index : word for word, index in original_vocab.items()}
        next_index = len(original_vocab)
        
        self.indexes_to_be_flipped = random.sample(range(2, next_index), next_index//9)
        print(f"We are flipping: {len(self.indexes_to_be_flipped)} words")
        self.vocab = original_vocab.copy()
        self.index_to_flipped = {i : i for i in range(0, next_index)}
    
        for index in self.indexes_to_be_flipped:
            if (index_to_word[index][::-1] in self.vocab):
                continue
            self.vocab[index_to_word[index][::-1]] = next_index
            self.index_to_flipped[index] = next_index
            next_index += 1
        
        self.index_to_word = {index : word for word, index in self.vocab.items()}
        print(f"We have a vocab size of: {len(self.index_to_word)}")
        self.reviews = np.loadtxt(reviews_path)
        print(f"We have a dataset size of: {len(self.reviews)}")

    def __len__(self):
        # returns the length of the dataset
        return len(self.reviews)

    def __getitem__(self, idx):
        y = torch.from_numpy(self.reviews[idx]).to(torch.long)
        x = torch.tensor([self.index_to_flipped[int(i)] for i in y], dtype=torch.long)
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
        output_file_path = "cipher_data.jsonl"
        num_dataset = 200000
        train_num = 0.8 * num_dataset
        prompt = "decipher this sentence into a readable version"
        test_input = []
        test_output = []
        print("start conversion...")

        with open(output_file_path, 'w') as output_file:
            for i, example in tqdm(zip(range(num_dataset), ciper_dataset)):
                x, y = example
                input = ' '.join([ciper_dataset.index_to_word[int(c)] for c in x]).strip() 
                output = ' '.join([ciper_dataset.index_to_word[int(c)] for c in y]).strip()
                if i < train_num:
                    data = {"text":f'''[INST] {input} [/INST] {output} </s>'''}
                    output_file.write(json.dumps(data) + '\n')
                else:
                    test_input.append(f'''[INST] {input} [/INST] ''')
                    test_output.append(f'''{output} </s>''')

        test_dataset = [test_input, test_output]
        with open('test_dataset.pickle', 'wb') as handle:
            pickle.dump(test_dataset, handle)

    else:
        raise ValueError("Unknown dataset type in command line args: {}"
                .format(args.dataset_type))

