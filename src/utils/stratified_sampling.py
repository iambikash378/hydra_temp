from torch.utils.data import Dataset, Subset
from typing import List
from sklearn.model_selection import train_test_split
import numpy as np

def data_split(dataset: Dataset, split_ratio: List):
    labels = [ dataset[i][1] for i in range(len(dataset))]
    labels = np.array(labels)
    indices = np.arange(len(labels))

    train_idx, val_or_test_idx, train_y, val_or_test_y = train_test_split(
        indices, labels, test_size = split_ratio[1], stratify= labels, random_state= 42)
    
    train_dataset = Subset(dataset, train_idx)
    val_or_test_dataset = Subset(dataset, val_or_test_idx)

    return train_dataset, val_or_test_dataset
    


    

    
    
    






