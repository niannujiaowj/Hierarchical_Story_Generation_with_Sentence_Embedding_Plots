from Scripts.util import count_line
import numpy as np
import h5py
import gc
import torch
from tqdm import tqdm
#
# train 272600, valid 15620, test 15138

'''load = np.load("Dataset/SenEmbedding/train.npz", allow_pickle=True)
print(load.files)
n=0
for file in load.files:
   n+=len(load[file])
print(n)'''


'''data = torch.load("Data/WritingPrompts2SenEmbeddings/valid_SenEmbedding_dict.pt")
print("data loaded")


with h5py.File("Data/WritingPrompts2SenEmbeddings/valid_SenEmbedding_dict.hdf5", "w") as f:
    for k,v in tqdm(data.items()):
        dset = f.create_dataset(str(k), data = v)
        #print("{} out of {} is saved".format(n, enumerate(data.items())))

f = h5py.File("Data/WritingPrompts2SenEmbeddings/valid_SenEmbedding_dict.hdf5", 'r')["10000000"]
#data = f["10000000"]
print(f[...])
#print(data[...])'''
