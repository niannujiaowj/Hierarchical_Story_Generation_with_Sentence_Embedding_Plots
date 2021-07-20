from Scripts.util import count_line
import numpy as np
import h5py

#
"""# train 272600, valid 15620, test 15138
#train = np.load("Dataset/SenEmbedding/train.npy", allow_pickle=True)
#train1 = np.load("Dataset/SenEmbedding/train_c_1_100000.npy", allow_pickle=True) # (100000, *, 768)
#train2 = np.load("Dataset/SenEmbedding/train_c_100000_final.npy", allow_pickle=True)  # (172600, *, 768)
valid = np.load("Dataset/SenEmbedding/valid.npy", allow_pickle=True)
#test = np.load("Dataset/SenEmbedding/test.npy", allow_pickle=True)

#print(train.shape,valid.shape,test.shape)
#print(valid.shape, test.shape)
#print(train1.shape, train2.shape)


f = h5py.File('Dataset/Senembedding/valid.h5', 'a')

f.create_dataset("data",data=valid)
f.close()

'''for i in range(10):

    # Data to be appended
    new_data = np.ones(shape=(100,64,64)) * i
    new_label = np.ones(shape=(100,1)) * (i+1)

    if i == 0:
        # Create the dataset at first
        f.create_dataset('data', data=new_data, compression="gzip", chunks=True, maxshape=(None,64,64))
        f.create_dataset('label', data=new_label, compression="gzip", chunks=True, maxshape=(None,1))
    else:
        # Append new data to it
        f['data'].resize((f['data'].shape[0] + new_data.shape[0]), axis=0)
        f['data'][-new_data.shape[0]:] = new_data

        f['label'].resize((f['label'].shape[0] + new_label.shape[0]), axis=0)
        f['label'][-new_label.shape[0]:] = new_label

    print("I am on iteration {} and 'data' chunk has shape:{}".format(i,f['data'].shape))

f.close()'''"""

'''train1 = np.load("Dataset/SenEmbedding/train_100000_150000.npy", mmap_mode=None, allow_pickle=True)
#train2 = np.load("Dataset/SenEmbedding/train_c_100000_final.npy", mmap_mode=None, allow_pickle=True)

print(train1.shape)'''

'''import json

with open("Dataset/SenEmbedding/train_100000_150000.json") as f:
    data = json.load(f)

np.save("Dataset/SenEmbedding/train_100000_150000.npy",data)'''

final = np.load("Dataset/SenEmbedding/train_100000_150000.npy",allow_pickle=True)
print("final",final)

for i in final:
    print(i)