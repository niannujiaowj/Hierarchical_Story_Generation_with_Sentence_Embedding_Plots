from Scripts.util import count_line
import numpy as np
import h5py
import gc
import torch
from tqdm import tqdm
#
# Story No.: train 272600, valid 15620, test 15138
# Sentence No.: train 12563178, valid 718950, test 700158


# train.npz
#
#        0:20000       917383 dic3
#        20000:40000   921843 dic1
#        40000:60000   924270 dic4
#        60000:80000   920013 dic5
#        80000:100000  917629  dic2

#1 arr_1 100000:120000    914933 dic6
#2 arr_2 120000:150000    1391276 dic7

#3 arr_3 150000:170000    928822 dic8
#4 arr_4 170000:200000    1388162 dic9
#5 arr_5 200000:220000    914188 dic10
#6 arr_6 220000:250000    1380127 dic11
#7 arr_7 250000:272600    1044532 dic12


import glob
import re
from Scripts.util import count_line


'''class process:
    def __init__(self,datasetname):
        self.datasetname = datasetname

    def data_check(self):
        num_list = []
        for file in glob.glob("Dataset/Plots/{}*".format(self.datasetname)):
            begin = re.findall(r".*/{}_(.*)_".format(self.datasetname),file)
            num_list.append(int(begin[0]))
            num_list.sort()
        #print(num_list)
        #print(len(num_list))

        self.file_list = []
        for i in num_list:
            for file in glob.glob("Dataset/Plots/{}*".format(self.datasetname)):
                begin = re.findall(r".*/{}_(.*)_".format(self.datasetname), file)
                if int(begin[0]) == i:
                    self.file_list.append(file)

        #print(self.file_list)

        # missing file list
        missing_file_list = []
        for n in range(len(self.file_list)-1):
            try:
                end_of_previous_file = int(re.findall(r".*/{}.*_(.*)".format(self.datasetname),self.file_list[n])[0])
            except:
                end_of_previous_file = "final"
            begin_of_next_file = int(re.findall(r".*/{}_(.*)_".format(self.datasetname),self.file_list[n+1])[0])


            if end_of_previous_file != begin_of_next_file:
                print(n)
                print(self.file_list[n])
                print(self.file_list[n+1])
                print(end_of_previous_file,begin_of_next_file)
                missing_file_list.append(end_of_previous_file)
        print("missing_file_list", missing_file_list)



        n = 0
        for file in self.file_list:
            begin = re.findall(r".*/{}_(.*)_".format(self.datasetname),file)
            end = re.findall(r".*/{}.*_(.*)".format(self.datasetname),file)
            line = count_line(file)
            try:
                if int(end[0]) - int(begin[0]) == line:
                    print(file + " is checked. Number of lines is {}".format(line))
                    n += line
                else:
                    print("!!!Caution!!! of {}".format(file))
            except:
                print(file + " needs check!!! Number of lines is {}".format(line))
                n += line

        print("Total number is {}".format(n))

        return self.file_list

    def data_merge(self):
        with open("Dataset/Plots/{}".format(self.datasetname),"a") as f_in:
            for file in self.file_list:
                with open(file,"r") as f_out:
                    data = f_out.read()
                    f_in.write(data)


        print("final number of lines is {}".format(count_line("Dataset/Plots/{}".format(self.datasetname))))


a = process("train")
a.data_check()
a.data_merge()'''


# Story No.: train 272600, valid 15620, test 15138

with open("Dataset/Plots/train","a") as f:
    for n in tqdm(range(0,272600)):
        with open("Dataset/WritingPrompts/train.wp_source","r") as f1:
            for line in f1.readlines()[n:n+1]:
                data1 = line.strip("\n")
        with open("Dataset/Plots/train_plots","r") as f2:
            for line in f2.readlines()[n:n+1]:
                data2 = line.strip("\n")
        #data = data1 + " " +data2
        f.write(data1 + " " +data2 + "\n") 

print(count_line("Dataset/Plots/train"))
