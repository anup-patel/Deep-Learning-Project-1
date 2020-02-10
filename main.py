#!/usr/bin/env python
# coding: utf-8

# In[1]:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import pandas as pd
import sys
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[2]:


# from google.colab import drive
# drive.mount('/content/gdrive')

# In[3]:
if __name__ == "__main__":
	#num=int(input("Enter the Number"))
	##### PART 1
	a = str(sys.argv[1])
	data = str(sys.argv[2])
	data=pd.read_csv(data,header=None)


	# In[4]:


	data=np.array(data.values)
	print("Dept:: CSA")
	print("Person Name:: Anup Patel")


	# In[5]:


	#data[0]


	# In[6]:


	f=open("Software1.txt","w+")
	output=[]
	for val in data:
	    #Divide by 3
	    if (val%3==0 and val%5==0):
	        #print("FizzBuzz")
	        output.append("Fizzbuzz")
	        #f.write("FizzBuzz\n")
	    #Divide by 5
	    elif val%3==0:
	        #print("Fizz")
	        output.append("Fizz")
	        #f.write("Fizz\n")
	    #Divide by both 3 and 5
	    elif val%5==0 :
	        #print("Buzz")
	        output.append("Buzz")
	        #f.write("Buzz\n")
	    else:
	        #print(val)
	        output.append(str(val[0]))
	        #f.write(str(val[0])+"\n")
	for item in output:
	    f.write("%s\n" % item)
	f.close()
	print("Software1 File Generated")
	#output=pd.read_csv("Software1.txt",header=None).values
	# actual=pd.read_csv("test_output.txt",header=None).values
	# mismatch_count=0
	# for i in range(len(output)):
	#     if(output[i]!=actual[i][0]):
	#         mismatch_count+=1
	# #print(mismatch_count)

	# #Accuracy
	# accuracy=(1-mismatch_count/len(output))*100
	# print("Accuracy (P1) :: ",accuracy,"%")


	#### PART 2

	### Training Part - Uncomment to Train Model (I am importing saved model)
	# train_x=pd.read_csv("training_data_x_part_2.txt",header=None).values
	# train_y=pd.read_csv("training_data_y_part_2.txt",header=None).values
	# ### input encoding to 16 bit
	# encoded_x=[]
	# input_features=[]
	# for i in range(len(train_x)):
	#     bit_array=[]
	#     tmp=int(train_x[i])
	#     tmp_bin=format(tmp, '016b')
	#     encoded_x.append(tmp_bin)
	#     for bit in tmp_bin:
	#         bit_array.append(int(bit))
	#     input_features.append(bit_array)

	# ### output encoding to 4 bit
	# y=[]
	# for val in train_y:
	#     if val=="Fizz":
	#         y.append(np.array([1,0,0,0]))
	#     elif val=="Buzz":
	#         y.append(np.array([0,1,0,0]))
	#     elif val=="FizzBuzz":
	#         y.append(np.array([0,0,1,0]))
	#     else:
	#         y.append(np.array([0,0,0,1]))

	# input_features=np.array(input_features)
	# y=np.array(y)

	## Model (Pytorch - 1.3)


	#### Model Architecture
	input_size = 16
	hidden_sizes = [150,150,150]
	output_size = 4
	# Build a feed-forward network
	model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
	                      nn.ReLU(),
	                      nn.BatchNorm1d(hidden_sizes[0]),
	                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
	                      nn.ReLU(),
	                      nn.BatchNorm1d(hidden_sizes[1]),
	                      nn.Linear(hidden_sizes[1], hidden_sizes[2]),
	                      nn.ReLU(),
	                      nn.BatchNorm1d(hidden_sizes[2]),
	                      nn.Linear(hidden_sizes[2], output_size)).to(device)

	#### Data Preprocessing
	# train_x=pd.read_csv("training_data_x_part_2.txt",header=None).values
	# train_y=pd.read_csv("training_data_y_part_2.txt",header=None).values
	# ### input encoding to 16 bit
	# encoded_x=[]
	# input_features=[]
	# for i in range(len(train_x)):
	#     bit_array=[]
	#     tmp=int(train_x[i])
	#     tmp_bin=format(tmp, '016b')
	#     encoded_x.append(tmp_bin)
	#     for bit in tmp_bin:
	#         bit_array.append(int(bit))
	#     input_features.append(bit_array)

	# ### output encoding to 4 bit
	# y=[]
	# for val in train_y:
	#     if val=="Fizz":
	#         y.append(0)
	#     elif val=="Buzz":
	#         y.append(1)
	#     elif val=="FizzBuzz":
	#         y.append(2)
	#     else:
	#         y.append(3)


	# input_features=np.array(input_features)
	# input_features=torch.from_numpy(input_features).float()
	# y=np.array(y)
	# y=torch.from_numpy(y).long()

	# #### Hyper-parameter
	# learning_rate = 0.05
	# criterion = nn.CrossEntropyLoss()
	# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	# def accuracy(y_hat, y):
	# 	pred = torch.argmax(y_hat, dim=1)
	# 	return (pred == y).float().mean()

	# ### Dataloader
	# train_dataset = Data.TensorDataset(input_features.to(device),
 #                                    y.to(device))
	# loader = Data.DataLoader(dataset=train_dataset,
	#                         batch_size=128*5,
	#                         shuffle=True)

	# #### Training Model
	# model.train()
	# epochs = 1000
	# for epoch in range(1,epochs):
	#     correct = 0
	#     total = 0
	#     for step,(batch_x, batch_y) in enumerate(loader):
	#         out = model(batch_x).to(device)
	#         #print(out)
	#         loss = criterion(out, batch_y) 
	#         optimizer.zero_grad() 
	#         loss.backward() 
	#         optimizer.step() 
	#     _, predicted = torch.max(out.data, 1)
	#     total = total + batch_y.size(0)
	#     correct = correct + (predicted == batch_y).sum().item()
	#     acc = 100*correct/total
	#     model.train()
	#     print('Epoch : {:0>4d} | Loss : {:<6.4f} | Train Accuracy : {:<6.2f}%'.format(epoch,loss,acc))

	# #### Save Model
	# torch.save(model, 'torch_model.pth')

	#### Load Saved Model
	#model=torch.load('model/torch_model.pth')
	model=torch.load('model/torch_model.pth',map_location=torch.device('cpu')) # To run model on CPU Only system
	

	#### Testing Data
	test_data=pd.read_csv(str(sys.argv[2]),header=None).values
	model.eval()

	### Encoding test data (input Features)
	encoded_x=[]
	test_features=[]
	for i in range(len(test_data)):
	    bit_array=[]
	    tmp=int(test_data[i])
	    tmp_bin=format(tmp, '016b')
	    encoded_x.append(tmp_bin)
	    for bit in tmp_bin:
	        bit_array.append(int(bit))
	    test_features.append(bit_array)

	#tmp = model.predict_classes(test_data)
	test_features=np.array(test_features)
	test_features=torch.from_numpy(test_features).float().to(device)
	test_features=test_features.to(device)

	#Prediction Step
	pred = model(test_features).to(device)

	#print(pred)
	_, predicted = torch.max(pred.data, 1)

	#Output Decoding
	#Output Decoding
	### output encoding to 4 bit
	out=[]
	for i in range(len(predicted)):
	    if predicted[i]==0:
	        out.append("Fizz")
	    elif predicted[i]==1:
	        out.append("Buzz")
	    elif predicted[i]==2:
	        out.append("FizzBuzz")
	    else:
	        tmp=''.join(str(i) for i in test_data[i])
	        #tmp=int(tmp, 2)
	        out.append(str(tmp))

	# actual=pd.read_csv("test_output.txt",header=None).values
	# mismatch_count=0
	# #print(out)
	# for i in range(len(out)):
	#     if(out[i].lower()!=actual[i][0].lower()):
	#         mismatch_count+=1
	# #print(mismatch_count)

	# #Accuracy
	# accuracy=(1-mismatch_count/len(test_data))*100
	# print("Accuracy (P2) :: ",accuracy,"%")

	### Storing Test Output file
	file=open("Software2.txt","w+")
	for item in out:
	    file.write("%s\n" % item)
	file.close()
	print("Software2 File Generated")
