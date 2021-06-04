

import os, sys, glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, accuracy_score

from dataset import DelDataset, TrainValTestSplit
from model import DELCNN

num_class = 2
input_size = 256
batch_size = 32
num_workers = 8
learning_rate = 0.001
num_epochs = 1000
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

T## run rainValTestSplit(...) first
train_csv = ""
test_csv = ""
train_image_dir = ""
test_image_dir = ""



train_data = DelDataset(train_csv, train_image_dir)
test_data = DelDataset(test_csv, test_image_dir)
print("Prepare DataLoader ")
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers= num_workers)#sampler=train_sampler, num_workers=1 )# sampler=SubsetRandomSampler() )
test_loader =  DataLoader(test_data, batch_size=batch_size, num_workers= num_workers)

print("Build Model")
model = DELCNN(input_size, num_class = num_class)
model.to(device)

# weight = [class_sample_count[0] / class_sample_count[1]]
# weight = [(len(y) - sum(y))/sum(y)]
# criterion = nn.BCEWithLogitsLoss(pos_weight= torch.FloatTensor(weight).to(device))
# optimizer = torch.optim.SGD(model.parameters(), 
#                             lr=learning_rate, 
#                             momentum=0.9, weight_decay=0.0005) 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, )
# let learning_rate decrease by 50% at 500, 1000 and 2000-th epoch
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [500, 1000, 2000], gamma=0.5)


print("Start training")
# Training the Model
os.makedirs("checkpoints", exist_ok=True)
last_loss = 1000
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    #running_loss2 = 0.0
    for i, embeds  in enumerate(train_loader):
        inputs, targets = embeds['image'], embeds['label']
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # target size must be the same as ouput size
        loss = criterion(outputs, targets) # 
        loss.backward()
        optimizer.step()
        scheduler.step()
        # print statistics
        running_loss += loss.item()
    if (epoch+1) % 10 == 0:     
        print('epoch [%d] loss: %.7f' % (epoch + 1, running_loss /10))
        PATH = f'checkpoints/model.epoch.{epoch+1}.pth'
        if running_loss < last_loss:
            last_loss = min(last_loss, running_loss)
            torch.save({'model_state_dict': model.state_dict(),         
                        'optimizer_state_dict': optimizer.state_dict()}, PATH)


        model.eval() # fixed dropout, batchnorm layer
        with torch.no_grad():
            test_loss = 0
            y_test = []
            y_pred_prob = []
            for embeds in test_loader:
                inputs, targets = embeds['image'], embeds['label']
                inputs = inputs.reshape((-1, input_size)).to(device)
                targets = targets.view(-1).to(device)
                #targets = targets.view(-1).type(torch.int)
                outputs = model(inputs)
                test_loss += criterion(outputs, targets) # 
                # y_pred = F.sigmoid(outputs).detach().cpu().numpy()
                y_pred = F.softmax(outputs, dim=-1).argmax().detach().cpu().numpy()
                y_pred_prob.append(y_pred)
                targets = targets.detach().cpu().numpy()
                y_test.append(targets)

            y_test = np.concatenate(y_test)
            y_pred_prob = np.concatenate(y_pred_prob)
            average_precision = average_precision_score(y_test, y_pred_prob)
            auc = roc_auc_score(y_test, y_pred_prob)
            acc = accuracy_score(y_test, y_pred_prob > 0.5)
            print(f"test_loss: {test_loss}, Accuary: {acc}, Average precision: {average_precision}, AUC: {auc}")