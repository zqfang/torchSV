
import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, accuracy_score

from dataset import DelDataset
from model import DELCNN

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_class = 2
input_size = 256
batch_size = 32
num_workers = 8
learning_rate = 0.001
num_epochs = 1000

test_csv = ""
test_image_dir = ""
MODEL_PATH = "checkpoints/model.epoch.1000.pth"


test_data = DelDataset(test_csv, test_image_dir)
print("Prepare DataLoader ")
test_loader =  DataLoader(test_data, batch_size=batch_size, num_workers= num_workers)

print("Build Model")
# model = LogisticRegression(input_size)
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


# load the model checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
print('Previously trained model weights state_dict loaded...', file=sys.stderr)
# load trained optimizer state_dict
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print('Previously trained optimizer state_dict loaded...', file=sys.stderr)
# load the criterion
criterion = checkpoint['loss']
print('Trained model loss function loaded...', file=sys.stderr)


## Evaluation
model.eval()
with torch.no_grad():
    test_loss = 0
    auc = 0
    acc = 0
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
        y_pred = F.softmax(outputs, dim=-1).detach().cpu().numpy()
        y_pred_prob.append(y_pred)
        targets = targets.detach().cpu().numpy()
        y_test.append(targets)


y_test = np.concatenate(y_test)
y_pred_prob = np.concatenate(y_pred_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
precision, recall, pr_threshold = precision_recall_curve(y_test, y_pred, pos_label=1)

fig, ax = plt.subplots(1,2, figsize=(8,4))
ax[0].plot(fpr, tpr, color='darkorange',
           lw=2, label="MLP AUC = {:.2f}".format(auc))
ax[0].legend()
ax[0].plot([0,1],[0,1], linestyle='--', color='navy', lw=2,)
ax[0].set_title(f"ROC")
ax[0].set_xlim([0.0, 1.0])
ax[0].set_ylim([0.0, 1.05])
ax[0].set_xlabel('False Positive Rate')
ax[0].set_ylabel('True Positive Rate')
## precision recall
ax[1].step(recall, precision, color='g', alpha=0.2, where='post')
ax[1].fill_between(recall, precision, alpha=0.2, color='g', step='post')
ax[1].set_xlabel('Recall')
ax[1].set_ylabel('Precision')
ax[1].set_ylim([0.0, 1.0])
ax[1].set_xlim([0.0, 1.0])
ax[1].set_title('Precision-Recall curve')
plt.tight_layout()
fig.savefig("evaluation.pdf")