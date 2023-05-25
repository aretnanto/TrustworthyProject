import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd
import numpy as np
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
import re
from torch.utils.data import Dataset, DataLoader

def tokenize(text, max_length):
    tokenizer = get_tokenizer('basic_english')
    text = text.lower()
    text = re.sub(r"([.!?,'*])", r"", text)
    text = re.sub(r"([-])", r" ", text)
    tokens = tokenizer(text)
    if len(tokens) < max_length:
      tokens.extend(['<PAD>']*(max_length - len(tokens)))
    tokens = tokens[:max_length]
    tokens = [glove.stoi.get(token, len(glove.stoi) - 1) for token in tokens]
    return tokens

class CustomDataset(Dataset):
    def __init__(self, tensor1, tensor2):
        self.tensor1 = tensor1
        self.tensor2 = tensor2

    def __len__(self):
        return len(self.tensor1)

    def __getitem__(self, index):
        return self.tensor1[index], self.tensor2[index]

class ToxicClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embeddings_vectors, hidden_dim = 32, output_dim = 1):
        super(ToxicClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings_vectors, freeze=True)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        _, embedded = self.rnn(embedded)
        return self.output(embedded[-1])
        
#init variables
batch_size = 32
token_dim = 50
length = 100


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Handle Glove
glove = GloVe(name='6B', dim=token_dim)
padding_vector = torch.zeros(token_dim)
padding_token = '<PAD>'
glove.itos.append(padding_token)  
glove.stoi[padding_token] = len(glove.itos) - 1 
glove.vectors = torch.cat((glove.vectors, padding_vector.unsqueeze(0)), dim=0) 

##Load Data
path = os.path.join('data')
train = pd.read_csv(os.path.join(path, 'train.csv'))
train = train[train['identity_annotator_count'] == 0] #fairness through unawareness
label = train['target'].apply(lambda x: 0 if x <= 0.5 else 1)
data = train['comment_text'].apply(lambda x: tokenize(x, length))
data = data.values
data = np.array(list(data), dtype=np.int64)
data = torch.Tensor(data).long()
data = data.to(device)
label = label.values
label = label.reshape(-1,1)
label = torch.Tensor(label)
label = label.to(device)
train_data = CustomDataset(data, label)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

#Load Test Data
test = pd.read_csv(os.path.join(path, 'test_public_expanded.csv'))
test2 = pd.read_csv(os.path.join(path, 'test_private_expanded.csv'))
all_test = pd.concat([test, test2], axis = 0)
test_label = all_test['toxicity'].apply(lambda x: 0 if x <= 0.5 else 1)
test_data = all_test['comment_text'].apply(lambda x: tokenize(x, length))
test_data = test_data.values
test_data = np.array(list(test_data), dtype=np.int64)
test_data = torch.Tensor(test_data).long()
test_data = test_data.to(device)
test_label = test_label.values
test_label = test_label.reshape(-1,1)
test_label = torch.Tensor(test_label)
test_label = test_label.to(device)
test_data = CustomDataset(test_data, test_label)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


#Train Model 
model = ToxicClassifier(len(glove), token_dim, glove.vectors.to(device))
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
num_epochs = 100
for epoch in range(0, num_epochs):
  train_loss = 0.0
  train_correct = 0
  train_total = 0
  model.train()
  pbar = tqdm(train_loader)
  for _, batch in enumerate(pbar, 0):
      input = batch[0]
      label = batch[1]
      output = model(input)
      loss = criterion(output, label)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      predicted_labels = (torch.sigmoid(output) >= 0.5).float()
      train_correct += (predicted_labels == label).sum().item()
      train_total += len(label)
      pbar.set_postfix(MSE=loss.item())
      train_loss += loss
  train_loss /= len(train_loader.dataset)
  train_acc = train_correct / train_total
  print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")  
  
  test_loss = 0.0
  test_correct = 0
  test_total = 0
  pbar = tqdm(test_loader)
  with torch.no_grad():
      for i, data in enumerate(pbar, 0):
          input = batch[0]
          label = batch[1]
          loss = criterion(output, label)
          predicted_labels = (torch.sigmoid(output) >= 0.5).float()
          test_total += len(label)
          test_correct += (predicted_labels == label).sum().item()
          test_loss += loss
          pbar.set_postfix(MSE=loss.item())
      
  test_loss /= len(test_loader.dataset)
  test_acc = test_correct / test_total
  print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")