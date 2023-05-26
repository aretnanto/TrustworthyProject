from torchtext.vocab import GloVe
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms

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

def tokenize(text, max_length = 100):
    tokenizer = get_tokenizer('basic_english')
    text = text.lower()
    text = re.sub(r"([.!?,'*])", r"", text)
    text = re.sub(r"([-])", r" ", text)
    tokens = tokenizer(text)
    if len(tokens) < max_length:
      tokens.extend(['<PAD>']*(max_length - len(tokens)))
    tokens = tokens[:max_length]
    tokens = [glove.stoi.get(token, len(glove.stoi) - 1) for token in tokens]
    tokens = np.array(tokens, dtype=np.int64)
    return tokens

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
token_dim = 50
length = 100

dataset = get_dataset(dataset="civilcomments", download=True)
train_data = dataset.get_subset(
    "train")
train_loader = get_train_loader("standard", train_data, batch_size=32)
test_data = dataset.get_subset(
    "val")
test_loader = get_train_loader("standard", train_data, batch_size=32)

from wilds.common.grouper import CombinatorialGrouper
unawareness = CombinatorialGrouper(dataset, ['identity_any'])

train_loader = get_train_loader(
    "group", train_data, grouper=unawareness, n_groups_per_batch=1, batch_size=32
)

model = ToxicClassifier(len(glove), token_dim, glove.vectors.to(device))
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-1)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience = 10)
num_epochs = 100
for epoch in range(0, num_epochs):
  train_loss = 0.0
  train_correct = 0
  train_total = 0
  model.train()
  pbar = tqdm(train_loader)
  for _ , cur_train in enumerate(pbar):
      input, label, metadata = cur_train
      if metadata[:,8].sum() == 0:
        input = tuple(map(tokenize, input))
        input = torch.Tensor(input).long().to(device)
        label = label.to(device)
        output = model(input).reshape(-1)
        loss = criterion(output, label.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predicted_labels = (torch.sigmoid(output) >= 0.5).float()
        train_correct += (predicted_labels == label).sum().item()
        train_total += len(label)
        pbar.set_postfix(MSE=loss.item())
        train_loss += loss
  train_loss /= len(train_loader.dataset)
  scheduler.step(train_loss)
  train_acc = train_correct / train_total
  print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")  

  test_loss = 0.0
  test_correct = 0
  test_total = 0
  pbar = tqdm(test_loader)
  model.eval()
  with torch.no_grad():
      for i, data in enumerate(pbar, 0):
          input, label, groupings = cur_train
          input = tuple(map(tokenize, input))
          input = torch.Tensor(input).long().to(device)
          label = label.to(device)
          output = model(input).reshape(-1)
          loss = criterion(output, label.float())
          predicted_labels = (torch.sigmoid(output) >= 0.5).float()
          test_total += len(label)
          test_correct += (predicted_labels == label).sum().item()
          test_loss += loss
          pbar.set_postfix(MSE=loss.item())
      
  test_loss /= len(test_loader.dataset)
  test_acc = test_correct / test_total
  print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")