import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


##read dataset
res = pd.read_csv('supervised_dataset.txt', sep="|")
valset = pd.read_csv('cadec_dataset.txt', sep="|")

#initialize embedding model
embedmodel = SentenceTransformer('all-MiniLM-L6-v2')

#compute embeddings for the result set
phrases = res['phrases'].tolist()
concepts = res['concepts'].tolist()
phrases_embed = embedmodel.encode(phrases, convert_to_tensor=True)
concepts_embed = embedmodel.encode(concepts, convert_to_tensor=True)

#compute embeddings for the test set
phrases_test = valset['phrases'].tolist()
concepts_test = valset['concepts'].tolist()
phrases_test_embed = embedmodel.encode(phrases_test, convert_to_tensor=True)
concepts_test_embed = embedmodel.encode(concepts_test, convert_to_tensor=True)


#Define Feedforward model
class FeedForward(nn.Module):
    
    def __init__(self, embedding_dim=384):
        super().__init__()
        
        self.fc1 = nn.Linear(embedding_dim,576)
        self.fc2 = nn.Linear(576,embedding_dim)
    
    def forward(self, phrases_embeddings):
        res_embeddings = self.fc2(torch.tanh(self.fc1(x)))
        return res_embeddings

model = FeedForward()
print(model)

#Define loss and optimizer
criterion =  nn.CosineEmbeddingLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

#define Train Loader
train_dataset = TensorDataset(phrases_embed, concepts_embed)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#define Test Loader
test_dataset = TensorDataset(phrases_test_embed, concepts_test_embed)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

#Evaluation function on cadec
def evaluate(model, loader, concepts_corpus):
    model.eval()
    acc_result = []

    for x, y in loader:
        y_hat = model(x)

        #use 1-Nearest Neighbor to classify to a concept
        out = []
        for entry in util.semantic_search(y_hat, concepts_corpus, top_k=1):
          n = concepts_corpus[entry[0]['corpus_id']]
          out.append(n)

        out = torch.stack(out)
        
        #compare predicted result with truth
        acc_result.append(torch.equal(out,y))

    train_acc = np.sum(acc_result)/len(acc_result)
    print(f"acc: {train_acc:.3f}")
    return train_acc

#train the model
n_epochs = 100

model.train()

train_loss_arr = []
for epoch in range(n_epochs):
    
    train_loss = 0
    for x, y in train_loader:
        """ Step 1. clear gradients """
        optimizer.zero_grad()
        """ 
        TODO: Step 2. perform forward pass using `model`, save the output to y_hat;
              Step 3. calculate the loss using `criterion`, save the output to loss.
        """
        y_hat = model.forward(x)
        loss = criterion(y_hat, y, target=torch.ones(y_hat.shape[0]))
        # your code here
        #raise NotImplementedError
        """ Step 4. backward pass """
        loss.backward()
        """ Step 5. optimization """
        optimizer.step()
        """ Step 6. record loss """
        train_loss += loss.item()
        
    train_loss = train_loss / len(train_loader)
    if epoch % 20 == 0:
        train_loss_arr.append(np.mean(train_loss))
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        evaluate(model, test_loader, concepts_embed)