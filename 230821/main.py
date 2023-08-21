from torch_geometric.datasets import Planetoid

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.loader import DataLoader


dataset = Planetoid(root='/tmp/Cora', name='Cora')


# In each GCNConv pass, there is a 
# - linear projection X' = XW 
# - normalization - c_ij = 1/sqrt(d_i * d_j) where d_i is the degree of node i
# - aggregation - X'' = /sum_j c_ij * X'_j
# Accuracy: 0.7860
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# In each GATConv pass, there is a 
# - linear projection X' = XW 
# - attention - e_ij = LeakyReLU(a^T [Wx_i || Wx_j])
# - normalization - c_ij = exp(e_ij) / sum_k/in_N(i) exp(e_ik)
# - aggregation - X'' = /sum_j c_ij * X'_j
# Accuracy: 0.7820
class GAT(torch.nn.Module): 
    def __init__(self): 
        super().__init__()
        self.conv1 = GATConv(dataset.num_node_features, 16)
        self.conv2 = GATConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index) 
        x = F.relu(x) 
        x = F.dropout(x, training=self.training) 
        x = self.conv2(x, edge_index) 
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loader = DataLoader(dataset, batch_size=4, shuffle=False)

epochs = 100
for epoch in range(epochs):
    model.train()
    for i, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')

model.eval()

num_correct = 0
total = 0
for data in loader:
    data = data.to(device)
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    num_correct += correct
    total += data.test_mask.sum()
    
acc = int(num_correct) / int(total)
print(f'Accuracy: {acc:.4f}')
