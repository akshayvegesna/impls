# Max value difference.
import torch 
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader




class MaxValueDiff(Dataset): 
    def __init__(self, n):
        print
        min_value = 0 
        max_value = 9
        examples = torch.randint(min_value, max_value, (n, 3))

        max_value_diff = torch.max(examples, dim=1)[0] - torch.min(examples, dim=1)[0]
        max_value_diff = max_value_diff.unsqueeze(1)
        
        # One hot encoding 
        examples = torch.nn.functional.one_hot(examples, num_classes=10).float()

        assert examples.shape[0] == max_value_diff.shape[0]
        self.examples = [examples, max_value_diff]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return (self.examples[0][idx], self.examples[1][idx])

    
class DeepSet(torch.nn.Module):
    def __init__(self, input_dim, output_dim): 
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(64, input_dim)
        )
        self.rho = torch.nn.Sequential(
            torch.nn.Linear(output_dim, 64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(64, output_dim)
        )
    
    def forward(self, x): 
        b, n, d = x.shape
        x = x.view(b, -1)
        
        x = self.phi(x)
        x = x.view(b, n, -1)
        x = torch.sum(x, dim=1)
        x = self.rho(x)
        return x


if __name__ == '__main__': 
    n = 10000
    dataloader = DataLoader(MaxValueDiff(n), batch_size=4, shuffle=True)
    model = DeepSet(3*10, 10)
    optimizer = Adam(model.parameters(), lr=1e-2)

    epochs = 10
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader): 
            input = batch[0]
            target = batch[1]
            optimizer.zero_grad()
            logits = model(input)

            loss = torch.nn.functional.cross_entropy(logits, target.view(-1))
            loss.backward()
            optimizer.step()

            if i % 100 == 0: 
                print('i: {} loss: {}'.format(i, loss.item()))
        print('epoch: {} loss: {}'.format(epoch, loss.item()))
    
    # Test
    test_input = torch.nn.functional.one_hot(torch.tensor([[1, 2, 5]]), num_classes=10).float()
    test_target = torch.tensor([[4]])
    logits = model(test_input)
    print('test prediction: {}'.format(torch.argmax(logits, dim=1)))
