import torch 
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy, relu
from torch.optim import Adam


# 1 Epoch, Accuracy: 83%
class MLP1(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.linear = torch.nn.Linear(784, 10)

    def forward(self, x): 
        x_flat = x.view(-1, 784)
        logits = self.linear(x_flat)

        return logits

# 1 Epoch, Accuracy: 85%
class MLP2(torch.nn.Module): 
    def __init__(self): 
        super().__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 10)
    
    def forward(self, x): 
        x_flat = x.view(-1, 784)
        logits = self.linear2(relu(self.linear1(x_flat)))
        return logits

# 1 epoch, accuracy 87%
class Conv1(torch.nn.Module): 
    def __init__(self): 
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, (3, 3), padding='same')
        self.conv2 = torch.nn.Conv2d(4, 8, (3, 3), padding='same')
        self.conv3 = torch.nn.Conv2d(8, 16, (3, 3), padding='same')
        self.conv4 = torch.nn.Conv2d(16, 8, (3, 3), padding='same')
        self.conv5 = torch.nn.Conv2d(8, 4, (3, 3), padding='same')
        self.linear1 = torch.nn.Linear(3136, 784)
        self.linear2 = torch.nn.Linear(784, 10)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = relu(self.conv3(x))
        x = relu(self.conv4(x))
        x = relu(self.conv5(x))
        x = x.view(batch_size, -1)
        logits = self.linear2(relu(self.linear1(x)))
        return logits

def train():
    train_dataset = torchvision.datasets.FashionMNIST('./train', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]))

    test_dataset = torchvision.datasets.FashionMNIST(root='./test', train=False, 
        transform=transforms.Compose([transforms.ToTensor(),]),download=True)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    model = Conv1()
    optimizer = Adam(model.parameters(), lr=3e-4)
    print('Num parameters, ', sum(p.numel() for p in model.parameters()))

    def run_eval(): 
        correct = 0
        total = 0
        with torch.no_grad(): 
            for input, target in test_dataloader: 
                logits = model(input)
                preds = torch.argmax(logits, dim=1)
                num_correct = (preds == target).sum() 
                correct += num_correct.item() 
                total += target.numel()
        accuracy = correct / total
        print('Accuracy: ', accuracy)

        return accuracy

    epochs = 1
    for _ in range(epochs):
        for i, (inputs, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_logits = model(inputs)
            loss = cross_entropy(input=input_logits, target=targets)
            loss.backward()
            optimizer.step()
            if (i % 100 == 0): 
                print('Epoch: {} Loss: {}'.format(i, loss.item()))
            if (i % 2000 == 0): 
                run_eval()

    
    run_eval()

if __name__ == '__main__': 
    train() 
