
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import tqdm 
import random 



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return x



class MnistContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):

        # let's just use the first 1000 samples
        images_per_class = {i: [] for i in range(10)}
        for i in tqdm.tqdm(range(len(dataset))):
            image, idx = dataset[i]
            images_per_class[idx].append(image)
            
        # we learn the trend 6 or, not 
        positives = images_per_class[5] 
        negatives = [images_per_class[i] for i in range(10) if i != 5]
        negatives = [item for sublist in negatives for item in sublist]


        random.seed(42)
        data = []
        for i in range(len(positives)):

            negatives_i = random.sample(negatives, 2) 
            
            data.append((positives[i], *negatives_i))
        print(f'Loaded {len(data)} samples.')

        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

            
        


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, examples in enumerate(train_loader):
        examples = [example.to(device) for example in examples]
        optimizer.zero_grad()

        # positives 
        outputs = []
        for example in examples:
            x = model(example)
            outputs.append(x) 

        raw_outputs = torch.concat(outputs, dim=1)
        outputs = F.log_softmax(raw_outputs, dim=1)
        # print('>outputs', outputs)
        target = torch.zeros(outputs.shape[0], device=device, dtype=torch.long)
        loss = F.nll_loss(outputs, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # norm of x 
            # print('Average norm of x: ', x.mean().item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Avg norm x {:.6f}'.format(
                epoch, batch_idx * len(examples), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), outputs.mean().item()))
            print('Positive average:', raw_outputs[:, 0].mean().item())
            print('Negative average:', raw_outputs[:, 1:].mean().item())

            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, examples in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    raw_dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    raw_dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    dataset1 = MnistContrastiveDataset(raw_dataset1)
    dataset2 = MnistContrastiveDataset(raw_dataset2)
    
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        # test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()