# Pondernet
from torch.utils.data import Dataset, DataLoader 


class ParityDataset(Dataset): 
    def __init__(self): 
        super().__init__() 
        n = 100 
        x = torch.zeros((n, 64))
        y = torch.zeros((n, 1))
        for i in range(n): 
            nums = torch.randint(0, 64, ())
            parity = torch.randint(0, 2, (nums,))
            x[i, :nums] = parity
            y[i] = torch.tensor([parity.sum()%2])
        
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


pd = ParityDataset()
dl = DataLoader(pd, batch_size=8, shuffle=True)