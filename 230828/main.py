import torch 
from torch.utils.data import Dataset, DataLoader 
torch.random.manual_seed(0)
import itertools

# vocab size is just numbers from 0 to 10 and some simple tokens.
vocab = '0123456789+= '
vocab_size = len(vocab)
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}

class Config: 
    def __init__(self): 
        self.vocab_size = vocab_size 
        self.block_size = 128
        self.embedding_dim = 512
        self.num_heads = 8
        self.num_layers = 6

class TransformerLayer(torch.nn.Module): 
    def __init__(self, config): 
        super().__init__()
        self.config = config 

        self.head_dim = config.embedding_dim // config.num_heads

        self.v = torch.nn.Linear(config.embedding_dim, config.embedding_dim)
        self.k = torch.nn.Linear(config.embedding_dim, config.embedding_dim)
        self.q = torch.nn.Linear(config.embedding_dim, config.embedding_dim)

        self.out_proj = torch.nn.Linear(config.embedding_dim, config.embedding_dim)
        self.layer_norm = torch.nn.LayerNorm(config.embedding_dim)

    def forward(self, embed): 
        b, t, c = embed.shape
        config = self.config
        hiddens = []
        
        # Split weight matrices into heads. 
        ks = self.k(embed)
        ks = ks.view(b, t, config.num_heads, self.head_dim)
        
        qs = self.q(embed)
        qs = qs.view(b, t, config.num_heads, self.head_dim)

        vs = self.v(embed)
        vs = vs.view(b, t, config.num_heads, self.head_dim)

        att = qs @ ks.transpose(-2, -1)
        att = att / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        att = torch.nn.functional.softmax(
            att, dim=-1
        )
        hiddens = att @ vs
        hiddens = hiddens.view(b, t, config.embedding_dim)
        x = self.out_proj(hiddens)
        x += embed 
        x = self.layer_norm(x)
        return x

class Transformer(torch.nn.Module):
    def __init__(self, config=Config()): 
        super().__init__()
        self.config = config 

        self.wte = torch.nn.Embedding(config.vocab_size, config.embedding_dim)
        self.wpe = torch.nn.Embedding(config.block_size, config.embedding_dim)

        layers = []
        for i in range(config.num_layers): 
            layers.append(TransformerLayer(config))
        self.layers = torch.nn.ModuleList(layers)

        self.lm_head = torch.nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, token_idxs, label=None): 
        config = self.config
        device = token_idxs.device
        b, t = token_idxs.shape 

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_embed = self.wte(token_idxs)
        pos_embed = self.wpe(pos)
        embed = tok_embed + pos_embed

        for layer in self.layers: 
            embed = layer(embed)

        logits = self.lm_head(embed)

        loss = None
        if label is not None: 
            loss = torch.nn.functional.cross_entropy(input=logits[:, -1, :], target=label.view(-1))

        return logits, loss  

class AdditionDataset: 
    def __init__(self): 
        min_value = 1 
        max_value = 20
        n = 10
        
        # adjust based on dataset complexity.
        self.max_size = 12

        self.str_examples = []
        for _ in range(n):
            a = torch.randint(min_value, max_value, (1,))
            b = torch.randint(min_value, max_value, (1,))
            c = a + b
            sample = str(a.item()) + '+' + str(b.item()) + '=' + str(c.item()) + ' '
            self.str_examples.append(sample)  
        
        self.examples = []
        for ex in self.str_examples: 
            for i in range(1, len(ex)): 
                inp = ex[:i]
                out = ex[i]

                encoded_input = torch.tensor([stoi[c] for c in inp])
                encoded_input = torch.nn.functional.pad(encoded_input, (0, self.max_size - len(encoded_input)))
                encoded_output = torch.tensor([stoi[c] for c in out])
                self.examples.append({
                    "input": encoded_input,
                    "target": encoded_output
                })
            
    def __len__(self): 
        return len(self.examples)      
    
    def __getitem__(self, idx): 
        return self.examples[idx]

model = Transformer()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4) 
# 6388749
print('number of parameters: {}'.format(sum(p.numel() for p in model.parameters())))

dataset = AdditionDataset() 
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
epochs = 10

for epoch in range(epochs):
    for i, batch in enumerate(dataloader): 
        input = batch['input']
        target = batch['target']
        optimizer.zero_grad()
        logits, loss = model(input, target)
        loss.backward() 
        optimizer.step()

        if i % 100 == 0:
            print('i: {} loss: {}'.format(i, loss.item()))
    print('epoch: {} loss: {}'.format(epoch, loss.item()))

def generate_one_token(input): 
    mapped_inp = torch.tensor([[stoi[c] for c in input]])
    logits, _ = model(mapped_inp)
    logits = logits[0, -1, :]
    probs = torch.nn.functional.softmax(logits, dim=0)
    next_token = torch.multinomial(probs, num_samples=1)
    next_token = itos[next_token.item()]
    print('generated input {} token: {}'.format(input, next_token))
    return next_token

def generate(input, max_len=10):
    for _ in range(max_len): 
        next_token = generate_one_token(input)
        input += next_token
        if next_token == ' ': 
            break
    return input

gen_1 = generate('6+16=')
print(gen_1)