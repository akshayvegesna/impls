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

        self.n_experts = 4

# https://nn.labml.ai/transformers/switch/index.html used as reference.
class SwitchLinear(torch.nn.Module):
    def __init__(self, config): 
        super().__init__()
        self.config = config

        self.capacity_factor = 1.2 
        self.n_experts = config.n_experts

        experts = []
        for i in range(self.n_experts):
            experts.append(torch.nn.Linear(config.embedding_dim, config.embedding_dim))
        self.experts = torch.nn.ModuleList(experts)

        self.switch = torch.nn.Linear(config.embedding_dim, self.n_experts)

    def forward(self, x): 
        b, seq, embed_dim = x.shape 
        x = x.view(-1, embed_dim)

        probs = torch.nn.functional.softmax(self.switch(x), dim=-1)
        route_probs, routes = torch.max(probs, dim=-1)
        indices = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]

        
        dropped = []
        capacity = int(self.capacity_factor * len(x) / self.n_experts)

        for i in range(self.n_experts): 
            if len(indices[i]) <= capacity: 
                continue 
            indices[i] = indices[i][torch.randperm(len(indices[i]))]
            dropped.append(indices[i][capacity:])
            indices[i] = indices[i][:capacity]

        expert_output = []
        for i in range(self.n_experts): 
            expert_output.append(self.experts[i](x[indices[i], :]))

        final_output = torch.zeros_like(x)
        for i in range(self.n_experts): 
            final_output[indices[i], :] = expert_output[i]

        if dropped: 
            dropped = torch.cat(dropped) 
            final_output[dropped, :] = x[dropped, :]
        
        final_output = final_output * route_probs.view(-1, 1)
        final_output = final_output.view(b, seq, embed_dim)

        # TODO: Log the rest of it.
        return final_output

class TransformerLayer(torch.nn.Module): 
    def __init__(self, config): 
        super().__init__()
        self.config = config 
        self.heads = torch.nn.ModuleDict({})
        for i in range(config.num_heads): 
            embed_per_head = torch.tensor(int(config.embedding_dim / config.num_heads), dtype=torch.long)
            self.heads.update(
                {
                    f"v_{i}": torch.nn.Linear(config.embedding_dim, embed_per_head),
                    f"k_{i}": torch.nn.Linear(config.embedding_dim, embed_per_head), 
                    f"q_{i}": torch.nn.Linear(config.embedding_dim, embed_per_head)
                }
            )
        self.out_proj = SwitchLinear(config)
        self.layer_norm = torch.nn.LayerNorm(config.embedding_dim)

    def forward(self, embed): 
        config = self.config
        hiddens = []
        for i in range(config.num_heads): 
            qs = self.heads[f'q_{i}'](embed)
            ks = self.heads[f'k_{i}'](embed)
            vs = self.heads[f'v_{i}'](embed)

            att = qs @ ks.transpose(-2, -1)
            # causal 
            att = torch.tril(att)
            att = att / torch.sqrt(torch.tensor(config.embedding_dim, dtype=torch.float))
            att = torch.nn.functional.softmax(att, dim=2)
            out = att @ vs 
            hiddens.append(out)

        hiddens = torch.cat(hiddens, dim=2)
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
        n = 1
        
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
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
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