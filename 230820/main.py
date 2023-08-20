from transformers import AutoTokenizer, GPT2LMHeadModel

import torch 
from torch.utils.data import Dataset, DataLoader
import copy
from torch.optim import Adam


tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# inputs = tokenizer("2+2=", return_tensors="pt")
# outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=40)

# decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)


# Make a small dataset with arithmetic problems. 
class AdditionDataset(Dataset): 
    def __init__(self, n, min_value, max_value): 
        self.n = n
        self.min_value = min_value
        self.max_value = max_value
        self.str_examples = []
        for _ in range(n):
            a = torch.randint(min_value, max_value, (1,))
            b = torch.randint(min_value, max_value, (1,))
            c = a + b
            sample = str(a.item()) + ' + ' + str(b.item()) + ' = ' + str(c.item())
            self.str_examples.append(sample)        

        # Now convert the language modeled objective.
        self.examples = []
        for ex in self.str_examples: 
            inputs = tokenizer(ex, return_tensors="pt", max_length=40, truncation=True, padding="max_length")

            # translated inputs
            tokens = tokenizer.batch_decode([[ele] for ele in inputs['input_ids'][0]])
            # Find the token whose element contains =
            all_equal = [i for i, ele in enumerate(tokens) if '=' in ele]
            assert len(all_equal) == 1
            equal_idx = all_equal[0]

            num_examples = inputs['attention_mask']
            first_zero = (inputs['attention_mask']).sum(dim=1)

            for i in range(equal_idx + 2, first_zero + 1):
                attention_mask = torch.zeros_like(inputs['attention_mask'])
                attention_mask[0, :i] = 1

                input_ids = copy.deepcopy(inputs['input_ids'])
                input_ids[0, i:] = tokenizer.pad_token_id

                self.examples.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': input_ids # The masking is handled inside GPT2LMHead
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

dataset = AdditionDataset(10000, 0, 100)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

optimizer = Adam(model.parameters(), lr=3e-4)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: {}'.format(num_parameters))
print('Len of dataset: {}'.format(len(dataset)))

epochs = 1
for i in range(epochs):
    print('Epoch: {}'.format(i))
    for (batch, sample) in enumerate(dataloader): 
        optimizer.zero_grad()
        out =  model(input_ids=sample['input_ids'], labels=sample['labels'], attention_mask=sample['attention_mask'])
        loss = out.loss
        loss.backward()
        optimizer.step()

        if batch % 10 == 0: 
            print('Batch: {} Loss: {}'.format(batch, loss.item()))


inputs = tokenizer("35 + 71 =", return_tensors="pt")
outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=40)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

