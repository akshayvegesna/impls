


from model import MixtralForCausalLM
from dataset import AdditionDataset
import itertools 
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader

import torch


@torch.no_grad()
def run_eval(model, tokenizer): 
    ds = AdditionDataset(2, 'train', tokenizer)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    dl = itertools.cycle(dl)
    model.eval()
    print('Running evaluation...')

    total_loss = 0 

    for i in range(100):
        input_ids, attention_mask = next(dl)
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')

        # compute the loss
        outs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        total_loss += outs.loss.item() 

        # truncate to remove the answer for generation.
        first_nonzero = attention_mask[0].nonzero()[0][0]
        input_ids = input_ids[:, :first_nonzero+1]
        attention_mask = attention_mask[:, :first_nonzero+1]

        if i < 10:
            # greedy sampling
            for j in range(5):
                out = model(input_ids, attention_mask=attention_mask).logits[:, -1, :]
                next_token = torch.argmax(out, dim=-1).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            print(f'SAMPLE {i}')
            print(tokenizer.decode(input_ids[0]))
    avg_loss = total_loss / 100
    print(f"Average loss: {avg_loss}")
    model.train()
    print('Eval finished.')
    return avg_loss

def train():
    MAX_TRAIN_STEPS = 10000
    EVAL_EVERY_N = 1000
    LOG_LOSS_EVERY_N = 10

    model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

    model_config = AutoConfig.from_pretrained(model_name)
    model_config.num_hidden_layers = 1
    print('Model loading...')
    model = MixtralForCausalLM(model_config)
    print('Model loaded..')
    model.to('cuda')
    model.train()
    print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

    dataset = AdditionDataset(2, 'train', tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataloader = itertools.cycle(dataloader)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    total_train_loss = 0.0
    run_eval(model, tokenizer)
    

    for i in range(MAX_TRAIN_STEPS):
        input_ids, attention_mask = next(dataloader)
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        optimizer.zero_grad()
        loss = model(input_ids, attention_mask=attention_mask, labels=input_ids).loss
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        if i % LOG_LOSS_EVERY_N == 0:
            print(f"Step {i}, Loss: {total_train_loss / LOG_LOSS_EVERY_N}")
            total_train_loss = 0.0

        if i % EVAL_EVERY_N == 0:
            run_eval(model, tokenizer)

    run_eval(model, tokenizer)






if __name__ == '__main__': 
    train()