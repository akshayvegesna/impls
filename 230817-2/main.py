from torchtext.datasets import CNNDM
from functools import partial
from torch.utils.data import Dataset, DataLoader
import torchdata.datapipes as dp

from torch.optim import Adam
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class CNNDMDataset(Dataset): 
    def __init__(self, num_samples=100):
        super().__init__()

        def make_examples():
            cnndm_dataset = CNNDM(root='./train')
            examples = []
            for ex in cnndm_dataset: 
                for e in ex: 
                    examples.append(e)
                    if len(examples) > num_samples: 
                        break
            return examples
        
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.examples = make_examples()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i): 
        text, summary = self.examples[i]

        task_prefix = "Summarize: "
        text = task_prefix + text 
        encoding = self.tokenizer(text, max_length=2048, padding="max_length", return_tensors="pt")
        target_encoding = self.tokenizer(summary, max_length=2048, padding="max_length", return_tensors="pt")
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask 
        labels = target_encoding.input_ids 
        input_ids = input_ids.squeeze()
        attention_mask = attention_mask.squeeze() 
        labels = labels.squeeze()
        return input_ids, attention_mask, labels

def train(): 
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    dataset = CNNDMDataset(10)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    optimizer = Adam(model.parameters(), lr=3e-4)

    def eval_n_examples(n=5): 
        for i, ex in enumerate(dataloader):
            if i > n: 
                break
            input_ids, attention_mask, labels = ex 
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask) 
            out_decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            print('out decoded: ', out_decoded)

            label_decoded = tokenizer.batch_decode(labels, skip_special_tokens=True)
            print('label decoded: ', label_decoded)

    eval_n_examples()

    epochs = 3

    for idx in range(epochs): 
        for i, ex in enumerate(dataloader):
            input_ids, attention_mask, labels = ex 
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            loss.backward()
            optimizer.step()
            print('i: {}, loss: {}'.format(i, loss.item()))
        eval_n_examples()

if __name__ == '__main__': 
    train() 
