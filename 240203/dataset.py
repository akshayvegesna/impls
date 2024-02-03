from torch.utils.data import Dataset
import torch

# src: https://github.com/karpathy/minGPT/blob/master/projects/adder/adder.py
class AdditionDataset(Dataset):
    """
    Creates n-digit addition problems. For example, if n=2, then an example
    addition problem would be to add 85 + 50 = 135. This problem would be
    represented as the following string for the GPT:

    "8550531"

    This is because:
    - we are discarding the + and =, which are not necessary. We just encode the digits
      of the input numbers concatenated together.
    - the result 135 is encoded backwards to make the addition easier to learn for the
      GPT model, because of how the addition algorithm works.

    As one more example, the problem 6 + 39 = 45 would be encoded as:

    "0639054"

    where you will notice that we are padding with zeros to make sure that we always
    produce strings of the exact same size: n + n + (n + 1). When n=2, this is 7.
    At test time, we will feed in an addition problem by giving the first 2n digits,
    and hoping that the GPT model completes the sequence with the next (n+1) digits
    correctly.
    """

    def __init__(self, ndigit, split, tokenizer=None):
        self.ndigit = ndigit
        self.split = split # train/test

        # split up all addition problems into either training data or test data
        ndigit = ndigit
        assert ndigit <= 3, "the lines below would be very memory inefficient, in future maybe refactor to support"
        num = (10**ndigit)**2 # total number of possible addition problems with ndigit numbers
        rng = torch.Generator()
        rng.manual_seed(1337)
        perm = torch.randperm(num, generator=rng)
        num_test = min(int(num*0.2), 500) # 20% of the whole dataset, or only up to 500
        self.tokenizer = tokenizer
        self.ixes = perm[:10]
        # self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def get_vocab_size(self):
        return 10 # digits 0..9

    def get_block_size(self):
        # a,b,a+b, and +1 due to potential carry overflow,
        # but then also -1 because very last digit doesn't ever plug back
        # as there is no explicit <EOS> token to predict, it is implied
        return 3*self.ndigit + 1 - 1

    def __len__(self):
        return self.ixes.nelement()

    def __getitem__(self, idx):
        ndigit = self.ndigit
        # given a problem index idx, first recover the associated a + b
        idx = self.ixes[idx].item()
        nd = 10**ndigit
        a = idx // nd
        b = idx %  nd
        # calculate the "label" of the addition problem a + b
        c = a + b
        # encode the digits of a, b, c into strings
        astr = f'%0{ndigit}d' % a
        bstr = f'%0{ndigit}d' % b
        cstr = (f'%0{ndigit+1}d' % c)
        prefix = astr + '+' + bstr + '='
        suffix = cstr

        # convert these strings into a sequence of tokens
        prefix_encoded = self.tokenizer.encode(prefix, return_tensors="pt")
        suffix_encoded = self.tokenizer.encode(suffix, return_tensors="pt")[:, 1:]
        input_ids = torch.cat([prefix_encoded, suffix_encoded], dim=1).squeeze()
        attention_mask = torch.cat([torch.zeros_like(prefix_encoded), torch.ones_like(suffix_encoded)], dim=1).squeeze()
        return input_ids, attention_mask

if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1") 
    ds = AdditionDataset(2, 'train', tokenizer)

    print(ds[0])