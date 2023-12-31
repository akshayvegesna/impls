

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def apply_embed(inp, sd): 
    wte = sd['transformer.wte.weight']
    wpe = sd['transformer.wpe.weight']

    token_emb = wte[inp]
    pos_emb = wpe[torch.arange(0, len(inp))]
    out = token_emb + pos_emb
    return out

def layer_norm(inputs, weights, bias): 
    mean = inputs.mean(-1, keepdim=True)
    std = inputs.std(-1, keepdim=True)
    return (inputs - mean) / (std + 1e-5) * weights + bias

def attention(inp, attn_weight, attn_bias, attn_proj_weight, attn_proj_bias): 
    out = inp @ attn_weight.T + attn_bias
    query, key_value = out.split((2048, 2*128), dim=-1) 
    # q: 4, 2048 
    # k: 4, 256
    key, value = key_value.split((128, 128), dim=-1) # 4, 128

    # run Multi query attention 
    query_length = query.shape[-1]
    query = query.reshape(64, 128)
    attn_weights = query @ key.T / 128**0.5 # 64, 4
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    attn_out = attn_weights @ value # 64, 128
    attn_out = attn_out.reshape(4, 2048)

    attn_out = attn_out @ attn_proj_weight.T + attn_proj_bias

    return attn_out

def apply_mlp(inp, mlp, mlpb, mlp2, mlp2b): 
    out = inp @ mlp.T + mlpb
    out = out @ mlp2.T + mlp2b
    return out

def apply_transformer_block(inp, sd, block_num): 
    prefix = f'transformer.h.{block_num}.'

    ln1 = sd[prefix+'ln_1.weight']
    ln1b = sd[prefix+'ln_1.bias']

    residual = inp

    out = layer_norm(inp, ln1, ln1b)

    attn_weight = sd[prefix+'attn.c_attn.weight']
    attn_bias = sd[prefix+'attn.c_attn.bias']
    attn_proj_weight = sd[prefix+'attn.c_proj.weight']
    attn_proj_bias = sd[prefix+'attn.c_proj.bias']

    out = attention(out, attn_weight, attn_bias, attn_proj_weight, attn_proj_bias)

    out = residual + out

    ln2 = sd[prefix+'ln_2.weight']
    ln2b = sd[prefix+'ln_2.bias']

    out = layer_norm(out, ln2, ln2b)

    mlp = sd[prefix+'mlp.c_fc.weight']
    mlpb = sd[prefix+'mlp.c_fc.bias']
    mlp2 = sd[prefix+'mlp.c_proj.weight']
    mlp2b = sd[prefix+'mlp.c_proj.bias']

    out = apply_mlp(out, mlp, mlpb, mlp2, mlp2b)

    return out

def run_inference(inp, sd):
    out = apply_embed(inp, sd)
    for i in range(24): 
        out = apply_transformer_block(out, sd, i)
    out = layer_norm(out, sd['transformer.ln_f.weight'], sd['transformer.ln_f.bias'])
    out = out @ sd['lm_head.weight'].T
    return out





def main():


    model_id = "bigcode/starcoderbase-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(model_id)
    sd = model.state_dict() 
    text = "Hello my name is"
    inputs = tokenizer(text, return_tensors="pt")
    seq = inputs['input_ids'][0]


    # generate 10 tokens 
    for i in range(1): 
        logits = run_inference(seq, sd)
        idx = torch.argmax(logits[-1], dim=-1)
        seq = torch.cat((seq, idx.unsqueeze(0)), dim=0)

    print(tokenizer.decode(seq))

if __name__ == "__main__":
    main()
