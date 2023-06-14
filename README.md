## Running LLAMA

Below on AMD EPYC 7R32 + A10G (g5.2xlarge).

Running transformers model & generation:

```
python run_llama.py --model huggingface/llama-7b --preallocate no
```

gives
```
batch_size,prompt_length,new_tokens,cache_length,dtype,tok_per_s,max_mem_mb,hash
1,1000,200,1200,fp16,23.150,14776.09,0d6aa042
```

Running this repo model & generation:

```
python run_llama.py --model huggingface/llama-7b --preallocate yes
```

gives

```
batch_size,prompt_length,new_tokens,cache_length,dtype,tok_per_s,max_mem_mb,hash
1,1000,200,1200,fp16,27.329,14249.72,0d6aa042
```

The `hash` is used to "make sure" (see TODOS) the implementation is on par with transformers

## TODOS

- [ ] Experiment with `torch.compile`
- [ ] Experiment with iteratively allocated KV cache, see [this suggestion](https://github.com/huggingface/text-generation-inference/issues/376)
- [ ] Can we avoid aten::copy_ calls and aten::slice calls?
- [ ] Test on CPU
- [ ] Support cross-attention
- [ ] Support encoder-decoder architectures
- [ ] Support preallocated `attention_mask`, `token_type_ids`
- [ ] Would a single `qkv_proj` help instead of the current `q_proj` separate from `kv_proj`?
- [ ] Make sure the implementation is valid not on the argmax but on the logits directly
