## Install

```
pip install -e .
```

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

Running this repo model (as of [67a933c](https://github.com/fxmarty/accelerated-pytorch-transformers-generation/commit/67a933cb02def42f1fe98cc57d5077b976f1f51f)) & generation:

```
python run_llama.py --model huggingface/llama-7b --preallocate yes
```

gives

```
batch_size,prompt_length,new_tokens,cache_length,dtype,tok_per_s,max_mem_mb,hash
1,1000,200,1200,fp16,27.377,14247.73,0d6aa042
```

The `hash` is used to "make sure" the implementation is on par with transformers

The default

```python
BATCH_SIZES = [1]
PROMPT_LENGTHS = [1000]
NEW_TOKENS = [200]
```

can be edited to run a sweep, for example:

```python
BATCH_SIZES = [1, 2, 4, 8]
PROMPT_LENGTHS = [500, 1000, 4000]
NEW_TOKENS = [1000]
```
