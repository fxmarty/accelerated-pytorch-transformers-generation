## Install

```
pip install -e .
```

## Running LLAMA

Below on AMD EPYC 7R32 + A10G (g5.2xlarge).

Running default transformers model & generation:

```
python run_llama.py --model huggingface/llama-7b
```

Adding flags will change the behavior of text generation (use --help for the available flags):

```
python run_llama.py --model huggingface/llama-7b --preallocate --compile static
```

gives

|changes                                               |batch_size|prompt_length|new_tokens|cache_length|dtype|tok_per_s|max_mem_mb|hash    |
|------------------------------------------------------|----------|-------------|----------|------------|-----|---------|----------|--------|
|None                                                  |1         |1000         |200       |1200        |fp16 |23.150   |14776.09  |0d6aa042|
|Preallocated KV cache + SDPA + shared key/value linear|1         |1000         |200       |1200        |fp16 |27.329   |14249.72  |0d6aa042|
|above + preallocated attention_mask                   |1         |1000         |200       |1200        |fp16 |27.377   |14247.73  |0d6aa042|
|above + shared query/key/value linear                 |1         |1000         |200       |1200        |fp16 |27.444   |14247.79  |0d6aa042|

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
