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
python run_llama.py --model huggingface/llama-7b --preallocate --compile no
python run_llama.py --model huggingface/llama-7b --preallocate --compile static
```

You can profile a short run with `--profile`, with the TB logs being stored in `./tb_logs/`

```
python run_llama.py --model huggingface/llama-7b --preallocate --profile
```

## Performance

Running the Llama with the commands above gives, with `batch_size=1`, `prompt_length=1000`, `new_tokens=200`,
`cache_length=1200`, `dtype=fp16`:

| changes                                                     | compile | tok_per_s | max_mem_mb | hash     | commit                                   |
|-------------------------------------------------------------|---------|-----------|------------|----------|------------------------------------------|
| None                                                        | no      | 23.150    | 14776.09   | 0d6aa042 | /                                        |
| Preallocated KV cache + SDPA + shared key/value linear      | no      | 27.329    | 14249.72   | 0d6aa042 | 300840e4a6531d44d7129d341b6a24cf63947807 |
| above + preallocated attention_mask                         | no      | 27.377    | 14247.73   | 0d6aa042 | 67a933cb02def42f1fe98cc57d5077b976f1f51f |
| above + shared query/key/value linear                       | no      | 27.444    | 14247.79   | 0d6aa042 | f2e5881e8cf6d0e89f35356ff745e8bb02cb7ebc |
| above + `valid_past_index` as tensor + removed controlflows | no      | 27.166    | 14248.19   | 0d6aa042 | 83ca672ec3c0f2c93e70da6d79bafdeb7c2f7e90 |
| above      | yes (`dynamic=False`)     | 29.139    | 14223.17   | 0d6aa042 | 9c51dc0f10df27189141b1f98823ffba214f7e08 |
| above + avoid torch.cat in rotate_half     | yes (`dynamic=False`)     | 29.385    | 14223.17   | 0d6aa042 | cff4a09323048565961b26252183c947b2d8c51b |

The `hash` is used to "make sure" the implementation is on par with transformers.

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
