from tqdm import tqdm

import torch

from typing import Tuple

import hashlib
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from trfs_prealloc.llama import LlamaForCausalLM

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="gpt2",
    help="",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="fp16",
    help="",
)
parser.add_argument(
    "--preallocate",
    type=str,
    help="",
    choices=["yes", "no"],
    required=True
)


def timing_cuda_minimal(
    model: torch.nn.Module,
    tokenizer,
    num_runs: int,
    inputs: torch.LongTensor,
    max_new_tokens: int,
    device: torch.device,
    cache_length: int,
):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    config = model.config

    with torch.inference_mode():
        res = model.generate_minimal(
            **inputs,
            min_new_tokens=max_new_tokens,
            max_new_tokens=max_new_tokens,
            cache_length=cache_length,
        )

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        start_event.record()

        for _ in tqdm(range(num_runs)):
            res = model.generate_minimal(
                **inputs,
                min_new_tokens=max_new_tokens,
                max_new_tokens=max_new_tokens,
                cache_length=cache_length,
            )

        end_event.record()
        torch.cuda.synchronize()
        max_memory = torch.cuda.max_memory_allocated(device)

    h = hashlib.new('sha256')
    h.update(str(tokenizer.batch_decode(res)).encode())

    sha_hash = h.hexdigest()

    return (start_event.elapsed_time(end_event) * 1.0e-3) / num_runs, max_memory * 1e-6, sha_hash

def timing_cuda_trfs(
    model: torch.nn.Module,
    tokenizer,
    num_runs: int,
    inputs: torch.LongTensor,
    max_new_tokens: int,
    device: torch.device,
    cache_length: int,
):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    config = model.config

    with torch.inference_mode():
        res = model.generate(
            **inputs,
            min_new_tokens=max_new_tokens,
            max_new_tokens=max_new_tokens,
        )

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        start_event.record()

        for _ in tqdm(range(num_runs)):
            res = model.generate(
                **inputs,
                min_new_tokens=max_new_tokens,
                max_new_tokens=max_new_tokens,
            )

        end_event.record()
        torch.cuda.synchronize()
        max_memory = torch.cuda.max_memory_allocated(device)

    h = hashlib.new('sha256')
    h.update(str(tokenizer.batch_decode(res)).encode())

    sha_hash = h.hexdigest()

    return (start_event.elapsed_time(end_event) * 1.0e-3) / num_runs, max_memory * 1e-6, sha_hash

args = parser.parse_args()

torch.manual_seed(42)

if args.dtype == "fp16":
    dtype = torch.float16
elif args.dtype == "fp32":
    dtype = torch.float32
else:
    raise ValueError("Choose fp16 or fp32 dtype")

device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token

header = "batch_size,prompt_length,new_tokens,cache_length,dtype,tok_per_s,max_mem_mb,hash"
stats = {}

if args.preallocate == "yes":
    preallocate = True
else:
    preallocate = False

with device:
    if preallocate:
        model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)


BATCH_SIZES = [1]
PROMPT_LENGTHS = [200]
NEW_TOKENS = [800]

for batch_size in tqdm(BATCH_SIZES):
    for prompt_length in tqdm(PROMPT_LENGTHS):
        for max_new_tokens in tqdm(NEW_TOKENS):
            cache_length = 1 * (prompt_length + max_new_tokens)

            inp = {
                "input_ids": torch.randint(low=1, high=10, size=(batch_size, prompt_length)).to("cuda"),
                "attention_mask": torch.ones(batch_size, prompt_length, dtype=torch.int32).to("cuda")
            }

            h = hashlib.new('sha256')
            h.update(str(inp).encode())
            print("Input hash:", h.hexdigest()[:8])
            print("Cache preallocation:", preallocate)

            timing_func = timing_cuda_minimal if preallocate else timing_cuda_trfs

            time_per_generation, max_memory, sha_hash = timing_func(
                model=model,
                tokenizer=tokenizer,
                num_runs=5,
                inputs=inp,
                device=device,
                max_new_tokens=max_new_tokens,
                cache_length=cache_length,
            )

            tok_per_s = max_new_tokens / time_per_generation

            stats[(batch_size, prompt_length, max_new_tokens)] = {
                "cache_length": cache_length,
                "tok_per_s": tok_per_s,
                "hash": sha_hash[:8],
                "max_mem": max_memory
            }

# print csv
print(header)
for key, value in stats.items():
    batch_size, prompt_length, new_tokens = key
    print(",".join([str(batch_size), str(prompt_length), str(new_tokens), str(value["cache_length"]), args.dtype, f"{value['tok_per_s']:.3f}", f"{value['max_mem']:.2f}", value["hash"]]))
