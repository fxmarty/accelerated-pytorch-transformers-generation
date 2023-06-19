from tqdm import tqdm

import torch

from typing import Tuple, Dict

import hashlib
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from trfs_fast.llama import LlamaForCausalLM
from torch.profiler import ProfilerActivity, profile, record_function, tensorboard_trace_handler

from trfs_fast.utils import recurse_getattr, recurse_hasattr, recurse_setattr, recurse_delattr

import argparse
import copy


parser = argparse.ArgumentParser()

# TODO: support other archs than llama
parser.add_argument(
    "--model",
    type=str,
    default="huggingface/llama-7b",
    help="Name of the weights on the Hub",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="fp16",
    help="Type of the weights that will be used at test time",
)
parser.add_argument(
    "--preallocate",
    action='store_true',
    help="[TRIGGERS NEW CODE PATH] Whether to preallocate internal model tensors",
)
parser.add_argument(
    "--compile",
    action='store_true',
    help="Whether to compile the model forward pass with torch.compile",
)


def timing_cuda(
    tokenizer,
    generate_method,
    num_runs: int,
    inputs: Dict,
    max_new_tokens: int,
    device: torch.device,
    cache_length: int,
    preallocate: bool,
):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    if preallocate:
        inputs["cache_length"] = cache_length

    res = generate_method(
        **inputs,
        min_new_tokens=max_new_tokens,
        max_new_tokens=max_new_tokens,
    )

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    """
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=tensorboard_trace_handler(f"./tb_logs/preallocate_True"),
    ):
    """
    start_event.record()

    for _ in tqdm(range(num_runs)):
        res = generate_method(
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


if args.preallocate:
    with device:
        original_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    with torch.device("meta"):
        model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

        # replace back parameters and buffers that were untouched by the bettertransformer transform
        for path, param in model.state_dict().items():
            if "k_proj" not in path and "v_proj" not in path and "q_proj" not in path and "min_allowed" not in path:
                recurse_setattr(model, path, copy.deepcopy(recurse_getattr(original_model, path)))

                recurse_delattr(original_model, path)  # save mem

        # some buffers may be non-persistent, hence not in the state_dict (as token_type_ids for some models)
        for path, param in model.named_buffers():
            if "k_proj" not in path and "v_proj" not in path and "q_proj" not in path and "min_allowed" not in path:
                if recurse_hasattr(original_model, path):
                    recurse_setattr(model, path, copy.deepcopy(recurse_getattr(original_model, path)))

                    recurse_delattr(original_model, path)  # save mem
            if "min_allowed" in path:
                recurse_setattr(model, path, torch.tensor(torch.finfo(dtype).min, device=device))

        for name, module in model.named_parameters():
            if "qkv_proj" in name:
                base_root_query = ".".join(name.split(".")[:-2]) + ".q_proj.weight"
                base_root_key = ".".join(name.split(".")[:-2]) + ".k_proj.weight"
                base_root_value = ".".join(name.split(".")[:-2]) + ".v_proj.weight"
                root = ".".join(name.split(".")[:-1]) + ".weight"

                weight = torch.nn.Parameter(torch.cat([
                    copy.deepcopy(recurse_getattr(original_model, base_root_query)),
                    copy.deepcopy(recurse_getattr(original_model, base_root_key)), copy.deepcopy(recurse_getattr(original_model, base_root_value))
                ], dim=0))

                recurse_setattr(model, name, weight)

        del original_model
else:
    with device:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)


if args.compile:
    model.forward = torch.compile(model.forward, mode="reduce-overhead", dynamic=True)

if model.config.model_type != "llama":
    raise ValueError("This script currently only supports LLAMA")

BATCH_SIZES = [1]
PROMPT_LENGTHS = [1000]
NEW_TOKENS = [200]

for batch_size in tqdm(BATCH_SIZES):
    for prompt_length in tqdm(PROMPT_LENGTHS):
        for max_new_tokens in tqdm(NEW_TOKENS):
            cache_length = 1 * (prompt_length + max_new_tokens)

            inp = {
                "input_ids": torch.randint(low=1, high=10, size=(batch_size, prompt_length)).to("cuda"),
                "attention_mask": torch.ones(batch_size, prompt_length, dtype=torch.int32).to("cuda")
            }

            if batch_size > 1:
                inp["input_ids"][0, :10] = tokenizer.pad_token_id
                inp["attention_mask"][0, :10] = 0

            h = hashlib.new('sha256')
            h.update(str(inp).encode())
            print("\nInput hash:", h.hexdigest()[:8])
            print("Cache preallocation:", args.preallocate)

            generate_method = model.generate if not args.preallocate else model.generate_minimal
            time_per_generation, max_memory, sha_hash = timing_cuda(
                tokenizer=tokenizer,
                num_runs=3,
                inputs=inp,
                device=device,
                max_new_tokens=max_new_tokens,
                cache_length=cache_length,
                generate_method=generate_method,
                preallocate=args.preallocate,
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
