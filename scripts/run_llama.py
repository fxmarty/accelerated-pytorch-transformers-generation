import argparse
import copy
import contextlib
import hashlib
import os
import traceback
from typing import Dict

from tqdm import tqdm
import pandas as pd
import torch
import git
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

from transformers import AutoModelForCausalLM, AutoTokenizer

from trfs_fast.llama import LlamaForCausalLM
from trfs_fast.utils import recurse_getattr, recurse_hasattr, recurse_setattr, recurse_delattr


# Default case
BATCH_SIZES = [1]
PROMPT_LENGTHS = [1000]
NEW_TOKENS = [200]
WARMUP_RUNS = 2
NUM_RUNS = 5

# Modifiers for profiling (we want a short run)
PROFILE_NEW_TOKENS = [10]
PROFILE_NUM_RUNS = 1

# Modifiers for parameter sweeps
SWEEP_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
SWEEP_PROMPT_LENGTHS = [100, 200, 400, 800, 1600]
SWEEP_NUM_RUNS = 10

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
    "--profile",
    action='store_true',
    help="Does a storter run for profiling purposes",
)
parser.add_argument(
    "--compile",
    type=str,
    choices=["no", "static", "dynamic", "fullgraph"],
    default="no",
    help="If (and how) to compile the model forward pass with torch.compile",
)
parser.add_argument(
    "--sweep",
    type=str,
    choices=["", "batch", "length"],
    required=False,
    default="",
    help="Select which type of sweep to gather data for",
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
    do_profile: bool,
):
    warmup_start_event = torch.cuda.Event(enable_timing=True)
    warmup_end_event = torch.cuda.Event(enable_timing=True)

    if preallocate:
        inputs["cache_length"] = cache_length

    with torch.no_grad():
        print(f"Warming up ({WARMUP_RUNS} runs)...")
        warmup_start_event.record()
        for _ in range(WARMUP_RUNS):
            res = generate_method(
                **inputs,
                min_new_tokens=max_new_tokens,
                max_new_tokens=max_new_tokens,
            )
        warmup_end_event.record()
        torch.cuda.synchronize()
        print(f"Warmup/compilation time: {warmup_start_event.elapsed_time(warmup_end_event) * 1.0e-3:.2f} seconds ({WARMUP_RUNS} generate calls)")

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if do_profile:
            profile_dir = "./tb_logs"
            print("Profiling and writing to", profile_dir)
            cm = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                on_trace_ready=tensorboard_trace_handler(dir_name=profile_dir),
            )
        else:
            cm = contextlib.nullcontext()

        with cm:
            start_event.record()

            for _ in tqdm(range(num_runs), desc="Measuring generate"):
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
                    copy.deepcopy(recurse_getattr(original_model, base_root_key)),
                    copy.deepcopy(recurse_getattr(original_model, base_root_value))
                ], dim=0))

                recurse_setattr(model, name, weight)

        del original_model
else:
    with device:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)


if args.compile != "no":
    dynamic = args.compile == "dynamic"
    fullgraph = args.compile == "fullgraph"
    mode = "reduce-overhead" if not args.sweep else "max-autotune"
    model.forward = torch.compile(model.forward, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

if model.config.model_type != "llama":
    raise ValueError("This script currently only supports LLAMA")

if args.profile and args.sweep:
    raise ValueError("Cannot profile and sweep at the same time")
batch_sizes = BATCH_SIZES if args.sweep != "batch" else SWEEP_BATCH_SIZES
prompt_lengths = PROMPT_LENGTHS if args.sweep != "length" else SWEEP_PROMPT_LENGTHS
new_tokens = NEW_TOKENS if not args.profile else PROFILE_NEW_TOKENS
num_runs = NUM_RUNS
if args.profile:
    num_runs = PROFILE_NUM_RUNS
elif args.sweep:
    num_runs = SWEEP_NUM_RUNS

stats = {}
for batch_size in tqdm(batch_sizes):
    for prompt_length in tqdm(prompt_lengths):
        for max_new_tokens in tqdm(new_tokens):
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
            try:
                time_per_generation, max_memory, sha_hash = timing_cuda(
                    tokenizer=tokenizer,
                    num_runs=num_runs,
                    inputs=inp,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    cache_length=cache_length,
                    generate_method=generate_method,
                    preallocate=args.preallocate,
                    do_profile=args.profile,
                )
            except:
                traceback.print_exc()
                break  # in a sweep, might get OOM

            tok_per_s = (max_new_tokens * batch_size) / time_per_generation

            stats[(batch_size, prompt_length, max_new_tokens)] = {
                "cache_length": cache_length,
                "tok_per_s": tok_per_s,
                "hash": sha_hash[:8],
                "max_mem": max_memory
            }

# build dataframe with the results and store it
rows = []
repo = git.Repo(search_parent_directories=True)
current_git_hash = repo.git.rev_parse(repo.head, short=True)
for key, value in stats.items():
    batch_size, prompt_length, new_tokens = key
    rows.append({
        'Preallocate': args.preallocate,
        'Compile': args.compile,
        'Batch size': str(batch_size),
        'Prompt length': str(prompt_length),
        'New tokens': str(new_tokens),
        'Cache length': str(value["cache_length"]),
        'Weights dtype': args.dtype,
        'Tokens per second': f"{value['tok_per_s']:.3f}",
        'Max GPU memory (MB)': f"{value['max_mem']:.2f}",
        'Results hash': value["hash"],
        'Git hash': current_git_hash,
    })
df = pd.DataFrame(rows)
print(df)

os.makedirs("./results", exist_ok=True)
output_path = "./results/results_llama.csv"
df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
print(f"Results also appended to {output_path}")
