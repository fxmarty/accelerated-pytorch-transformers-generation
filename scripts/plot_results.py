"""
Plots the results of a sweep for the current git hash.
"""
import argparse

import git
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


DEFAULT_BATCH_SIZE = 1
DEFAULT_PROMPT_LENGTH = 1000


parser = argparse.ArgumentParser()
parser.add_argument(
    "--sweep",
    type=str,
    choices=["batch", "length"],
    required=True,
    help="Select which type of sweep to plot"
)
args = parser.parse_args()

# 1. Read file and retrieve relevant data
results_file = "./results/results_llama.csv"
df = pd.read_csv(results_file)

repo = git.Repo(search_parent_directories=True)
current_git_hash = repo.git.rev_parse(repo.head, short=True)
df = df[df["Git hash"] == current_git_hash]
if df.empty:
    raise ValueError(f"No results found for current git hash ({current_git_hash})")

if args.sweep == "batch":
    df = df[df["Prompt length"] == DEFAULT_PROMPT_LENGTH]
else:
    df = df[df["Batch size"] == DEFAULT_BATCH_SIZE]
    df = df[df["Prompt length"] != DEFAULT_PROMPT_LENGTH]

if df.empty:
    raise ValueError("Something went wrong -- no results in the filtered dataframe")

# 2. Plot -- we expect 3 series: original model, preallocated, and preallocated + compiled
if args.sweep == "batch":
    x_col_name = "Batch size"
else:
    x_col_name = "Prompt length"

df["Type"] = df["Preallocate"].astype("str") + df["Compile"]
df["Type"] = df["Type"].replace({"Falseno": "original", "Trueno": "Preallocate", "Truestatic": "Pre + comp."})

g = sns.catplot(
    data=df,
    kind="bar",
    x=x_col_name,
    y="Tokens per second",
    hue="Type",
    palette={"original": "blue", "Preallocate": "orange", "Pre + comp.": "red"},
    alpha=.9,
)
g.despine(left=True)
g.set_axis_labels("Batch size" if args.sweep == "batch" else "Prompt length", "Tokens per second")
g.legend.set_title("LLaMA code version")
plt.setp(g._legend.get_texts(), fontsize='7')  # for legend text

title_constant = f'{"Batch size = " + str(DEFAULT_BATCH_SIZE) if args.sweep == "length" else "Prompt length = " + str(DEFAULT_PROMPT_LENGTH)}'
g.set(title=f'LLaMA sweep ({title_constant})')

# Add the number to the top of each bar
ax = g.facet_axis(0, 0)
for i in ax.containers:
    ax.bar_label(i, fontsize=7)

g.tight_layout()
plt_path = f"./results/llama_sweep_{current_git_hash}_{args.sweep}.png"
plt.savefig(plt_path, dpi=300)
print(f"Plot stored at {plt_path}")
