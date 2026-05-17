# Datasets

Downloaded from: https://github.com/ai-prophet/ai-prophet-datasets/tree/main/datasets


## Available Datasets

| Name | File | Tasks | Has Outcomes | Best For |
|------|------|-------|--------------|---------|
| `sports` | `sample-sports/tasks.jsonl` | 16 | No | Testing sports research |
| `economics` | `sample-economics/tasks.jsonl` | 13 | No | Testing finance/politics research |
| `entertainment` | `sample-entertainment/tasks.jsonl` | 13 | No | Testing culture research |
| `resolved` | `sample-resolved/tasks.jsonl` | 26 | **Yes** | **Brier score benchmarking** |
| `small` | built-in stub in run_replay.py | 5 | Yes | Quick sanity check |

> Only `resolved` and `small` have outcomes — required to compute Brier score.
> The other datasets run the full pipeline but show predictions only (no scoring).

## How to Run

```bash
# Quick sanity check (built-in stub, 5 markets)
python scripts/run_replay.py --dataset small

# Real resolved dataset — computes actual Brier score (26 markets)
python scripts/run_replay.py --dataset resolved

# Sports predictions only (no Brier — no outcomes yet)
python scripts/run_replay.py --dataset sports

# Economics / finance
python scripts/run_replay.py --dataset economics

# Entertainment / culture
python scripts/run_replay.py --dataset entertainment

# Limit to first N markets for quick tests
python scripts/run_replay.py --dataset resolved --limit 5

# Custom JSON file
python scripts/run_replay.py --dataset path/to/your/file.json
```
