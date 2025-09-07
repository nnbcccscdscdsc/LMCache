# OpenAI Chat‑Completion TTFT Benchmark  

Measure **time‑to‑first‑token (TTFT)** — and optional cache‑hit latency — from **any
server that speaks the OpenAI `/v1` API** (vLLM, llama.cpp with `--api`,
OpenAI‑proxy, etc.).

> **Why run it?**  
> • Compare *cold* latency vs. *cache‑hit* latency.  
> • Verify whether a KV‑cache (VRAM, SSD, LMCache, …) actually helps.  
> • Collect JSONL you can plot 

---

## 1 · Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Running endpoint** | Must expose the OpenAI REST interface (default URL `http://localhost:8000/v1`). |

For more information on how to serve an endpoint using vllm and LMCache, 

---

## 2 · Command‑line flags

| Flag / shorthand | Default | Meaning |
|------------------|---------|---------|
| `--api_base`     | `http://localhost:8000/v1` | URL of the OpenAI‑style endpoint. |
| `--api_key`      | `EMPTY` | Any string (ignored by most local servers). |
| `--model`        | *first model from* `/models` | Explicit model ID. |
| `-C`, `--context_file` | *see table below* | Document inserted before the prompt. |
| `--max_ctx_tokens` | **131 072** | Upper bound *after* truncation. |
| `--prompt`       | `"Summarize this text"` | Prompt appended after the document. |
| `--num_following`| **1** | Extra TTFT‑measured requests after the baseline. |
| `-F`, `--flush_cache` | off | Flush GPU KV‑cache **once** after run 1. |
| `--out`          | `benchmark.jsonl` | JSONL log (cleared at start). |

### Behaviour of `--context_file`

| Invocation | Document used |
|------------|---------------|
| *(flag omitted)* | Synthetic ASCII filler based on max ctx length input|
| `--context_file` *(no path)* | Bundled `ffmpeg.txt` (one dir up) |
| `--context_file /path/doc.txt` | Exact file you specify |

> **Legacy shorthand** – you may also run  
> `python openai_chat_completion_client.py <PORT>`  
> and every other option remains default.

---

## 3 · Quick start

Cold + warm measurement (two requests total):

```bash
python openai_chat_completion_client.py --num_following 1
```

Example console output

```
=== Run 1: baseline TTFT ===
TTFT_1 = 0.429s
(no KV‑cache flush requested)

=== Run 2: TTFT continued ===
TTFT_2 = 0.081s
```

`benchmark.jsonl`

```json
{"run_index":1,"context_tokens":120938,"ttft_seconds":0.429}
{"run_index":2,"context_tokens":120938,"ttft_seconds":0.081}
```

---

## 4 · Advanced use

### 4.1 · Benchmark after cache eviction

```bash
python openai_chat_completion_client.py \
  -C war_and_peace.txt        \
  --num_following 3           \
  --flush_cache               \
  --prompt "Give me a concise outline." \
  --out warpeace_flush.jsonl
```

* Run 1 – cold  
* Cache flushed  
* Run 2 – cold again (miss)  
* Runs 3‑4 – warm (hits)

### 4.2 · Stress maximum context

```bash
python openai_chat_completion_client.py \
  --max_ctx_tokens 131072 \
  --num_following 1 -F
```

Generates a k‑char filler, truncates to fit
`≤ max_ctx ` tokens (keeps a **2 048‑token safety margin**), then
measures cold vs. warm TTFT.

---

## 5 · Output schema

Each JSONL line contains:

| Key | Type | Description |
|-----|------|-------------|
| `run_index`      | int   | 1 = baseline, 2… = follow‑ups |
| `context_tokens` | int   | Tokens after truncation |
| `ttft_seconds`   | float | Wall‑clock seconds to **first** streamed token |

Concatenate multiple logs with `cat` and plot as you like.

---

## 6 · Implementation notes

* **Safety margin** – `SAFETY_MARGIN = 2048` tokens so the request never
  overruns model context even on tokenizer quirks.
* **Spinner** – Red arrows animate while waiting for token #1, stop instantly
  on arrival for visual TTFT confirmation.
* **Tokenizer fallback** – If the matching tokenizer can’t load, the script
  degrades to the heuristic “≈ 4 chars = 1 token”.
* **Cache‑flush routine** – Sends ten *1‑token* completions built on a
  100 k‑char filler doc to evict KV blocks from VRAM.

## 7 · Batch driver script (`bench_ttft_sweep.sh`)

This is an example basic bash script you might use to do a sweep across different context lengths, combining results to one file for easy comparison of caching methods.



### What the script does

| Step | Detail |
|------|--------|
| **1.  Configure variables** | `BENCH` points to the Python benchmark, `MASTER_OUT` is the cumulative log, and `CONTEXT_SIZES` lists the target document lengths (in **tokens**). |
| **2.  Per‑size run** | For each length the script launches the benchmark with:<br>• custom `--max_ctx_tokens` (see above)<br>• one cache‑hit follow‑up (`--num_following 1`)<br>• an explicit **70 B** Llama 3 checkpoint via `--model` |
| **3.  Log collation** | Each invocation writes its own JSONL (`ttft_<N>.jsonl`). Those lines are immediately concatenated into **`all_ttft_results.jsonl`**, producing a tidy file like: <br>`{"run_index":1,"context_tokens":32000,"ttft_seconds":0.45}` |
| **4.  Done banner** | After the loop finishes you get a green check‑mark and the path to the merged log. |

#### Customising

* **Change the model** — edit `--model …` to point at any endpoint‑visible name.  
* **Different sizes** — just tweak the `CONTEXT_SIZES` array.  
* **More follow‑ups** — bump `--num_following` if you want deeper cache‑hit sampling.  

---

## 8 · vLLM模型启动指令

### 8.1 · Mistral-7B-Instruct-v0.1 (单卡)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 131072 \
  --lmcache \
  --lmcache-backend redis \
  --lmcache-redis-host localhost \
  --lmcache-redis-port 6379
```

### 8.2 · Yi-34B-200K (双卡)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model 01-ai/Yi-34B-200K \
  --port 8001 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 200000 \
  --lmcache \
  --lmcache-backend redis \
  --lmcache-redis-host localhost \
  --lmcache-redis-port 6379
```

### 8.3 · 其他大模型 (多卡)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <model_name> \
  --port <port> \
  --tensor-parallel-size <num_gpus> \
  --gpu-memory-utilization 0.8 \
  --max-model-len <context_length> \
  --lmcache \
  --lmcache-backend redis \
  --lmcache-redis-host localhost \
  --lmcache-redis-port 6379
```

---

## 9 · openai_chat_completion_client.py 运行指令

### 9.1 · 基础测试 (Mistral-7B)

```bash
python openai_chat_completion_client.py \
  --api_base http://localhost:8000/v1 \
  --model "mistralai/Mistral-7B-Instruct-v0.1" \
  --max_ctx_tokens 1000 \
  --context_file "" \
  --prompt "这个文档主要讲了什么？" \
  --num_following 1
```

### 9.2 · 基础测试 (Yi-34B)

```bash
python openai_chat_completion_client.py \
  --api_base http://localhost:8001/v1 \
  --model "01-ai/Yi-34B-200K" \
  --max_ctx_tokens 1000 \
  --context_file "" \
  --prompt "这个文档主要讲了什么？" \
  --num_following 1
```

### 9.3 · 缓存刷新测试

```bash
python openai_chat_completion_client.py \
  --api_base http://localhost:8000/v1 \
  --model "mistralai/Mistral-7B-Instruct-v0.1" \
  --max_ctx_tokens 1000 \
  --context_file "" \
  --prompt "请总结这个文档的主要内容" \
  --num_following 2 \
  --flush_cache
```

### 9.4 · 大上下文测试

```bash
python openai_chat_completion_client.py \
  --api_base http://localhost:8000/v1 \
  --model "mistralai/Mistral-7B-Instruct-v0.1" \
  --max_ctx_tokens 32000 \
  --context_file "" \
  --prompt "分析这个文档的结构和特点" \
  --num_following 1
```

### 9.5 · 自定义文档测试

```bash
python openai_chat_completion_client.py \
  --api_base http://localhost:8000/v1 \
  --model "mistralai/Mistral-7B-Instruct-v0.1" \
  --max_ctx_tokens 2000 \
  --context_file "your_document.txt" \
  --prompt "提取文档中的关键信息" \
  --num_following 1 \
  --out custom_test.jsonl
```

### 9.6 · 推荐提示词

| 提示词类型 | 示例 | 缓存效果 |
|------------|------|----------|
| **简单问答** | `"这个文档主要讲了什么？"` | 优秀 (1.67x) |
| **中文总结** | `"请用中文总结这个文档的主要内容"` | 良好 (1.06x) |
| **英文问答** | `"What is the main topic of this document?"` | 良好 (1.06x) |
| **功能询问** | `"FFmpeg的主要功能有哪些？"` | 一般 (1.04x) |
| **结构分析** | `"分析这个文档的结构和特点"` | 一般 (1.03x) |

### 9.7 · 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--api_base` | API服务器地址 | `http://localhost:8000/v1` (Mistral) / `http://localhost:8001/v1` (Yi-34B) |
| `--model` | 模型名称 | `"mistralai/Mistral-7B-Instruct-v0.1"` / `"01-ai/Yi-34B-200K"` |
| `--max_ctx_tokens` | 最大上下文长度 | `1000` (测试) / `32000` (大文档) |
| `--context_file` | 文档文件 | `""` (使用ffmpeg.txt) / `"your_file.txt"` |
| `--prompt` | 用户提示词 | 见上表推荐提示词 |
| `--num_following` | 后续测试次数 | `1` (基础) / `2-3` (详细测试) |
| `--flush_cache` | 是否刷新缓存 | 添加此参数测试缓存刷新效果 |

