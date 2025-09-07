#!/usr/bin/env python3
"""
Interactive TTFT benchmark using requests library (no OpenAI dependency).
Direct communication with vLLM or other API servers.
"""

# Standard
from io import StringIO
from pathlib import Path
from typing import List
import argparse
import json
import random
import string
import sys
import threading
import time

# Third Party
import requests
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# ----------------------------------------------------------------------
SAFETY_MARGIN = 2048  # tokens kept free below model ctx limit
FILLER_LEN_CHARS = 100_000  # ≈ length of each cache‑filler prompt
NUM_FILLER_PROMPTS = 10  # how many fillers to send for eviction
DEFAULT_FFMPEG = "ffmpeg.txt"
# ----------------------------------------------------------------------


# ---------------- helper utilities ------------------------------------
def rand_ascii(n: int) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    tok: PreTrainedTokenizerBase,
) -> str:
    ids = tok.encode(
        text, add_special_tokens=False, truncation=True, max_length=max_tokens
    )
    return tok.decode(ids, skip_special_tokens=True)


def log_jsonl(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        json.dump(rec, fh)
        fh.write("\n")


# ---------------- tiny CLI spinner ------------------------------------
class Printer:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _spin(self) -> None:
        idx = 0
        while not self._stop_event.is_set():
            print(f"\033[31m\r{'>' * (idx % 6):<6}\033[0m", end="", flush=True)
            idx += 1
            time.sleep(0.2)

    def start(self) -> None:
        if self._thread is None:
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None
        print("\033[31m\r>>>>> \033[0m", end="", flush=True)


# ---------------- benchmark helpers -----------------------------------
def build_chat(system_doc: str, user_prompt: str) -> List[dict]:
    return [
        {"role": "user", "content": f"I've got a document:\n```\n{system_doc}\n```"},
        {"role": "assistant", "content": "I've got your document."},
        {"role": "user", "content": user_prompt},
    ]


def get_models_list(base_url: str) -> List[str]:
    """Get available models from the server."""
    try:
        response = requests.get(f"{base_url}/models")
        response.raise_for_status()
        models_data = response.json()
        return [model['id'] for model in models_data.get('data', [])]
    except Exception as e:
        print(f"Warning: Could not fetch models list: {e}")
        return ["facebook/opt-125m"]  # fallback


def ttft_stream_requests(
    base_url: str,
    model: str,
    messages: list[dict],
    printer: Printer | None = None,
) -> tuple[float, str]:
    start = time.perf_counter()
    
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "stream": True,
                "max_tokens": 1024
            },
            stream=True,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        
        first_tok_t = None
        buf = StringIO()
        if printer:
            printer.start()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # 去掉 'data: ' 前缀
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk['choices'][0]['delta']
                        if 'content' in delta and delta['content']:
                            if first_tok_t is None:
                                first_tok_t = time.perf_counter()
                                if printer:
                                    printer.stop()
                            print(delta['content'], end="", flush=True)
                            buf.write(delta['content'])
                    except json.JSONDecodeError:
                        continue
        
        print()  # newline after streaming
        if first_tok_t is None:
            raise RuntimeError("no tokens returned")
        return first_tok_t - start, buf.getvalue()
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        raise
    finally:
        if printer:
            printer.stop()


def flush_kv_cache(base_url: str, model: str) -> None:
    """Flush KV cache by sending filler prompts."""
    filler_chat = build_chat(rand_ascii(FILLER_LEN_CHARS), "noop")
    for i in range(NUM_FILLER_PROMPTS):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": filler_chat,
                    "temperature": 0.0,
                    "max_tokens": 1,
                    "stream": False
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            print(f"Cache flush prompt {i+1}/{NUM_FILLER_PROMPTS} sent")
        except Exception as e:
            print(f"Warning: Cache flush prompt {i+1} failed: {e}")


# ---------------- command‑line parsing --------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="ttft_benchmark_requests",
        description="Interactive TTFT benchmark using requests library; \
        flush cache only with -F/--flush_cache.",
    )
    ap.add_argument("--api_base", default="http://localhost:8000/v1",
                   help="API base URL (default: http://localhost:8000/v1)")
    ap.add_argument("--model", help="Model name/ID; default = first entry from /models.")
    ap.add_argument(
        "-C",
        "--context_file",
        nargs="?",
        const="",
        default=None,
        help="FILE → use document, flag‑only → ffmpeg.txt, "
        "omit flag → synthetic filler",
    )
    ap.add_argument(
        "--max_ctx_tokens",
        type=int,
        default=131_072,
        help="Max tokens kept from the document after truncation.",
    )
    ap.add_argument(
        "--prompt",
        default="Summarize this text",
        help="User prompt appended after the document.",
    )
    ap.add_argument(
        "--num_following",
        type=int,
        default=1,
        help="Extra measured requests after run 1 to test cache retrieval.",
    )
    ap.add_argument(
        "--flush_cache",
        "-F",
        action="store_true",
        help="Evict GPU KV‑cache between run 1 and follow‑ups.",
    )
    ap.add_argument(
        "--out",
        default="benchmark_requests.jsonl",
        help="JSONL file for results (overwritten each run).",
    )
    return ap.parse_args()


# ---------------- main routine ----------------------------------------
def main() -> None:
    args = parse_args()

    # pick model (fallback = first listed on the server)
    if args.model:
        model_id = args.model
    else:
        models = get_models_list(args.api_base)
        model_id = models[0] if models else "facebook/opt-125m"
    
    print(f"Using model: {model_id}")
    print(f"API base: {args.api_base}")

    # ---------- choose / build the document ---------------------------
    if args.context_file is None:
        # flag omitted → synthetic filler
        raw_doc = rand_ascii(args.max_ctx_tokens * 4)  # ≈4 chars/token
    elif args.context_file == "":
        # flag present w/o file → bundled ffmpeg.txt
        raw_doc = Path(DEFAULT_FFMPEG).read_text(encoding="utf-8")
    else:
        raw_doc = Path(args.context_file).read_text(encoding="utf-8")

    # ---------- truncate ------------------------------------------------
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model_ctx = (
            tok.model_max_length if tok.model_max_length > 0 else args.max_ctx_tokens
        )
        doc = truncate_to_tokens(
            raw_doc, min(model_ctx - SAFETY_MARGIN, args.max_ctx_tokens), tok
        )
    except Exception:
        char_limit = (args.max_ctx_tokens - SAFETY_MARGIN) * 4  # ≈4 chars/token
        doc = raw_doc[:char_limit]

    out_path = Path(args.out)
    out_path.write_text("", encoding="utf-8")  # clear file
    printer = Printer()

    # ---------------- RUN 1 ----------------
    print("\n=== Run 1: baseline TTFT ===")
    base_chat = build_chat(doc, args.prompt)
    ttft1, gen1 = ttft_stream_requests(args.api_base, model_id, base_chat, printer)
    print(f"\033[33mTTFT_1 = {ttft1:.3f}s\033[0m")
    
    # Log results
    try:
        context_tokens = len(tok.encode(doc, add_special_tokens=False))
    except:
        context_tokens = len(doc) // 4  # rough estimate
    
    log_jsonl(
        out_path,
        {
            "run_index": 1,
            "context_tokens": context_tokens,
            "ttft_seconds": ttft1,
        },
    )

    # -------------- optional follow‑ups --------------
    if args.num_following > 0:
        if args.flush_cache:
            print(f"\nFlushing KV‑cache with {NUM_FILLER_PROMPTS} prompts …")
            flush_kv_cache(args.api_base, model_id)
        else:
            print("\n(no KV‑cache flush requested)")

        for run in range(2, 2 + args.num_following):
            label = "post‑flush" if args.flush_cache else "continued"
            print(f"\n=== Run {run}: TTFT {label} ===")
            ttft, gen = ttft_stream_requests(args.api_base, model_id, base_chat, printer)
            print(f"\033[33mTTFT_{run} = {ttft:.3f}s\033[0m")
            log_jsonl(
                out_path,
                {
                    "run_index": run,
                    "context_tokens": context_tokens,
                    "ttft_seconds": ttft,
                },
            )
            time.sleep(5)  # brief idle gap


if __name__ == "__main__":
    main()