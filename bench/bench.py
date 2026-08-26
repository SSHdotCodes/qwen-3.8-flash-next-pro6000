#!/usr/bin/env python3
"""Speed + quality harness for the Qwen3.8-Flash-Next deployment.

Quality gate: every profile runs greedy (temperature 0, top_k 1). A candidate
config is "no quality loss" only if its greedy continuation matches the pinned
baseline token-for-token.
"""
import argparse, json, os, statistics, sys, time
import urllib.request

BASE = "http://127.0.0.1:{port}"

TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Executes a bash command and returns its output.",
      "parameters": {"type": "object", "properties": {
          "command": {"type": "string", "description": "The command to execute"},
          "description": {"type": "string", "description": "Clear, concise description of what this command does"},
          "timeout": {"type": "number", "description": "Optional timeout in milliseconds"}},
       "required": ["command", "description"]}}},
    {"type": "function", "function": {"name": "read", "description": "Reads a file from the local filesystem.",
      "parameters": {"type": "object", "properties": {
          "filePath": {"type": "string"}, "offset": {"type": "number"}, "limit": {"type": "number"}},
       "required": ["filePath"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Performs exact string replacement in a file.",
      "parameters": {"type": "object", "properties": {
          "filePath": {"type": "string"}, "oldString": {"type": "string"},
          "newString": {"type": "string"}, "replaceAll": {"type": "boolean"}},
       "required": ["filePath", "oldString", "newString"]}}},
    {"type": "function", "function": {"name": "write", "description": "Writes a file to the local filesystem.",
      "parameters": {"type": "object", "properties": {
          "filePath": {"type": "string"}, "content": {"type": "string"}},
       "required": ["filePath", "content"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Fast content search using ripgrep.",
      "parameters": {"type": "object", "properties": {
          "pattern": {"type": "string"}, "path": {"type": "string"},
          "include": {"type": "string"}},
       "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "glob", "description": "Fast file pattern matching.",
      "parameters": {"type": "object", "properties": {
          "pattern": {"type": "string"}, "path": {"type": "string"}},
       "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "list", "description": "Lists files and directories.",
      "parameters": {"type": "object", "properties": {
          "path": {"type": "string"}, "ignore": {"type": "array", "items": {"type": "string"}}},
       "required": ["path"]}}},
    {"type": "function", "function": {"name": "todowrite", "description": "Update the structured task list.",
      "parameters": {"type": "object", "properties": {"todos": {"type": "array", "items": {"type": "object",
          "properties": {"id": {"type": "string"}, "content": {"type": "string"},
                         "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}},
          "required": ["id", "content", "status"]}}}, "required": ["todos"]}}},
    {"type": "function", "function": {"name": "webfetch", "description": "Fetch a URL and convert it to markdown.",
      "parameters": {"type": "object", "properties": {
          "url": {"type": "string"}, "format": {"type": "string", "enum": ["text", "markdown", "html"]}},
       "required": ["url", "format"]}}},
]

AGENT_SYSTEM = (
    "You are opencode, an autonomous coding agent running in the user's terminal.\n\n"
    "You are an interactive CLI tool that helps users with software engineering tasks. "
    "Use the instructions below and the tools available to you to assist the user.\n\n"
    "# Tone and style\n"
    "You should be concise, direct, and to the point. Answer the user's question directly, "
    "without elaboration, explanation, or details, unless the user asks for them. "
    "One word answers are best. Avoid introductions, conclusions, and explanations.\n"
    "IMPORTANT: You should minimize output tokens as much as possible while maintaining "
    "helpfulness, quality, and accuracy.\n"
    "IMPORTANT: Keep your responses short. You MUST answer concisely with fewer than 4 lines, "
    "unless the user asks for detail.\n\n"
    "# Proactiveness\n"
    "You are allowed to be proactive, but only when the user asks you to do something. "
    "Strike a balance between doing the right thing when asked, including taking actions and "
    "follow-up actions, and not surprising the user with actions you take without asking.\n\n"
    "# Following conventions\n"
    "When making changes to files, first understand the file's code conventions. Mimic code style, "
    "use existing libraries and utilities, and follow existing patterns.\n"
    "- NEVER assume that a given library is available, even if it is well known.\n"
    "- When you create a new component, first look at existing components.\n"
    "- When you edit a piece of code, first look at the code's surrounding context.\n"
    "- Always follow security best practices. Never introduce code that exposes or logs secrets.\n\n"
    "# Code style\n"
    "IMPORTANT: DO NOT ADD ANY COMMENTS unless asked.\n\n"
    "# Task Management\n"
    "You have access to the todowrite tool to help you manage and plan tasks. Use these tools "
    "VERY frequently to ensure that you are tracking your tasks and giving the user visibility.\n\n"
    "# Doing tasks\n"
    "The user will primarily request you perform software engineering tasks. For these tasks the "
    "following steps are recommended:\n"
    "- Use the todowrite tool to plan the task if required\n"
    "- Use the available search tools to understand the codebase and the user's query\n"
    "- Implement the solution using all tools available to you\n"
    "- Verify the solution if possible with tests\n"
    "- VERY IMPORTANT: When you have completed a task, you MUST run the lint and typecheck commands\n\n"
    "# Tool usage policy\n"
    "- When doing file search, prefer to use the Task tool in order to reduce context usage.\n"
    "- IMPORTANT: All tools are executed in parallel when multiple tool calls are sent in a single "
    "message. Send multiple tool calls in a single message to run them in parallel.\n\n"
    "Here is useful information about the environment you are running in:\n"
    "<env>\nWorking directory: /home/user/projects/inventory-svc\nIs directory a git repo: yes\n"
    "Platform: linux\nToday's date: 2026-08-26\n</env>\n"
)

FILE_CONTEXT = """<file path="src/queue/scheduler.rs">
use std::collections::BinaryHeap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::metrics::Counter;
use crate::task::{Task, TaskId, TaskState};

pub struct Scheduler {
    heap: Mutex<BinaryHeap<Entry>>,
    inflight: Mutex<Vec<TaskId>>,
    capacity: usize,
    lease_ttl: Duration,
    enqueued: Counter,
    dropped: Counter,
}

#[derive(PartialEq, Eq)]
struct Entry {
    deadline: Instant,
    task: Task,
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.deadline.cmp(&self.deadline)
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Scheduler {
    pub fn new(capacity: usize, lease_ttl: Duration) -> Arc<Self> {
        Arc::new(Self {
            heap: Mutex::new(BinaryHeap::new()),
            inflight: Mutex::new(Vec::new()),
            capacity,
            lease_ttl,
            enqueued: Counter::new("scheduler_enqueued"),
            dropped: Counter::new("scheduler_dropped"),
        })
    }

    pub fn enqueue(&self, task: Task) -> bool {
        let mut heap = self.heap.lock().unwrap();
        if heap.len() >= self.capacity {
            self.dropped.inc();
            return false;
        }
        let deadline = Instant::now() + self.lease_ttl;
        heap.push(Entry { deadline, task });
        self.enqueued.inc();
        true
    }

    pub fn claim(&self) -> Option<Task> {
        let mut heap = self.heap.lock().unwrap();
        let entry = heap.pop()?;
        let mut inflight = self.inflight.lock().unwrap();
        inflight.push(entry.task.id());
        Some(entry.task)
    }

    pub fn complete(&self, id: TaskId, state: TaskState) {
        let mut inflight = self.inflight.lock().unwrap();
        inflight.retain(|candidate| *candidate != id);
        if state == TaskState::Failed {
            drop(inflight);
        }
    }

    pub fn reap_expired(&self) -> Vec<TaskId> {
        let now = Instant::now();
        let mut heap = self.heap.lock().unwrap();
        let mut expired = Vec::new();
        let mut kept = BinaryHeap::new();
        while let Some(entry) = heap.pop() {
            if entry.deadline <= now {
                expired.push(entry.task.id());
            } else {
                kept.push(entry);
            }
        }
        *heap = kept;
        expired
    }
}
</file>
"""

PROFILES = {
    # Pure decode speed: long, low-entropy-free generation.
    "speed": {"max_tokens": 1024, "thinking": False, "system": None,
              "prompt": ("Write a detailed technical explanation of how a modern MoE transformer "
                         "serves a single decode step: routing, expert gather, attention, KV cache "
                         "reads, and where the memory bandwidth goes. Be thorough and specific.")},
    # Realistic coding output.
    "coding": {"max_tokens": 1024, "thinking": False, "system": None,
               "prompt": ("Write a complete production-quality TypeScript implementation of a bounded "
                          "asynchronous task queue with backpressure, cancellation, graceful shutdown, "
                          "typed errors, and deterministic tests. Output code only.")},
    # Reasoning-heavy decode (what OpenCode xhigh actually does).
    "thinking": {"max_tokens": 1536, "thinking": True, "system": None,
                 "prompt": ("Design a fault-tolerant distributed job scheduler. Analyze competing "
                            "consistency models, failure modes, leases, idempotency, fairness, and "
                            "recovery, then recommend a design with explicit tradeoffs.")},
    # OpenCode-shaped: full system prompt + tool schemas + file context.
    "agent": {"max_tokens": 1024, "thinking": True, "system": AGENT_SYSTEM, "tools": True,
              "prompt": (FILE_CONTEXT + "\nReview src/queue/scheduler.rs for correctness bugs. "
                         "There are at least two real defects. Explain each one and how to fix it.")},
    # Agent shape, short answer -> exercises tool-call emission.
    "agent_tool": {"max_tokens": 384, "thinking": False, "system": AGENT_SYSTEM, "tools": True,
                   "prompt": "Find every call site of `reap_expired` in this repo, then read the scheduler test file."},
}


def build_payload(profile_name, model, max_tokens, seed, temperature=0.0, nonce=0):
    p = PROFILES[profile_name]
    messages = []
    if p.get("system"):
        messages.append({"role": "system", "content": p["system"]})
    # A deterministic per-run nonce defeats radix prefix reuse without calling
    # /flush_cache. Two flush_cache calls around a generation corrupt the mamba
    # extra-buffer state and make the model emit \"!\" forever.
    prefix = "" if nonce == 0 else "[request %d]\n" % nonce
    messages.append({"role": "user", "content": prefix + p["prompt"]})
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens or p["max_tokens"],
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": p["thinking"], "preserve_thinking": p["thinking"]},
    }
    if temperature == 0.0:
        payload["top_k"] = 1
        payload["top_p"] = 1.0
    if p.get("tools"):
        payload["tools"] = TOOLS
        payload["tool_choice"] = "auto"
    return payload


def post_stream(url, payload, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = None
    usage = {}
    content, reasoning, toolcalls = [], [], []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            ch = chunk.get("choices") or []
            if not ch:
                continue
            d = ch[0].get("delta") or {}
            c = d.get("content") or ""
            r = d.get("reasoning_content") or d.get("reasoning") or ""
            tc = d.get("tool_calls") or []
            if (c or r or tc) and first is None:
                first = time.perf_counter()
            if c:
                content.append(c)
            if r:
                reasoning.append(r)
            if tc:
                toolcalls.append(tc)
    done = time.perf_counter()
    n = int(usage.get("completion_tokens") or 0)
    ttft = (first - started) if first else (done - started)
    dec = max(done - (first or started), 1e-9)
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": n,
        "elapsed_s": round(done - started, 4),
        "ttft_s": round(ttft, 4),
        "decode_tok_s": round(max(n - 1, 0) / dec, 2),
        "e2e_tok_s": round(n / max(done - started, 1e-9), 2),
        "reasoning": "".join(reasoning),
        "content": "".join(content),
        "tool_calls": toolcalls,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30010)
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--profiles", default="speed,coding,thinking,agent,agent_tool")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
    args = ap.parse_args()

    base = BASE.format(port=args.port)
    url = base + "/v1/chat/completions"
    out = {"tag": args.tag, "profiles": {}}

    for name in args.profiles.split(","):
        name = name.strip()
        if not name:
            continue
        payload = build_payload(name, args.model, args.max_tokens, args.seed, nonce=999)
        wp = dict(payload); wp["max_tokens"] = 32
        try:
            post_stream(url, wp)
        except Exception as e:
            print(f"warmup failed for {name}: {e}", file=sys.stderr)
        runs = []
        for i in range(args.runs):
            r = post_stream(url, build_payload(name, args.model, args.max_tokens, args.seed, nonce=i + 1))
            runs.append(r)
            print(json.dumps({"tag": args.tag, "profile": name, "run": i + 1,
                              **{k: v for k, v in r.items() if k not in ("reasoning", "content", "tool_calls")}}),
                  flush=True)
        out["profiles"][name] = {
            "median_decode_tok_s": round(statistics.median(r["decode_tok_s"] for r in runs), 2),
            "median_e2e_tok_s": round(statistics.median(r["e2e_tok_s"] for r in runs), 2),
            "median_ttft_s": round(statistics.median(r["ttft_s"] for r in runs), 4),
            "prompt_tokens": runs[0]["prompt_tokens"],
            "completion_tokens": [r["completion_tokens"] for r in runs],
            "reasoning": runs[0]["reasoning"],
            "content": runs[0]["content"],
            "tool_calls": runs[0]["tool_calls"],
        }

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{args.tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    summary = {n: {"decode_tok_s": v["median_decode_tok_s"], "ttft_s": v["median_ttft_s"],
                   "prompt_tokens": v["prompt_tokens"]}
               for n, v in out["profiles"].items()}
    print("SUMMARY " + json.dumps({"tag": args.tag, **summary}), flush=True)
    print("wrote " + path, flush=True)


if __name__ == "__main__":
    main()
