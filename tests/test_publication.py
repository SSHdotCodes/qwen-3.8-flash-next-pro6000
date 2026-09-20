import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import sustained
from summarize_sustained import summarize


class LauncherTests(unittest.TestCase):
    def command(self, **overrides):
        env = {
            **os.environ,
            "DRY_RUN": "1",
            "PORT": "30010",
            "NAME": "qwen38-flash-next-sglang",
            "HF_CACHE": "/tmp/model cache",
            "RUNTIME_CACHE": "/tmp/runtime cache",
            **overrides,
        }
        return subprocess.run(
            ["bash", str(ROOT / "serve/run-server.sh")],
            capture_output=True,
            text=True,
            env=env,
        )

    def test_protected_configuration_and_loopback(self):
        result = self.command()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = shlex.split(result.stdout)
        expected = {
            "-p": "127.0.0.1:30010:30010",
            "--context-length": "262144",
            "--max-total-tokens": "524288",
            "--kv-cache-dtype": "fp8_e4m3",
            "--quantization": "modelopt_fp4",
            "--max-running-requests": "2",
            "--speculative-num-steps": "3",
            "--speculative-num-draft-tokens": "4",
            "--mamba-ssm-dtype": "bfloat16",
            "--max-mamba-cache-size": "14",
            "--cuda-graph-max-bs-decode": "2",
        }
        for flag, value in expected.items():
            self.assertEqual(args[args.index(flag) + 1], value, flag)
        self.assertIn("--speculative-use-rejection-sampling", args)
        self.assertIn("--enable-response-store", args)
        self.assertIn("QWEN_MTP_HOTMAP=/opt/qwen-mtp-hotmap.json", args)
        self.assertNotIn("--network", args)
        self.assertNotIn("--disable-flashinfer-autotune", args)
        self.assertIn("/tmp/model cache:/root/.cache/huggingface", args)
        self.assertEqual(
            args[args.index("--model-path") + 1],
            args[args.index("--speculative-draft-model-path") + 1],
        )

    def test_port_override_and_rejection(self):
        args = shlex.split(self.command(PORT="30123", NAME="test-flash").stdout)
        self.assertEqual(args[args.index("-p") + 1], "127.0.0.1:30123:30123")
        self.assertEqual(args[args.index("--port") + 1], "30123")
        for value in ("0", "65536", "-1", "30010;echo BAD", "00008", "9" * 30):
            self.assertNotEqual(self.command(PORT=value).returncode, 0, value)


class EvidenceTests(unittest.TestCase):
    def test_export_checksums(self):
        subprocess.run(
            [sys.executable, str(ROOT / "serve/verify-runtime.py")], check=True
        )

    def test_aggregation_from_raw_results(self):
        directory = ROOT / "results/20260919"
        actual = summarize(directory)
        expected = json.loads((directory / "final-comparison.json").read_text())
        self.assertAlmostEqual(
            actual["geomean_gain_percent"], expected["geomean_gain_percent"]
        )
        self.assertEqual(
            actual["bootstrap_interval_percent"], expected["bootstrap_interval_percent"]
        )
        self.assertEqual(actual["cells"], expected["cells"])


class StreamingTests(unittest.TestCase):
    def test_recommended_sampler(self):
        p = sustained.payload("model", "prompt", 8192, "xhigh", 19)
        for key, value in {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "reasoning_effort": "xhigh",
            "max_tokens": 8192,
        }.items():
            self.assertEqual(p[key], value)

    def test_reasoning_usage_and_finish_reason(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def raise_for_status(self):
                pass

            def iter_lines(self):
                yield b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}'
                yield b'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}'
                yield b'data: {"usage":{"prompt_tokens":11,"completion_tokens":101},"choices":[]}'
                yield b"data: [DONE]"

        with (
            patch.object(sustained.requests, "post", return_value=Response()),
            patch.object(sustained.time, "perf_counter", side_effect=[1.0, 2.0, 4.0]),
        ):
            row = sustained.request("http://localhost", {})
        self.assertEqual(row["decode_tok_s"], 50.0)
        self.assertEqual(row["ttft_s"], 1.0)
        self.assertEqual(row["elapsed_s"], 3.0)
        self.assertEqual(row["reasoning"], "thinking")
        self.assertEqual(row["content"], "answer")
        self.assertEqual(row["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
