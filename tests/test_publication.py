import importlib.util
import copy
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import step_median
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
            "MODEL_VIEWS": "/tmp/views",
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
            "--max-total-tokens": "786432",
            "--kv-cache-dtype": "fp8_e4m3",
            "--quantization": "compressed-tensors",
            "--model-path": "/opt/redhat-model/target",
            "--speculative-draft-model-path": "/opt/redhat-model/draft",
            "--speculative-draft-model-quantization": "fp8",
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
        for env in (
            "QWEN_MTP_HOTMAP=/opt/qwen-mtp-hotmap.json",
            "QWEN_NVFP4_MARLIN=1",
            "QWENFAST_PK=1",
            "QWENFAST_W8=inproj,qkv,oproj,head,shared,outproj,hc",
            "QWEN_DRAFT_TRUNC=1",
            "QWENFAST_A16MOE=1",
            "QWENFAST_W8M=1",
            "QWENFAST_W8M_PDL=1",
            "QWENFAST_HCCOMB=1",
            "QWEN_MTP_A16=/opt/qwen-draft-a16.pt",
            "QWENFAST_HC_WP=1",
            "QWENFAST_ATTN=1",
            "QWENFAST_HEAD_SCREEN=1",
            "QWEN_MTP_HOT_SCREEN2=1",
            "QWENFAST_GDNOUT_WP=1",
            "QWENFAST_GDNOUT_VARIANT=replay",
            "QWENFAST_GDN_REPLAY=1",
        ):
            self.assertIn(env, args)
        self.assertNotIn("--network", args)
        self.assertNotIn("--disable-flashinfer-autotune", args)
        self.assertIn("/tmp/model cache:/root/.cache/huggingface", args)
        self.assertIn("/tmp/model cache:/tmp/model cache:ro", args)
        self.assertIn(
            "/tmp/views/c8f2fb1b9869f686b214782036123b10ff96d14a:/opt/redhat-model:ro", args
        )
        self.assertIn(
            "/tmp/views/c8f2fb1b9869f686b214782036123b10ff96d14a/draft-a16.pt:/opt/qwen-draft-a16.pt:ro", args
        )

    def test_installed_kern2_environment_and_flags(self):
        args = shlex.split(self.command().stdout)
        manifest = json.loads((ROOT / "serve/runtime.json").read_text())
        environment = dict(args[i + 1].split("=", 1) for i, value in enumerate(args) if value == "-e")
        self.assertEqual(environment, manifest["environment"])

        def flags(argv):
            result = {}
            i = 0
            while i < len(argv):
                key = argv[i]
                self.assertTrue(key.startswith("--"), key)
                self.assertNotIn(key, result)
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    result[key] = argv[i + 1]
                    i += 2
                else:
                    result[key] = True
                    i += 1
            return result

        self.assertEqual(
            flags(args[args.index("sglang.launch_server") + 1:]),
            flags(manifest["server_args"]),
        )

    def test_port_override_and_rejection(self):
        args = shlex.split(self.command(PORT="30123", NAME="test-flash").stdout)
        self.assertEqual(args[args.index("-p") + 1], "127.0.0.1:30123:30123")
        self.assertEqual(args[args.index("--port") + 1], "30123")
        for value in ("0", "65536", "-1", "30010;echo BAD", "00008", "9" * 30):
            self.assertNotEqual(self.command(PORT=value).returncode, 0, value)


class EvidenceTests(unittest.TestCase):
    def test_intentional_public_differences_preserve_installed_hashes(self):
        spec = importlib.util.spec_from_file_location("verify_runtime", ROOT / "serve/verify-runtime.py")
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        manifest = json.loads((ROOT / "serve/runtime.json").read_text())
        differences = verifier.public_differences(manifest)
        expected = {
            "sglang/srt/model_loader/loader.py": "ce8a8cf3c6c2b52ea5d574fcdd9312086c680278146f12429323ad9f8af76b31",
            "sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py":
                "d3a9930c7d97d7c56571155b9d797c3803aafa406ae28b4d79900d5685b3b901",
        }
        self.assertEqual({e["path"]: e["installed_sha256"] for e in differences}, expected)
        measured = {"files": {e["path"]: {
            "mount": e["container_path"], "sha256": e.get("installed_sha256", e["sha256"]),
        } for e in manifest["files"]}}
        verifier.compare_installed_files(manifest, measured)
        # An intentional rename must not allow arbitrary source changes to be ignored.
        altered = copy.deepcopy(measured)
        altered["files"]["sglang/srt/model_loader/loader.py"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Installed reference checksum mismatch"):
            verifier.compare_installed_files(manifest, altered)
        undocumented = copy.deepcopy(manifest)
        del next(e for e in undocumented["files"] if "installed_sha256" in e)["public_difference"]
        with self.assertRaisesRegex(ValueError, "Incomplete public/installed difference"):
            verifier.public_differences(undocumented)

    def test_prepared_views_enable_the_public_loader(self):
        spec = importlib.util.spec_from_file_location("prepare_model", ROOT / "serve/prepare-model.py")
        prepare_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare_model)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "snapshot"
            source.mkdir()
            config = {
                "text_config": {"layer_types": ["qwen_sparse_attention", "linear_attention"]},
                "quantization_config": {"ignore": []},
            }
            index = {"metadata": {"total_size": 100}, "weight_map": {
                "model.layers.0.weight": "model.safetensors", "mtp.layers.0.weight": "model.safetensors",
            }}
            (source / "config.json").write_text(json.dumps(config))
            (source / "model.safetensors.index.json").write_text(json.dumps(index))
            (source / "model.safetensors").write_bytes(b"synthetic fixture")
            for kind in ("target", "draft"):
                dest = Path(directory) / kind
                prepare_model.write_view(source, dest, kind)
                view = json.loads((dest / "config.json").read_text())
                self.assertEqual(view["text_config"]["layer_types"], ["full_attention", "linear_attention"])
                self.assertTrue((dest / "model.safetensors").is_symlink())
                if kind == "target":
                    self.assertTrue(view["redhat_streamed_load"])
                    self.assertEqual(view["text_config"]["ple_embedding_dtype"], "bfloat16")
                    self.assertIn('"redhat_streamed_load", False',
                                  (ROOT / "serve/overlays/sglang/srt/model_loader/loader.py").read_text())
                else:
                    self.assertNotIn("quantization_config", view)
                    draft_index = json.loads((dest / "model.safetensors.index.json").read_text())
                    self.assertEqual(set(draft_index["weight_map"]), {"mtp.layers.0.weight"})
            self.assertEqual(json.loads((source / "config.json").read_text()), config)

    def test_export_checksums(self):
        subprocess.run(
            [sys.executable, str(ROOT / "serve/verify-runtime.py")], check=True
        )

    def test_kernel_build_list_is_pinned(self):
        spec = importlib.util.spec_from_file_location(
            "verify_runtime", ROOT / "serve/verify-runtime.py"
        )
        verify_runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verify_runtime)
        sources = verify_runtime.build_sources(ROOT / "serve/qwenfast/build_qf.py")
        manifest = json.loads((ROOT / "serve/runtime.json").read_text())
        self.assertEqual(sources, manifest["qwenfast"]["compilation_units"])
        self.assertTrue(set(sources).issubset(manifest["qwenfast"]["sources"]))
        self.assertIn("qf_a16moe.cu", sources)
        # qfa16_bind.cpp is the standalone binding for test_a16moe.py; qwenfast.so has one module.
        self.assertNotIn("qfa16_bind.cpp", sources)
        self.assertTrue((ROOT / "serve/qwenfast/qfa16_bind.cpp").is_file())
        self.assertIn("gdn_replay_update.cuh", manifest["qwenfast"]["sources"])
        for name in ("gdn_replay.cu", "head_run.cpp", "qf_drafthead_hier.cu", "qf_gdnout_new.cu"):
            self.assertIn(name, sources)
        bindings = []
        for name in sources:
            text = (ROOT / "serve/qwenfast" / name).read_text()
            # head_run.cpp has a disabled standalone module for its original WP test.
            if name == "head_run.cpp":
                self.assertIn("#ifdef WP_HEAD_STANDALONE", text)
                text = re.sub(r"#ifdef WP_HEAD_STANDALONE\b.*?#endif", "", text, flags=re.S)
            if re.search(r"^\s*PYBIND11_MODULE\s*\(", text, re.M):
                bindings.append(name)
        self.assertEqual(bindings, ["qf_bind.cpp"])

    def test_kern2_timings_and_no_carried_forward_kld(self):
        data = json.loads((ROOT / "results/20261020/decode-summary.json").read_text())
        self.assertEqual(len(data["rows"]), 8)
        self.assertEqual({r["completion_tokens"] for r in data["rows"]}, {4096})
        self.assertEqual({r["finish_reason"] for r in data["rows"]}, {"length"})
        rates = []
        for profile in ("code", "agent", "prose", "chinese"):
            rows = [r for r in data["rows"] if r["profile"] == profile]
            self.assertEqual({r["run"] for r in rows}, {0, 1})
            rates.append(sum(r["decode_tok_s"] for r in rows) / 2)
            for row in rows:
                self.assertAlmostEqual(
                    row["decode_tok_s"],
                    (row["completion_tokens"] - 1) / (row["elapsed_s"] - row["ttft_s"]),
                )
                self.assertNotIn("content", row)
                self.assertNotIn("reasoning", row)
        self.assertAlmostEqual(sum(rates) / 4, data["mean_decode_tok_s"])
        self.assertEqual(round(data["mean_decode_tok_s"], 1), 303.8)
        self.assertIsNone(data["decode_path_kld"])

    def test_draft_preparation_help_is_cpu_only(self):
        subprocess.run(
            [sys.executable, str(ROOT / "serve/prepare-draft-a16.py"), "--help"],
            check=True, capture_output=True,
        )

    def test_october_10_kld_data(self):
        directory = ROOT / "results/20261010/kld"
        expected = {
            "oct9-build-prefill-path.json": (0.011087, 0.1158, 0.9698),
            "oct9-build-decode-path.json": (0.011134, 0.1189, 0.9695),
            "oct10-build-decode-path.json": (0.011127, 0.1153, 0.9689),
        }
        self.assertEqual(sorted(p.name for p in directory.glob("*.json")), sorted(expected))
        for name, (kld, p99, top1) in expected.items():
            data = json.loads((directory / name).read_text())
            self.assertEqual(data["positions"], 45610, name)
            self.assertEqual(data["nonfinite"], 0, name)
            self.assertEqual(round(data["kld"], 6), kld, name)
            self.assertEqual(round(data["kld_p99"], 4), p99, name)
            self.assertEqual(round(data["top1_agree"], 4), top1, name)

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


class StepRateTests(unittest.TestCase):
    def test_median_of_single_request_lines(self):
        def line(time, running, accept, throughput):
            return (
                f"[2026-10-10 {time}] Decode batch, #running-req: {running}, #full token: 4096, "
                f"accept len: {accept}, gen throughput (token/s): {throughput}"
            )

        lines = [
            line("20:00:01", 1, "2.00", "200.00"),
            line("20:00:02", 1, "2.50", "275.00"),
            line("20:00:03", 1, "2.40", "264.00"),
            line("20:00:04", 2, "2.00", "400.00"),  # two requests: not a single-request step
            line("20:00:05", 1, "1.00", "40.00"),  # warm-up / idle tail
            line("20:00:09", 1, "2.00", "300.00"),  # outside the window
            "unrelated log line",
        ]
        rates = step_median.step_rates(lines, "20:00:00", "20:00:08")
        self.assertEqual(sorted(rates), [100.0, 110.0, 110.0])


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
