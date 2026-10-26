"""Public helpers for the ported October 20 GPU tests; no private captures."""
import importlib.machinery
import importlib.util
import os
import statistics
import torch


def extension():
    path = os.environ.get("QWENFAST_SO", "/opt/qwenfast/qwenfast.so")
    loader = importlib.machinery.ExtensionFileLoader("qwenfast", path)
    spec = importlib.util.spec_from_loader("qwenfast", loader, origin=path)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def bench(fn, calls=24, replays=20):
    for i in range(3):
        fn(i)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(calls):
            fn(i)
    graph.replay()
    values = []
    for _ in range(3):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(replays):
            graph.replay()
        b.record()
        b.synchronize()
        values.append(a.elapsed_time(b) * 1000 / (calls * replays))
    return statistics.median(values)


def err(a, b):
    bitwise = a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )
    delta = (a.float() - b.float()).abs()
    return dict(bitwise=bitwise, max_abs=delta.max().item())
