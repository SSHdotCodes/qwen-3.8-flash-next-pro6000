"""Fresh completed programming tasks; fixed held-out tests run without network/GPU."""

import argparse, json, re, os, subprocess, hashlib, uuid
from pathlib import Path
import requests
from sustained import payload, request

A = Path(__file__).resolve().parents[1] / "results/local"
IMAGE = "local/qwen-flash-next:0.5.20-20260919-hotmap"
TASKS = {
    "retry": """Implement a production-quality Python standard-library-only module for a HTTP client. Return exactly one Python code block containing the complete module. Public API:
parse_retry_after(value: str | None, now: float) -> float | None: HTTP delta-seconds (nonnegative ASCII integer) or RFC HTTP date converted to seconds after the supplied Unix timestamp; past dates return 0. Invalid/negative/fractional values return None. Strip outer whitespace.
TokenBucket(rate: float, capacity: float, clock): thread-safe rate limiter, initially full; positive finite rate and capacity required (ValueError otherwise). clock is an injected callable returning monotonic seconds. acquire(amount: float=1) -> bool atomically refills to capacity and consumes if enough tokens, otherwise false without consuming. Amount must be finite and >0 and <=capacity, otherwise ValueError. Never move the internal last-refill time backward if clock decreases. No sleeps. Include useful docstrings and type annotations. Focus on robust edge cases and concurrency. Do not include a CLI or your own tests.""",
    "migrate": """Implement a production-quality standard-library Python SQLite migration module. Return exactly one Python code block containing the complete module. Public API apply_migrations(connection, migrations) -> list[int], where migrations is an iterable of (version:int, statements:list[str]). Each statement is exactly one SQL statement. Version must be a positive integer (bool invalid), with no duplicates; input order can vary, apply in increasing version order. Validate all version values and duplicates before any DB mutation. Maintain table _migrations(version INTEGER PRIMARY KEY, checksum TEXT NOT NULL). The checksum is SHA256 of UTF8 json.dumps(statements,ensure_ascii=False,separators=(',',':')). Already applied versions with matching checksum are skipped. If an existing version has changed statements raise ValueError before applying any new migration. If connection already has an active transaction raise ValueError without altering it. Use one explicit transaction for the entire call including creating _migrations, all new statements, and journal inserts; any failure rolls everything back and leaves connection out of transaction. Do not use executescript, which commits implicitly. Return only newly applied versions. Include type hints and docstrings. No CLI or embedded tests.""",
}
TESTS = {
    "retry": r"""
import unittest, math, threading
from solution import parse_retry_after, TokenBucket
class Checks(unittest.TestCase):
 def test_dates(self):
  self.assertEqual(parse_retry_after(' 120 ',0),120)
  self.assertEqual(parse_retry_after('Wed, 21 Oct 2015 07:28:00 GMT',1445412470),10)
  self.assertEqual(parse_retry_after('Wed, 21 Oct 2015 07:28:00 GMT',1445412490),0)
  for x in [None,'','1.5','-1','nonsense','NaN','+3','١٢']:
   with self.subTest(x=x):self.assertIsNone(parse_retry_after(x,0))
 def test_refill(self):
  now=[100.]; b=TokenBucket(2,4,lambda:now[0])
  self.assertTrue(b.acquire(4));self.assertFalse(b.acquire())
  now[0]=100.25;self.assertFalse(b.acquire());self.assertTrue(b.acquire(.5))
  now[0]=99.;self.assertFalse(b.acquire());now[0]=100.5;self.assertTrue(b.acquire(.5));self.assertFalse(b.acquire(.1))
  now[0]=200.;self.assertTrue(b.acquire(4));self.assertFalse(b.acquire())
 def test_validation(self):
  for rate,cap in [(0,2),(-1,2),(1,0),(float('inf'),2),(1,float('nan'))]:
   with self.assertRaises(ValueError):TokenBucket(rate,cap,lambda:0)
  b=TokenBucket(1,2,lambda:0)
  for amount in [0,-1,3,float('nan'),float('inf')]:
   with self.assertRaises(ValueError):b.acquire(amount)
  self.assertTrue(b.acquire(2))
 def test_atomic(self):
  b=TokenBucket(1,37,lambda:0); wins=[]; lock=threading.Lock()
  def worker():
   n=sum(b.acquire() for _ in range(50))
   with lock:wins.append(n)
  ts=[threading.Thread(target=worker) for _ in range(16)]
  [t.start() for t in ts];[t.join() for t in ts];self.assertEqual(sum(wins),37)
unittest.main()
""",
    "migrate": r"""
import unittest,sqlite3,hashlib,json
from solution import apply_migrations
class Checks(unittest.TestCase):
 def test_order_idempotence(self):
  c=sqlite3.connect(':memory:'); a=['CREATE TABLE t(v INTEGER)']; b=['INSERT INTO t VALUES(42)']
  self.assertEqual(apply_migrations(c,[(2,b),(1,a)]),[1,2]);self.assertFalse(c.in_transaction)
  self.assertEqual(apply_migrations(c,[(1,a),(2,b)]),[])
  self.assertEqual(c.execute('select v from t').fetchall(),[(42,)])
  self.assertEqual(c.execute('select checksum from _migrations where version=1').fetchone()[0],hashlib.sha256(json.dumps(a,ensure_ascii=False,separators=(',',':')).encode()).hexdigest())
  with self.assertRaises(ValueError):apply_migrations(c,[(3,['INSERT INTO t VALUES(9)']),(1,['SELECT 1'])])
  self.assertEqual(c.execute('select v from t').fetchall(),[(42,)]);self.assertFalse(c.in_transaction)
 def test_atomic_new_database(self):
  c=sqlite3.connect(':memory:')
  with self.assertRaises(sqlite3.Error):apply_migrations(c,[(1,['CREATE TABLE t(v)']),(2,['INSERT INTO missing VALUES(4)'])])
  self.assertFalse(c.in_transaction);self.assertEqual(c.execute("select name from sqlite_master where type='table'").fetchall(),[])
 def test_invalid_before_mutation(self):
  for versions in [[1,1],[1,0],[1,-2],[1,True],[1,1.5]]:
   c=sqlite3.connect(':memory:')
   with self.assertRaises(ValueError):apply_migrations(c,[(v,['SELECT 1']) for v in versions])
   self.assertEqual(c.execute('select name from sqlite_master').fetchall(),[])
 def test_caller_transaction(self):
  c=sqlite3.connect(':memory:');c.execute('CREATE TABLE keep(x)');c.execute('INSERT INTO keep VALUES(7)')
  with self.assertRaises(ValueError):apply_migrations(c,[])
  self.assertTrue(c.in_transaction);self.assertEqual(c.execute('select x from keep').fetchall(),[(7,)])
 def test_later_rollback(self):
  c=sqlite3.connect(':memory:');apply_migrations(c,[(1,['CREATE TABLE t(x)'])])
  with self.assertRaises(sqlite3.Error):apply_migrations(c,[(2,['INSERT INTO t VALUES(3)','SELECT * FROM missing'])])
  self.assertFalse(c.in_transaction);self.assertEqual(c.execute('select * from t').fetchall(),[])
  self.assertEqual(c.execute('select version from _migrations').fetchall(),[(1,)])
unittest.main()
""",
}


def run(args):
    tag = args.tag
    A = Path(args.outdir).resolve()
    A.mkdir(parents=True, exist_ok=True)
    out = []
    base = f"http://127.0.0.1:{args.port}"
    for i, (name, prompt) in enumerate(TASKS.items()):
        if args.flush_cache:
            requests.post(base + "/flush_cache", timeout=30).raise_for_status()
        row = request(
            base, payload(args.model, prompt, args.max_tokens, "xhigh", 20260919 + i)
        )
        row.update(task=name, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", row["content"], re.S)
        d = A / "real-output" / tag / name
        d.mkdir(parents=True, exist_ok=True)
        code = max(blocks, key=len) if blocks else row["content"]
        (d / "solution.py").write_text(code)
        (d / "checks.py").write_text(TESTS[name])
        container = "flash-repair-check-" + uuid.uuid4().hex
        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            container,
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--cpus=2",
            "--memory=512m",
            "--pids-limit=64",
            "--user=65534:65534",
            "--tmpfs=/tmp:rw,nosuid,nodev,size=64m",
            "-v",
            str(d) + ":/task:ro",
            "-w",
            "/task",
            args.validation_image,
            "python3",
            "-B",
            "checks.py",
        ]
        try:
            v = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=45,
                env={
                    **os.environ,
                    "DOCKER_HOST": os.environ.get(
                        "DOCKER_HOST", f"unix:///run/user/{os.getuid()}/docker.sock"
                    ),
                },
            )
            row.update(tests_pass=v.returncode == 0, validation=v.stdout + v.stderr)
        except subprocess.TimeoutExpired:
            row.update(tests_pass=False, validation="Validation timeout")
        finally:
            subprocess.run(
                ["docker", "rm", "-f", container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                env={
                    **os.environ,
                    "DOCKER_HOST": os.environ.get(
                        "DOCKER_HOST", f"unix:///run/user/{os.getuid()}/docker.sock"
                    ),
                },
            )
        out.append(row)
        (A / (tag + "-real.json")).write_text(
            json.dumps(out, indent=2, ensure_ascii=False)
        )
        print(
            json.dumps(
                {k: v for k, v in row.items() if k not in ("content", "reasoning")},
                ensure_ascii=False,
            ),
            flush=True,
        )


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tag")
    p.add_argument("--port", type=int, default=30010)
    p.add_argument("--model", default="qwen3.8-flash-next")
    p.add_argument("--outdir", default=str(A))
    p.add_argument("--max-tokens", type=int, default=65536)
    p.add_argument("--validation-image", default=IMAGE)
    p.add_argument("--flush-cache", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(arguments())
