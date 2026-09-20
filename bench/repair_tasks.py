"""Bounded repository repair tasks, retaining xhigh and the same sampler."""

import real_tasks as tasks

tasks.TASKS = {
    "retry": """Fix the clock rollback bug in this HTTP client's TokenBucket. A failing regression starts clock=100, drains capacity4, moves clock to99, then moves to100.25; with rate2 only0.5 tokens should be available. Current code wrongly credits tokens. Preserve public APIs and all other behavior. Return the complete corrected module in one Python code block. This is a focused bug fix, not a redesign.
```python
from __future__ import annotations
import math, threading
from email.utils import parsedate_to_datetime
from datetime import timezone
def parse_retry_after(value: str | None, now: float) -> float | None:
    if value is None: return None
    value=value.strip()
    if value.isascii() and value.isdigit(): return float(value)
    try:
        d=parsedate_to_datetime(value)
        if d.tzinfo is None: d=d.replace(tzinfo=timezone.utc)
        return max(0.,d.timestamp()-now)
    except (ValueError,TypeError,OverflowError,OSError): return None
class TokenBucket:
    def __init__(self, rate: float, capacity: float, clock):
        if not math.isfinite(rate) or not math.isfinite(capacity) or rate<=0 or capacity<=0:
            raise ValueError('rate and capacity must be positive and finite')
        self.rate=rate;self.capacity=capacity;self.clock=clock
        self.tokens=capacity;self.last=clock();self.lock=threading.Lock()
    def acquire(self, amount: float=1) -> bool:
        if not math.isfinite(amount) or amount<=0 or amount>self.capacity: raise ValueError('invalid amount')
        with self.lock:
            now=self.clock()
            self.tokens=min(self.capacity,self.tokens+max(0,now-self.last)*self.rate)
            self.last=now
            if self.tokens>=amount:
                self.tokens-=amount
                return True
            return False
```
""",
    "migrate": r"""Fix this SQLite migration runner's atomicity bug. executescript commits the transaction before running statements, so a later failed migration leaves partial changes behind. Each string in statements is exactly one SQL statement. Preserve existing validation, ordering, checksums, APIs and idempotence. A call must roll back all changes including creating _migrations when a statement fails. Return the complete corrected module in one Python code block. This is a focused bug fix, not a redesign.
```python
import hashlib,json
def apply_migrations(connection,migrations):
    if connection.in_transaction: raise ValueError('active transaction')
    records=list(migrations)
    versions=[v for v,_ in records]
    if any(type(v) is not int or v<=0 for v in versions) or len(set(versions))!=len(versions):
        raise ValueError('invalid or duplicate versions')
    records.sort(key=lambda x:x[0])
    def checksum(statements):
        return hashlib.sha256(json.dumps(statements,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()
    applied=[]
    connection.execute('BEGIN IMMEDIATE')
    try:
        connection.execute('CREATE TABLE IF NOT EXISTS _migrations(version INTEGER PRIMARY KEY,checksum TEXT NOT NULL)')
        existing=dict(connection.execute('SELECT version,checksum FROM _migrations'))
        for v,statements in records:
            if v in existing and existing[v]!=checksum(statements):raise ValueError('checksum changed')
        for v,statements in records:
            if v in existing:continue
            connection.executescript(';\n'.join(statements))
            connection.execute('INSERT INTO _migrations VALUES(?,?)',(v,checksum(statements)))
            applied.append(v)
        connection.commit()
        return applied
    except BaseException:
        connection.rollback()
        raise
```
""",
}
if __name__ == "__main__":
    tasks.run(tasks.arguments())
