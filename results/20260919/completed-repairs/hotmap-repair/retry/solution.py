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
            elapsed=now-self.last
            if elapsed>0:
                self.tokens=min(self.capacity,self.tokens+elapsed*self.rate)
                self.last=now
            if self.tokens>=amount:
                self.tokens-=amount
                return True
            return False
