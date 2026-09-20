
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
