
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
