import unittest

import torch

from utils.security import SecurityException, safe_eval


class TestSafeEval(unittest.TestCase):
    def test_safe_arithmetic(self):
        self.assertEqual(safe_eval("1 + 1"), 2)
        self.assertEqual(safe_eval("a + b", local_vars={"a": 1, "b": 2}), 3)

    def test_safe_torch(self):
        t = torch.tensor([1.0])
        res = safe_eval(
            "torch.add(t, 1)", global_vars={"torch": torch}, local_vars={"t": t}
        )
        self.assertTrue(torch.equal(res, torch.tensor([2.0])))

    def test_unsafe_import(self):
        with self.assertRaises(SecurityException):
            safe_eval("import os")

    def test_unsafe_dunder(self):
        with self.assertRaises(SecurityException):
            safe_eval("__import__('os')")

    def test_unsafe_attribute(self):
        class Foo:
            pass

        f = Foo()
        # This should fail naturally because __class__ is blocked by name check?
        # Actually accessing .__class__ is an Attribute node with attr='__class__'
        with self.assertRaises(SecurityException):
            safe_eval("f.__class__", local_vars={"f": f})


if __name__ == "__main__":
    unittest.main()
