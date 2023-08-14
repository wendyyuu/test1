import unittest
from gradescope_utils.autograder_utils.decorators import weight, visibility, number
from casadi_test import eval


class TestComplex(unittest.TestCase):
    def setUp(self):
        
        pass

    @weight(2)
    @number("1.0")
    def test_eval_negative_number(self):
        """Evaluate -2 + 6"""
        val = eval()
        print(val)
        self.assertEqual(val, 4)

# A = TestComplex()
# A.test_eval_negative_number()        
