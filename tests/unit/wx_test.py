import unittest


class WxTestCase(unittest.TestCase):
    def __str__(self):
        return f"{self.__class__.__qualname__}.{self._testMethodName}"


class WxTestSuite(unittest.TestSuite):
    def run(self, result, debug=False):
        for test in self:
            # print(f"running {test.__class__}.{test._testMethodName}")
            print(f"Running {test}")
            test.run(result)
        return result
