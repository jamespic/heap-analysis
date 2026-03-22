import pytest
from ..long_stack import run_with_long_stack


def test_long_stack():
    def long_recursion(n):
        if n == 0:
            return "done"
        return (yield long_recursion(n - 1))

    assert run_with_long_stack(long_recursion(1000000)) == "done"


def test_exceptions():
    class OddException(Exception):
        pass

    class EvenException(Exception):
        pass

    def long_recursion(n):
        if n == 0:
            raise EvenException()
        else:
            try:
                yield long_recursion(n - 1)
            except EvenException:
                raise OddException()
            except OddException:
                raise EvenException()

    with pytest.raises(EvenException):
        run_with_long_stack(long_recursion(10000))

    with pytest.raises(OddException):
        run_with_long_stack(long_recursion(9999))
