import math

from Lab05.src.autodiff import Value


def test_gradient_matches_expected_for_shared_parent() -> None:
    value = Value(3.0)
    output = value * value + value
    output.backward()

    assert output.data == 12.0
    assert value.grad == 7.0


def test_relu_and_power_gradients() -> None:
    positive = Value(2.0)
    negative = Value(-2.0)
    output = (positive**3).relu() + (negative**2).relu()
    output.backward()

    assert math.isclose(positive.grad, 12.0)
    assert math.isclose(negative.grad, -4.0)


def test_repeated_backward_accumulates_gradients() -> None:
    value = Value(4.0)
    output = value * 2.0

    output.backward()
    output.backward()

    assert value.grad == 4.0
