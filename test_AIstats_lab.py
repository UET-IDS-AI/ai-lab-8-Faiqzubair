import numpy as np
import AI_stats_lab as lab


def test_sigmoid():
    assert abs(lab.sigmoid(0) - 0.5) < 1e-6
    assert lab.sigmoid(5) > 0.99
    assert lab.sigmoid(-5) < 0.01


def test_forward_pass_shapes():
    X = np.array([[5.0, 3.0]])

    W1 = np.array([[1.0, -1.0, 3.0],
                   [2.0,  2.0, -1.0]])

    W2 = np.array([[0.2, -0.1],
                   [0.4,  0.3],
                   [-0.5, 0.2]])

    W3 = np.array([[0.7],
                   [-0.4]])

    h1, h2, y = lab.forward_pass(X, W1, W2, W3)

    assert h1.shape == (1, 3)
    assert h2.shape == (1, 2)
    assert y.shape == (1, 1)


def test_forward_pass_values():
    X = np.array([[5.0, 3.0]])

    W1 = np.array([[1.0, -1.0, 3.0],
                   [2.0,  2.0, -1.0]])

    W2 = np.array([[0.2, -0.1],
                   [0.4,  0.3],
                   [-0.5, 0.2]])

    W3 = np.array([[0.7],
                   [-0.4]])

    h1, h2, y = lab.forward_pass(X, W1, W2, W3)

    expected_h1 = 1 / (1 + np.exp(-X.dot(W1)))
    expected_h2 = 1 / (1 + np.exp(-expected_h1.dot(W2)))
    expected_y = 1 / (1 + np.exp(-expected_h2.dot(W3)))

    assert np.allclose(h1, expected_h1)
    assert np.allclose(h2, expected_h2)
    assert np.allclose(y, expected_y)


def test_backward_pass_shapes():
    X = np.array([[5.0, 3.0]])

    W1 = np.array([[1.0, -1.0, 3.0],
                   [2.0,  2.0, -1.0]])

    W2 = np.array([[0.2, -0.1],
                   [0.4,  0.3],
                   [-0.5, 0.2]])

    W3 = np.array([[0.7],
                   [-0.4]])

    label = 1

    h1, h2, y = lab.forward_pass(X, W1, W2, W3)
    dW1, dW2, dW3, loss = lab.backward_pass(X, h1, h2, y, label, W1, W2, W3)

    assert dW1.shape == W1.shape
    assert dW2.shape == W2.shape
    assert dW3.shape == W3.shape
    assert np.isscalar(loss) or loss.shape == (1, 1)


def test_backward_pass_values_label_1():
    X = np.array([[5.0, 3.0]])

    W1 = np.array([[1.0, -1.0, 3.0],
                   [2.0,  2.0, -1.0]])

    W2 = np.array([[0.2, -0.1],
                   [0.4,  0.3],
                   [-0.5, 0.2]])

    W3 = np.array([[0.7],
                   [-0.4]])

    label = 1

    h1, h2, y = lab.forward_pass(X, W1, W2, W3)
    dW1, dW2, dW3, loss = lab.backward_pass(X, h1, h2, y, label, W1, W2, W3)

    expected_loss = -np.log(y)

    dJ_dy = -1 / y
    dy_dz3 = y * (1 - y)
    grad3 = dJ_dy * dy_dz3

    expected_dW3 = h2.T.dot(grad3)

    dJ_dh2 = grad3.dot(W3.T)
    dh2_dz2 = h2 * (1 - h2)
    grad2 = dJ_dh2 * dh2_dz2

    expected_dW2 = h1.T.dot(grad2)

    dJ_dh1 = grad2.dot(W2.T)
    dh1_dz1 = h1 * (1 - h1)
    grad1 = dJ_dh1 * dh1_dz1

    expected_dW1 = X.T.dot(grad1)

    assert np.allclose(loss, expected_loss)
    assert np.allclose(dW3, expected_dW3)
    assert np.allclose(dW2, expected_dW2)
    assert np.allclose(dW1, expected_dW1)


def test_backward_pass_values_label_0():
    X = np.array([[5.0, 3.0]])

    W1 = np.array([[1.0, -1.0, 3.0],
                   [2.0,  2.0, -1.0]])

    W2 = np.array([[0.2, -0.1],
                   [0.4,  0.3],
                   [-0.5, 0.2]])

    W3 = np.array([[0.7],
                   [-0.4]])

    label = 0

    h1, h2, y = lab.forward_pass(X, W1, W2, W3)
    dW1, dW2, dW3, loss = lab.backward_pass(X, h1, h2, y, label, W1, W2, W3)

    expected_loss = -np.log(1 - y)

    dJ_dy = 1 / (1 - y)
    dy_dz3 = y * (1 - y)
    grad3 = dJ_dy * dy_dz3

    expected_dW3 = h2.T.dot(grad3)

    dJ_dh2 = grad3.dot(W3.T)
    dh2_dz2 = h2 * (1 - h2)
    grad2 = dJ_dh2 * dh2_dz2

    expected_dW2 = h1.T.dot(grad2)

    dJ_dh1 = grad2.dot(W2.T)
    dh1_dz1 = h1 * (1 - h1)
    grad1 = dJ_dh1 * dh1_dz1

    expected_dW1 = X.T.dot(grad1)

    assert np.allclose(loss, expected_loss)
    assert np.allclose(dW3, expected_dW3)
    assert np.allclose(dW2, expected_dW2)
    assert np.allclose(dW1, expected_dW1)
