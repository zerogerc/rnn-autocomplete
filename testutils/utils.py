def assert_numbers_almost_equal(x, y, eps=1e-9):
    assert abs(x - y) < eps


def assert_tensors_equal(t1, t2, eps=1e-9):
    assert t1.size() == t2.size()

    t1 = t1.view(-1)
    t2 = t2.view(-1)

    for i in range(t1.size()[0]):
        assert_numbers_almost_equal(t1[i], t2[i], eps=eps)
