from zerogercrnn.lib.utils.split import get_split_indexes


def test_get_split_indexes_should_not_intersect():
    train, validation, test = get_split_indexes(100, validation_percentage=0.2, test_percentage=0.2)

    assert len(set(train)) == 60
    assert len(set(validation)) == 20
    assert len(set(test)) == 20

    train = set(train)
    validation = set(validation)
    test = set(test)

    assert len(train.union(validation).union(test)) == 100