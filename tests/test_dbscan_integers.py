# Test DBSCAN on a nice easy case -- integers

import math

import metric_dbscan

import pytest

@pytest.fixture
def zero_to_one_hundred():
    return list(range(0, 100))

@pytest.fixture
def one_thousand_to_two_thousand():
    return list(range(1000, 2000))

@pytest.fixture
def integers_to_cluster(zero_to_one_hundred, one_thousand_to_two_thousand):
    outliers = [-10000, 10000]
    return zero_to_one_hundred + one_thousand_to_two_thousand + outliers

def test_dbscan_integers(integers_to_cluster):
    def integer_distance(a, b):
        return math.fabs(a - b)

    actual_labels = metric_dbscan.cluster_items(integers_to_cluster,
                                                integer_distance,
                                                4, # max neighbor distance
                                                5 # min cluster size
                                                )

    assert len(set(actual_labels[0:100])) == 1
    assert len(set(actual_labels[1100:2100])) == 1
    assert actual_labels[-1] == metric_dbscan.OUTLIER
    assert actual_labels[-2] == metric_dbscan.OUTLIER