# Test an infinite recursion bug while building a VP tree

import logging
import random

import Levenshtein
import pytest

import metric_dbscan
from metric_dbscan.locator import vantage_point_tree

def random_string(alphabet: str, length: int) -> str:
    """Generate a random string from a set of characters

        Arguments:
            alphabet {string}: Characters to choose from
            length {int}: How many characters to choose

        Returns:
            String composed of characters chosen at random from alphabet
    """

    new_word = [random.choice(alphabet) for _ in range(length)]
    return ''.join(new_word)

@pytest.fixture
def tight_cluster1():
    return ['A' + random_string("abcde", 10) for _ in range(400)]

@pytest.fixture
def tight_cluster2():
    return ['B' + random_string("hijkl", 10) for _ in range(400)]

@pytest.fixture
def tight_cluster3():
    return ['C' + random_string("mnopq", 10) for _ in range(400)]

@pytest.fixture
def tight_cluster4():
    return ['D' + random_string("rstuv", 10) for _ in range(400)]

@pytest.fixture
def tight_clusters(tight_cluster1, tight_cluster2, tight_cluster3, tight_cluster4):
    return tight_cluster1 + tight_cluster2 + tight_cluster3 + tight_cluster4

def test_vptree_infinite_recursion(tight_clusters):
    my_tree = vantage_point_tree.VantagePointTree(Levenshtein.distance, tight_clusters)

def test_tight_clusters_dbscan(tight_cluster1,
                               tight_cluster2,
                               tight_cluster3,
                               tight_cluster4,
                               tight_clusters):
    logging.getLogger().setLevel(logging.DEBUG)
    cluster_ids = metric_dbscan.cluster_items(tight_clusters,
                                              Levenshtein.distance,
                                              9,
                                              5)

    clusters = {0: [], 1: [], 2: [], 3: []}
    assert len(tight_clusters) == len(cluster_ids)
    for (i, cluster_id) in enumerate(cluster_ids):
        clusters[cluster_id].append(tight_clusters[i])

    # We don't know which of the input clusters will get which cluster ID;
    # figure that out here.
    input_clusters = {
        'A': sorted(tight_cluster1),
        'B': sorted(tight_cluster2),
        'C': sorted(tight_cluster3),
        'D': sorted(tight_cluster4)
    }
    output_clusters = {}
    for (cluster_id, words) in clusters.items():
        output_clusters[words[0][0]] = sorted(words)

    assert len(output_clusters) == 4

    for first_letter in input_clusters.keys():
        assert input_clusters[first_letter] == output_clusters[first_letter]
