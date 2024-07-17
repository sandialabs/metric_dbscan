### Copyright 2024 National Technology & Engineering Solutions of Sandia,
### LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
### U.S. Government retains certain rights in this software.
###
### Redistribution and use in source and binary forms, with or without
### modification, are permitted provided that the following conditions are
### met:
###
### 1. Redistributions of source code must retain the above copyright
###    notice, this list of conditions and the following disclaimer.
###
### 2. Redistributions in binary form must reproduce the above copyright
###    notice, this list of conditions and the following disclaimer in
###    the documentation and/or other materials provided with the
###    distribution.
###
### 3. Neither the name of the copyright holder nor the names of its
###    contributors may be used to endorse or promote products derived
###    from this software without specific prior written permission.
###
### THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
### “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
### LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
### A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
### HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
### SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
### LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
### DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
### THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
### (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
### OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import random

import pytest

from metric_dbscan.locator import vantage_point_tree as vptree

def real_line_distance(a, b) -> float:
    return math.fabs(a-b)

@pytest.fixture
def tree_with_integers():
    contents = list(range(100))
    random.shuffle(contents)
    tree = vptree.VantagePointTree(real_line_distance, contents)
    return tree


def test_vptree_population():
    contents = range(100)
    tree = vptree.VantagePointTree(real_line_distance, contents)
    tree.print()

def test_items_in_closed_ball(tree_with_integers):
    tree_with_integers.print()
    items_in_ball = tree_with_integers.find_items_within_radius(10, 3)
    assert sorted(items_in_ball) == [7, 8, 9, 10, 11, 12, 13]

def test_items_in_open_ball(tree_with_integers):
    items_in_ball = tree_with_integers.find_items_within_radius(
        10, 3, include_boundary=False)
    assert sorted(items_in_ball) == [8, 9, 10, 11, 12]

def test_k_nearest_neighbors_key_in_tree(tree_with_integers):
    nearest_neighbors = tree_with_integers.k_nearest_neighbors(50, 6)
    assert 50 not in nearest_neighbors
    expected_neighbors = [47, 48, 49, 51, 52, 53]
    for neighbor in expected_neighbors:
        assert neighbor in nearest_neighbors


    # Make sure they're sorted by increasing distance from 50
    distances_from_50 = [
        math.fabs(neighbor - 50) for neighbor in nearest_neighbors
    ]
    for i in range(1, len(distances_from_50)):
        assert distances_from_50[i] >= distances_from_50[i-1]


def test_k_nearest_neighbors_key_not_in_tree(tree_with_integers):
    nearest_neighbors = tree_with_integers.k_nearest_neighbors(50.1, 7)

    expected_neighbors = [47, 48, 49, 50, 51, 52, 53]
    for neighbor in expected_neighbors:
        assert neighbor in nearest_neighbors

    # Make sure they're sorted by their distance from 50
    distances_from_50 = [
        math.fabs(neighbor - 50) for neighbor in nearest_neighbors
    ]
    for i in range(1, len(distances_from_50)):
        assert distances_from_50[i] >= distances_from_50[i-1]


if __name__ == '__main__':
    test_vptree_population()

