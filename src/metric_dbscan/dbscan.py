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

"""DBSCAN density-based clustering for metric spaces"""

import collections
from typing import Callable
from tqdm import trange

from metric_dbscan.locator import spatial_index
from metric_dbscan.locator import wrapping
from metric_dbscan.locator import vantage_point_tree as vptree

from metric_dbscan.dbscan_types import ClusterableItem, DistanceFunction, OUTLIER
from typing import Callable, List

import psutil

# This is for benchmarking purposes only.  After we run DBSCAN, we save the
# process's memory usage so that we can tell how much we used before
# garbage collection comes in and cleans up.  It is not meant for general
# use.
SAVED_USS = None

NeighborSearchFunction = Callable[[int], List[int]]

OUTLIER = -1

def cluster_items(items: List[ClusterableItem],
                  distance_function: DistanceFunction,
                  minimum_cluster_size: int,
                  maximum_neighbor_distance: float) -> List[int]:
    """Group items into clusters using DBSCAN

    This algorithm assigns an integer label to each item in the input.
    A label L >= 0 indicates that item N belongs to cluster L.  A label
    of -1 (aka metric_dbscan.OUTLIER) indicates that item L
    is an outlier that does not belong to any cluster.

    Arguments:
        items (list): items to cluster.  This really does need to be
            a list rather than a generic iterable because we return cluster
            IDs as a list with the same length as the input.
        distance_function (function from 2 items -> float): Metric
            function that measures distance between items.
        minimum_cluster_size (int): How many nearby neighbors a item
            must have to be considered a "core item" for a cluster.
            Must be positive.
        maximum_neighbor_distance (float): How close two items must be
            in order to be considered part of the same cluster.  Must
            be positive.

    Returns:
        List of item labels represented as integers.  A label of -1 indicates
            that the corresponding item is an outlier that does not belong to
            any cluster.

    Raises:
        ValueError: either minimum_cluster_size <= 1 or
            maximum_neighbor_distance <= 0
    """

    global SAVED_USS

    if minimum_cluster_size <= 1:
        raise ValueError("DBSCAN: minimum cluster size must be at least 2.")

    if maximum_neighbor_distance <= 0:
        raise ValueError("DBSCAN: maximum neighbor distance must be positive")

    # A previous version of metric DBSCAN had locator_type as a keyword
    # argument in case one wanted to substitute some other spatial index.
    # We never used that in practice, so we've removed the argument but
    # left the infrastructure in place in case you'd like to play with it.
    locator_type = vptree.VantagePointTree

    num_items = len(items)
    cluster_labels = [None] * num_items
    current_item_id = 0
    next_cluster_id = 0

    find_neighbor_item_ids = _build_locator_function(locator_type,
                                                     items,
                                                     distance_function,
                                                     maximum_neighbor_distance)

    # This code is almost straight out of the Wikipedia article on DBSCAN.
    # We've added a guard (the check for 'pid in
    # items_processed_this_cluster') to keep from repeatedly checking the
    # same items in really dense clusters.

    for current_item_id in trange(num_items):
        if cluster_labels[current_item_id] is not None:
            continue

        neighbor_ids = find_neighbor_item_ids(current_item_id)
        if len(neighbor_ids) < minimum_cluster_size:
            cluster_labels[current_item_id] = OUTLIER
            continue

        # We found an unlabeled core item.  Label it and start a new cluster.
        current_cluster_id = next_cluster_id
        next_cluster_id += 1
        cluster_labels[current_item_id] = current_cluster_id

        potential_expansion_items = set(neighbor_ids)
        potential_expansion_items.remove(current_item_id)
        items_processed_this_cluster = set([current_item_id])

        while len(potential_expansion_items) > 0:
            neighbor_id = potential_expansion_items.pop()
            items_processed_this_cluster.add(neighbor_id)

            if cluster_labels[neighbor_id] == OUTLIER:
                # The item isn't noise; it's an edge item of the current
                # cluster
                cluster_labels[neighbor_id] = current_cluster_id
                continue
            elif cluster_labels[neighbor_id] is not None:
                # The item has already been labeled as part of another cluster
                # NOTE: This is where nondeterminism can happen.  It is possible
                # in DBSCAN for a non-core item to be reachable from core items
                # of multiple different clusters.  Whichever label gets applied
                # first wins.  There is a variant of DBSCAN called DBSCAN*
                # that assigns the outlier label to all items that are not core
                # items.  That version is deterministic.
                continue
            else:
               cluster_labels[neighbor_id] = current_cluster_id
               more_expansion_items = find_neighbor_item_ids(neighbor_id)
               if len(more_expansion_items) >= minimum_cluster_size:
                   # The neighbor item we're looking at is also a core item.
                   # Keep expanding by looking at all of its neighbors too.
                   for pid in more_expansion_items:
                       if not pid in items_processed_this_cluster:
                           potential_expansion_items.add(pid)

    SAVED_USS = psutil.Process().memory_full_info().uss
    return _remap_by_size(cluster_labels)


def _remap_by_size(initial_labels: List[int]) -> List[int]:
    """Remap cluster IDs so largest cluster is 0

    This is a quality-of-life improvement.  We will return
    clusters in descending order by size.  0 will be the
    largest, 1 the next largest, and so on.

    Cluster -1 will always be the outliers.

    Arguments:
        initial_labels (list of int): Cluster labels computed
            with DBSCAN

    Returns:
        New list of cluster labels sorted as described above
    """

    cluster_sizes = collections.Counter(initial_labels)
    if -1 in cluster_sizes:
        cluster_sizes.pop(-1)

    sizes_and_labels = [(size, cid) for (cid, size) in cluster_sizes.items()]
    sizes_and_labels = sorted(sizes_and_labels, reverse = True)
    sorted_old_labels = [si[1] for si in sizes_and_labels]
    remap_labels = {
        old_label: new_label
        for (new_label, old_label) in enumerate(sorted_old_labels)
    }
    remap_labels[-1] = -1

    new_labels = [
        remap_labels[old_label] for old_label in initial_labels
    ]
    return new_labels


def _build_locator_function(locator_type: type[spatial_index.SpatialIndex],
                            items: List[ClusterableItem],
                            distance_fn: DistanceFunction,
                            query_distance: float) -> NeighborSearchFunction:
    """Internal utility function -- do not call from user code

    Our DBSCAN implementation constructs a mapping from integer item ID
    to a cluster ID.  This function wraps up all the bookkeeping of
    doing queries on the user's items but referring to them with their IDs.

    Arguments:
        locator_type (metric_dbscan.locator.spatial_index.SpatialIndex):
            Callable (class or function) that will instantiate a spatial index
        items (list of ClusterableItem): The original items the user passed
            in for clustering
        distance_fn (DistanceFunction): The original distance function from
            the user
        query_distance (float): the DBSCAN epsilon parameter (how close two
            items must be to be considered neighbors)

    Returns:
        New function: (item id) -> (list of neighboring item IDs)
    """

    wrapped_items = wrapping.add_item_ids(items)
    wrapped_metric = wrapping.wrap_distance_function(distance_fn)

    locator = locator_type(wrapped_metric, wrapped_items)

    def find_nearby_neighbors(query_item_id: int) -> List[int]:
        wrapped_neighbors = locator.find_items_within_radius(wrapped_items[query_item_id],
                                                              query_distance)
        item_ids = [
            wrapping.item_id(p) for p in wrapped_neighbors
        ]
        return item_ids

    return find_nearby_neighbors

