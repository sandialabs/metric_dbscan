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

"""vantage_point_tree - Spatial index for general metric spaces

This module provides a single class, VantagePointTree, that
accelerates nearest-neighbor, K-nearest-neighbors, and
points-inside-ball searches for general metric spaces.  It
requires that you supply a distance function along with the
items to be indexed.

Members:

VantagePointTree (class)
"""

import logging
import math
import random
import statistics

from metric_dbscan.locator import spatial_index

from typing import Any, Callable, List, NewType, Optional, Sequence, Tuple

Indexable = NewType("Indexable", Any)
MetricFunction = Callable[[Indexable, Indexable], float]
DistanceWithIndexable = Tuple[float, Indexable]

LOG = logging.getLogger(__name__)

class VantagePointTree(spatial_index.SpatialIndex):
    """Spatial locator for general metric spaces

    This class allows you to search for nearby items in a general metric
    space.  A general metric space is one where you can compute the distance
    between any two items, but not a path between those two items, and
    is defined by the type of item and the function used to compute
    distance.  Character strings plus Levenshtein distance form a metric
    space, as do binary numbers of fixed length plus Hamming distance.

    This implementation does not yet support adding items to an already-
    initialized tree, including adding points one at a time.  You must
    supply the list of items to index up front as an argument to the
    constructor.

    Methods:
        items_within_ball: Find items within a given radius of an example
        nearest_neighbor: Find the item nearest an exemplar
        k_nearest_neighbors: Find the k items nearest an exemplar
    """


    def __init__(self,
                 metric_function: MetricFunction,
                 items: Sequence[Indexable],
                 depth: int=0,
                 max_items_per_node: int=10,
                 max_depth: int=20,
                 min_split_fraction: float=0.01,
                 max_shuffle_count: int=5):
        """Initialize a new vantage point tree

        You can create a new vantage point tree with just a metric function.
        You can also initialize it with a list of objects to index if you
        have that list available.

        The metric function must take two arguments, each of which is one
        of the items being indexed in the tree, and return a non-negative
        number.  This distance must satisfy the properties of a metric.
        Consult the Wikipedia article on metric spaces
        (https://en.wikipedia.org/wiki/Metric_space) for details.  In short:
        d(x, y) == 0 implies x == y; x != y implies d(x, y) > 0;
        d(x, y) == d(y, x), and d(x, z) <= d(x, y) + d(y, z).

        Vantage point trees perform best when the distances between points
        are evenly distributed.  If they are not, or (especially) if the
        set of distances has low cardinality (string edit distance between
        short strings, for example), we can get into a situation where the
        construction algorithm cannot split the nodes to continue building
        the tree.  In that situation we take the following measures:

        1. Try splitting on the mean distance instead of the median.
        2. Choose a different anchor (at random).
        3. If all else fails, stop trying to split the data and then
           create a single node with all the remaining items.

        Arguments:
            metric_function {vantage_point_tree.MetricFunction}: Distance
                metric for this space.
            items {sequence of indexable items}: Items to insert into the tree.

        Keyword Arguments:
            max_items_per_node {int}: When building the tree, any node
                with more than this many items will be split into a subtree.
                Defaults to 10.  Don't go below 3.
            max_depth {int}: How deep the tree can be.  At this depth, we
                will create a single node with all the items remaining on
                the branch.  This is a guard against infinite recursion.
                Defaults to 20.
            min_split_fraction {float}: Another guard against infinite
                recursion.  When we partition the nodes into nearby and
                distant sets, each set must have at least this fraction of
                the available nodes.  Defaults to 0.01 (1%).
            max_shuffle_count {int}: A limit on the number of times we'll
                shuffle the list and select a different anchor in an
                attempt to get a better split.  Defaults to 5.
            depth {int}: Depth of this node.  You do not need to set this
                yourself.
        Raises:
            ValueError: max_items_per_node must be at least 3.
        """

        self._metric = metric_function
        self._anchor = None
        self._local_items = None
        self._nearby_children = None
        self._distant_children = None
        self._threshold_distance = None
        self._max_items = max_items_per_node
        self._depth = depth
        self._max_shuffle_count = max_shuffle_count
        self._min_split_fraction = min_split_fraction
        self._max_depth = max_depth


        if max_items_per_node < 3:
            raise ValueError((
                f"max_items_per_node must be at least 3.  You specified "
                f"{max_items_per_node}."
            ))
        self.insert(items)


    def clear(self):
        """Empty out the vantage-point tree.

        This method clears the contents of the tree and returns it to an
        uninitialized state.

        No arguments.  Returns None.
        """
        self._anchor = None
        self._local_items = None
        self._nearby_children = None
        self._distant_children = None
        self._threshold_distance = None

    def find_items_within_radius(self,
                                 center: Indexable,
                                 radius: float,
                                 include_boundary: bool=True
                        ) -> List[Indexable]:
        """Find all items within a given distance of a center point

        This is one of the ways you can search the tree: given a ball defined
        by its center and radius, find all the items within that ball.  You can
        choose whether to include items that lie exactly on the ball's surface
        with the ``include_boundary`` argument.

        Arguments:
            center {indexable}: Item at the center of the search ball.  This
                does not need to be one of the items in the tree.
            radius {float}: How far out to look from the center.

        Keyword Arguments:
            include_boundary {bool}: If True (the default), items that lie
                exactly on the ball's surface will be included in the result.
                If False, they will not be included.

        Returns:
            All items inside the ball
        """

        # If we're just keeping items in our pocket, do the search and
        # be done with it
        if self._local_items is not None:
            return _items_within_distance(self._local_items,
                                          center, radius,
                                          self._metric,
                                          include_boundary)


        distance_to_center = self._metric(self._anchor, center)
        nearby_items = []

        # Is the anchor within the ball?
        if distance_to_center < radius or (
            distance_to_center == radius and include_boundary
            ):
            nearby_items.append(self._anchor)

        # Is the ball close enough that it could overlap our nearby-items
        # list?
        if distance_to_center <= (self._threshold_distance + radius):
            # Yes.  We need to search our nearby children.
            nearby_items.extend(self._nearby_children.find_items_within_radius(
                center, radius, include_boundary=include_boundary))

        # Is the ball close enough that it's included entirely within our
        # nearby-items subtree?
        if distance_to_center + radius < self._threshold_distance:
            # Yes; no distant items can be included.
            distant_items = []
        else:
            # We need to include the distant children.
            distant_items = self._distant_children.find_items_within_radius(
                center, radius, include_boundary=include_boundary
            )

        return nearby_items + distant_items


    def k_nearest_neighbors(self,
                            center: Indexable,
                            k: int) -> List[Indexable]:
        """Find the K nearest neighbors to a given item.

        The query item at the center of the search does not have to be one
        of the items in the tree.

        If the set of k nearest neighbors is not unique -- that is, if there
        are lots of items at the same distance, tied for the privilege of
        being the k'th neighbor -- then one will be chosen arbitrarily.

        Arguments:
            center {indexable}: Center point for search.  This does not have
                to be one of the items in the tree.  If it is, it will not be
                included as one of the neighbors.
            k {int}: How many neighbors to find

        Returns:
            List of K nearest neighbors, or list of all items in the tree if
            there are fewer than K
        """

        neighbors_with_distances = self._k_nearest_neighbors_recursive(
            center, k, math.inf)

        return [neighbor for (_, neighbor) in neighbors_with_distances]


    def insert(self, items: Sequence[Indexable]) -> None:
        """Add items to a vantage-point tree

        This function is the preferred way to add items to a tree.  By
        supplying lots of items at once, we can do a better job of balancing
        the tree for fast lookups later on.

        After this function returns, the items inserted will show up in
        searches executed on this tree with nearest_neighbor(),
        k_nearest_neighbors(), or points_within_ball().

        Note: This function can only be called on an unpopulated node.

        Arguments:
            items {sequence of indexable items}: Items to insert

        Returns:
            None.  Tree is modified in place.

        Raises:
            RuntimeError: Node is already populated.
        """

        if (self._anchor is not None
                or self._nearby_children is not None
                or self._distant_children is not None):
            raise RuntimeError((
                "Vantage point tree is already populated.  You "
                "can only call insert() on an empty tree."
            ))
        items = list(items)

        shuffle_count = 0

        # Bailout cases: if we don't have many items or we're already
        # too far down in the tree, just store the items locally.
        if (self._depth > self._max_depth
            or len(items) < self._max_items):
            self._local_items = items
            return


        # Try to get a good partition of nearby and distant items.
        nearby = []
        distant = []
        threshold_distance = []
        partition_ok = False

        while (shuffle_count < self._max_shuffle_count
            and not partition_ok):

            # Recursive case: Pick an item to be the anchor and find the
            # distance from the anchor to each item.  Sort that list by
            # distance and split it in half.  The nearer items and the
            # farther items both get their own subtrees.

            (nearby, distant, threshold_distance) = self._split_nearby_distant(items[0], items[1:])
            min_split_count = len(items) * self._min_split_fraction

            if len(nearby) < min_split_count or len(distant) < min_split_count:
                # Shuffle and try again.
                shuffle_count += 1
                LOG.debug(("Shuffling after (%d, %d) split at depth %d to "
                           "try to get a better partition. "),
                          len(nearby), len(distant), self._depth)
                random.shuffle(items)
            else:
                partition_ok = True

        if partition_ok:
            LOG.debug(
                ("Splitting node at depth %d with threshold distance %f.  "
                 "Nearby child will contain %d children.  Distant child "
                 "will contain %d children."),
                 self._depth, threshold_distance, len(nearby), len(distant))

            self._anchor = items.pop(0)
            self._threshold_distance = threshold_distance
            self._nearby_children = self._make_child(nearby)
            self._distant_children = self._make_child(distant)
        else:
            LOG.warning(
                ("Cannot split items.  Creating one very large VP-tree "
                "node with %d items.  This is not an error, but execution "
                "may be slow."),
                len(items))
            self._local_items = items
            return


    def _make_child(self, items: List[Indexable]) -> "VantagePointTree":
        """Make a child with the specified nodes

        Make a child node.  Copy all the properties of this node, increment
        the depth, and insert the listed items.

        Arguments:
            items {list of Indexable}: Items to insert

        Returns:
            New VantagePointTree node
        """

        return VantagePointTree(
            self._metric,
            items,
            max_items_per_node=self._max_items,
            max_depth=self._max_depth,
            depth=self._depth+1,
            min_split_fraction=self._min_split_fraction,
            max_shuffle_count=self._max_shuffle_count
        )

    def _k_nearest_neighbors_local(self,
                                   center: Indexable,
                                   k: int) -> List[DistanceWithIndexable]:
        """Search for the K nearest neighbors in just this node

        Arguments:
            center {Indexable}: Center point for search
            k {int}: How many neighbors to return

        Returns:
            List of up to k (distance, item) tuples, sorted by increasing
            distance from center
        """

        assert self._anchor is None
        assert self._nearby_children is None
        assert self._distant_children is None

        distance = lambda x: self._metric(center, x)

        exclude_center = [item for item in self._local_items if item != center]
        items_with_distances = sorted([(distance(item), item)
                                       for item in exclude_center])
        return items_with_distances[0:k]


    def _k_nearest_neighbors_recursive(self,
                                       center: Indexable,
                                       k: int,
                                       farthest_neighbor_distance: float
                                       ) -> List[DistanceWithIndexable]:
        """Search for K nearest neighbors in subtree

        Arguments:
            center {Indexable}: Center point for search
            k {int}: How many neighbors to find
            farthest_neighbor_distance {int}: Distance to the kth nearest
                neighbor found so far

        Returns:
            List of up to k (distance, item) pairs, sorted by increasing
            distance from center.  The center item will not be included
            in the result.
        """

        if self._local_items is not None:
           return self._k_nearest_neighbors_local(center, k)

        assert self._anchor is not None
        assert self._nearby_children is not None
        assert self._distant_children is not None
        assert self._local_items is None

        center_result = []
        center_anchor_distance = 0
        if center != self._anchor:
            center_anchor_distance = self._metric(center, self._anchor)
            if center_anchor_distance < farthest_neighbor_distance:
                center_result.append((center_anchor_distance, self._anchor))

        # Can any of the members of the nearby tree be closer than the
        # nearest neighbor so far?  That is, does the ball containing the
        # nearby children overlap the ball containing the K nearest neighbors
        # so far?
        if (self._threshold_distance + farthest_neighbor_distance
            >= center_anchor_distance):
            # Yes, they can.
            nearby_neighbors = self._nearby_children._k_nearest_neighbors_recursive(
                center, k, farthest_neighbor_distance)
        else:
            nearby_neighbors = []

        neighbors_so_far = _sorted_merge_keep_k(nearby_neighbors, center_result, k)

        if len(neighbors_so_far) == k:
            farthest_neighbor_distance = min(farthest_neighbor_distance,
                                             neighbors_so_far[-1][0])

        # We might not need to search the far-away items.  If we've already
        # found at least k neighbors and they're all within the nearby
        # subtree, there's no point in searching the far-away items.
        if (len(neighbors_so_far) < k or
            center_anchor_distance + farthest_neighbor_distance >=
            self._threshold_distance):

            # The farthest neighbor is outside our nearby shell, so
            # there's a chance we could yet find one closer than that.
            distant_results = self._distant_children._k_nearest_neighbors_recursive(
                center, k, farthest_neighbor_distance
                )
        else:
            distant_results = []

        final_neighbors = _sorted_merge_keep_k(neighbors_so_far,
                                               distant_results, k)

        return final_neighbors

    def nearest_neighbor(self, center: Indexable) -> Indexable:
        """Return the nearest neighbor to a query point

        The closest point that is not identical to the query point
        will be returned.  Identical points are those that compare
        equal under Python's ``==`` operator.

        If multiple points tie for the nearest neighbor, one will be
        selected arbitrarily.

        Arguments:
            center {indexable}: Query point

        Returns:
            Nearest point not identical to the query point, or
            None if all points in the tree are identical to the
            query point
        """

        result = self.k_nearest_neighbors(center, 1)
        if len(result) > 0:
            return result[0]
        return None

    def _split_nearby_distant(self,
                              anchor: Indexable,
                              items: List[Indexable]
                              )-> Tuple[List[Indexable], List[Indexable], float]:
        """Helper: Partition a list by distance from an anchor point

        The central operation in constructing a vantage-point tree is to
        partition a list of items into two halves: those close to and
        far away from a given anchor point.

        This function implements that operation.

        First we try the median distance to try to get a good split.
        If that fails, try the average.

        Arguments:
            anchor {indexable}: Central item whence we measure distances
            items {list of indexable}: Items to partition

        Returns:
            Tuple of (nearby_items, distant_items, threshold_distance).
            All items closer than threshold_distance to anchor are in the
            list of nearby items.  All items farther away than that are in
            the distant_items list.

        Note:
            If lots of items are at the threshold distance, we may wind
            up with an unbalanced tree.  Them's the breaks.  It may be
            possible to avoid that by shuffling your item list before
            creating the tree.
        """

        assert anchor is not None
        distances_with_items = [
            (self._metric(anchor, item), item)
            for item in items
        ]
        distances = [d for (d, _) in distances_with_items]

        median_distance = statistics.median(distances)
        mean_distance = sum(distances) / len(distances)

        # Split by median and by mean and choose whichever one gives
        # us the better split.
        median_nearby = [item for (distance, item) in distances_with_items
                         if distance <= median_distance]
        median_distant = [item for (distance, item) in distances_with_items
                          if distance > median_distance]

        mean_nearby = [item for (distance, item) in distances_with_items
                       if distance <= mean_distance]
        mean_distant = [item for (distance, item) in distances_with_items
                        if distance > mean_distance]

        # Which one has the better split?  Split ratio cannot be greater
        # than 1; the higher, the better.
        median_top = min(len(median_nearby), len(median_distant))
        median_bottom = max(len(median_nearby), len(median_distant))
        median_split_ratio = median_top / median_bottom

        mean_top = min(len(mean_nearby), len(mean_distant))
        mean_bottom = max(len(mean_nearby), len(mean_distant))
        mean_split_ratio = mean_top / mean_bottom

        if median_split_ratio > mean_split_ratio:
            return (median_nearby, median_distant, median_distance)
        return (mean_nearby, mean_distant, mean_distance)


    def print(self, indent: int=0) -> None:
        """Print a text representation of the tree

        Keyword Arguments:
            indent {int}: How much to indent each successive level of the tree
        """

        spaces = '  ' * indent
        if self._local_items is not None:
            print(f"{spaces} Leaf node ({len(self)} items): {self._local_items}")
        else:
            nearby_depth = self._nearby_children.depth()
            distant_depth = self._distant_children.depth()
            nearby_count = len(self._nearby_children)
            distant_count = len(self._distant_children)
            print(
                f"{spaces}Interior node:\n"
                f"  {spaces}near depth {nearby_depth}, nearby items {nearby_count},\n"
                f"  {spaces}far depth {distant_depth}, far_count {distant_count},\n"
                f"  {spaces}anchor {self._anchor}, "
                f"threshold distance {self._threshold_distance}")
            print(f"{spaces}Nearby child:")
            self._nearby_children.print(indent=indent+1)
            print(f"{spaces}Distant child:")
            self._distant_children.print(indent=indent+1)


    def depth(self):
        return self._depth


    def __len__(self):
        """Number of items contained in this tree"""

        if self._local_items is not None:
            return len(self._local_items)
        else:
            return 1 + len(self._nearby_children) + len(self._distant_children)

# End of class definition - helper functions below here


def _items_within_distance(items: Sequence[Indexable],
                           center: Indexable,
                           radius: float,
                           metric: MetricFunction,
                           include_boundary: bool
                           ) -> List[Indexable]:
    """Helper function -- filters a sequence for items within a ball

    Given a center and a radius, filter a sequence of items to keep
    only those inside a ball.

    Arguments:
        items {sequence of Indexable}: Items to filter
        center {Indexable}: Center of ball
        radius {float}: Radius of ball
        metric {MetricFunction}: Function to be used to compute distances
        include_boundary {bool}: Whether to keep items exactly on the
            ball's boundary

    Returns:
        List of items inside ball
    """

    if include_boundary:
        close_enough = lambda x: metric(center, x) <= radius
    else:
        close_enough = lambda x: metric(center, x) < radius

    return [
        item for item in items if close_enough(item)
    ]


def _sorted_merge_keep_k(items1: List[DistanceWithIndexable],
                         items2: List[DistanceWithIndexable],
                         keep_count: int) -> List[DistanceWithIndexable]:
    """Merge two sorted lists, keeping at most k items

    This is a sorted merge.  It does the moral equivalent of the
    following:

    ```
    merged = sorted(items1 + items2)
    return merged[0:k]
    ```

    However, it is smart enough to stop after k items instead of
    processing the entirety of both lists.

    Arguments:
        items1 {list of (distance, indexable) tuples}: First input
            to merge
        items2 {list of (distance, indexable) tuples}: Second input
            to merge
        keep_count {int}: How many items to keep in result

    Returns:
        List of up to keep_count (distance, indexable) tuples in sorted
        order
    """

    result = []
    items_remaining = keep_count
    while items_remaining > 0 and len(items1) > 0 and len(items2) > 0:
        distance1 = items1[0][0]
        distance2 = items2[0][0]
        if distance1 <= distance2:
            result.append(items1.pop(0))
        else:
            result.append(items2.pop(0))
        items_remaining -= 1

    # It's possible that we've exhausted one of the lists.  If so, just fill
    # the result list from whatever's left.
    if items_remaining > 0:
        if len(items1) > 0:
            result += items1[0:items_remaining]
        elif len(items2) > 0:
            result += items2[0:items_remaining]

    return result