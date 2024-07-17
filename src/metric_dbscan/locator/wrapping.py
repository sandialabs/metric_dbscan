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

### This file contains helper functions for adding item IDs
### and reading them back out.  This lets you operate on indices
### instead of bare items -- useful when you're maintaining parallel
### lists of IDs instead of passing around the objects themselves.


from metric_dbscan.dbscan_types import (
    ClusterableItem, DistanceFunction, ItemWithId
)

from typing import List

def wrap_distance_function(dist: DistanceFunction) -> DistanceFunction:
    """Helper: create a distance function that operates on (item, id)

    It is often helpful to get back a list of item IDs instead of the
    item objects themselves.  This function will take in your favorite
    distance function that poerates on items and return a new function
    that operates on (item, id) tuples.

    Arguments:
        dist (DistanceFunction): Distance function that operates on
            item objects

    Returns:
        New function that operates on (item, id) tuples by calling
        `dist` on the underlying items
    """

    def wrapped_distance(x: ItemWithId, y: ItemWithId) -> float:
        return dist(x.item, y.item)
    return wrapped_distance

def add_item_ids(items: List[ClusterableItem]) -> List[ItemWithId]:
    """Add an integer ID to every item in a list

    Another helper function.  Use this in conjunction with
    wrap_distance_function() if you want to work with item IDs
    instead of the raw items.

    Generated IDs will range from 0 to len(items)-1 inclusive.

    Arguments:
        items (list of ClusterableItem): items to label with IDs.
            You may get strange results if you pass in an arbitrary
            iterable instead of a list.

    Returns:
        List of (item, id) tuples
    """

    return [
        ItemWithId(item=item, id=i) for (i, item) in enumerate(items)
    ]


def item_id(item_with_id: ItemWithId) -> ClusterableItem:
    """Extract an ID from a labeled item

    This is the inverse of add_item_ids.  Given a labeled item,
    it returns that item's integer ID.

    Arguments:
        item_with_id (ItemWithId): item plus ID

    Returns:
        Integer ID from item
    """

    return item_with_id.id