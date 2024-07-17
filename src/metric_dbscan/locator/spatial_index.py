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

"""Superclass defining interface for spatial index used in DBSCAN
"""

import abc

from metric_dbscan.dbscan_types import ClusterableItem, DistanceFunction

from typing import List, Optional, Sequence

class SpatialIndex(abc.ABC):
    """Abstract superclass for spatial indices

    Properties:
        distance (DistanceFunction): Metric function to use for computing
            distance between item

    Methods:
        insert(items): Add one or more items to the locator
        find_items_within_radius(center, radius): Find all items
            in neighborhood
        clear(): Clear out list of itemss for reinitialization

    Treat a locator as immutable once it contains a set of items.  That is,
    once you've called `insert()` or initialized it with a list of items,
    no new items can be added.

    """

    def __init__(self,
                 distance: DistanceFunction,
                 items: Optional[Sequence[ClusterableItem]] = None):
        self.distance = distance


    @abc.abstractmethod
    def insert(self, items: Sequence[ClusterableItem]) -> None:
        """Add a list of items to the locator.

        Arguments:
            items (sequence of ClusterableItem): Items to add
        """
        ...


    @abc.abstractmethod
    def find_items_within_radius(self,
                                 center: ClusterableItem,
                                 radius: float,
                                 include_boundary: bool=True) -> List[int]:
        """Find all the items within a specified radius of a query item.

        This is the locator's main function.  This is where you
        should use whatever acceleration you have in mind.

        Arguments:
            center (ClusterableItem): Item whose neighborhood you
                want.  This does not have to be one of the items in the
                locator -- it can be any item.
            radius (float): How far out to search from the query item.
                The search volume will be a sphere centered on the query
                item.

        Keyword Arguments:
            include_boundary (bool): If True (the default), items at exactly
                the query radius will be included in the results.  In other
                words, the query volume is a sphere including its boundary.

        Raises:
            AttributeError: No distance function was set on the locator.

        Returns:
            List of items in neighborhood
        """
        ...


    @abc.abstractmethod
    def clear(self) -> None:
        """Clear out the locator

        This method removes all items from the locator but keeps
        the distance function with which it was initialized.

        No arguments.  Returns None.
        """
        ...
