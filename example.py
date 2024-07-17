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

"""Example of using Metric DBSCAN on random strings

This is an end-to-end example.  We create four clusters of random strings
from overlapping alphabets, then cluster them with Metric DBSCAN and
print out the size of each cluster.

You will need metric_dbscan on your Python path in order to run this.
You can either do that by adding <current_directory>/src to the
environment variable PYTHONPATH or by running `pip install -e .` in
the directory containing this file.
"""

import random
import sys

try:
    import metric_dbscan
except ImportError:
    print((
        "You need the metric_dbscan package (this package) on your Python "
        "path in order to run this example.  You can achieve this by "
        "adding the directory containing this file (example.py) to your "
        "PYTHONPATH environment variable or by running 'pip install -e .'"
        "in the directory containing this file."
    ))
    sys.exit(1)

try:
    import Levenshtein
except ImportError:
    print((
        "This example depends on the 'python-Levenshtein' library to "
        "compute distances between strings.  Please install it with "
        "'pip install python-Levenshtein' or (if you're using Anaconda) "
        "'conda install python-levenshtein'."
    ))
    sys.exit(2)



def random_string(alphabet: str, length: int) -> str:
    """Return a string of characters chosen at random from a given alphabet.

    Arguments:
        alphabet (str}: Characters from which to construct the string
        length {int}: How many characters to put in the string

    Example:
       >>> my_word = random_string("abcde", 10)
       'eacebaeeed'

    Returns:
        String composed of characters chosen randomly (with replacement)
        from the given alphabet
    """
    characters = [random.choice(alphabet) for _ in range(length)]
    return ''.join(characters)


def main():
    # Create 4 clusters of 100 random strings each with partially
    # overlapping alphabets
    cluster1 = [random_string("abcdeAB", 20) for _ in range(100)]
    cluster2 = [random_string("fghijAB", 20) for _ in range(100)]
    cluster3 = [random_string("klmnoAB", 20) for _ in range(100)]
    cluster4 = [random_string("pqrstAB", 20) for _ in range(100)]

    # Glue them into one long list for metric_dbscan
    all_strings = cluster1 + cluster2 + cluster3 + cluster4

    # Compute an integer cluster label for each string using Levenshtein
    # edit distance as our metric.  We happen to know (because we've played
    # with it) that a neighbor distance threshold of XXX will give us
    # a bunch of clusters.  We encourage you to play with that threshold to see
    # what happens to the number of clusters.
    cluster_ids = metric_dbscan.cluster_items(all_strings,
                                              Levenshtein.distance,
                                              5, # minimum cluster size
                                              12)

    # Turn the list of cluster IDs into lists of the items assigned to
    # each cluster
    cluster_contents = {}
    for (item_id, cluster_id) in enumerate(cluster_ids):
        if cluster_id not in cluster_contents:
            cluster_contents[cluster_id] = []
        cluster_contents[cluster_id].append(item_id)

    print("Cluster sizes for random strings:")
    for (cluster_id, contents) in sorted(cluster_contents.items()):
        print(f"Cluster {cluster_id}: {len(contents)} items")
        first_members = [all_strings[i] for i in contents[0:5]]
        print(f"\tFirst 5 members: {first_members}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

