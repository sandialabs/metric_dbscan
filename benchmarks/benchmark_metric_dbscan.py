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

# This file runs Metric DBSCAN on a collection of random strings.  We
# note the memory usage and time before we start so that we can get a
# fair sense for how much memory the code is using.

import collections
import itertools
import random
import sys
import time

import Levenshtein
import psutil

import metric_dbscan
import metric_dbscan.dbscan

from typing import List

def random_string(alphabet: str, length: int) -> str:
    characters = [random.choice(alphabet) for _ in range(length)]
    return ''.join(characters)

def random_strings(alphabet: str, length: int, how_many: int) -> List[str]:
    return [random_string(alphabet, length) for _ in range(how_many)]

def replace_first_letter(word: str, new_first_letter: str) -> str:
    return new_first_letter + word[1:]

def string_clusters(alphabets: List[str], word_length: int, cluster_size: int, mark_clusters: bool=True) -> List[str]:
    cluster_markers = "!@#$%^&*()"

    clusters = []

    for (cluster_id, alphabet) in enumerate(alphabets):
        cluster = random_strings(alphabet, word_length, cluster_size)
        if mark_clusters:
            cluster = [replace_first_letter(word, cluster_markers[cluster_id]) for word in cluster]
        clusters.append(cluster)

    return clusters


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} num_strings")
        print("\tRuns metric DBSCAN on the specified number of strings and ")
        print("\tprints memory and runtime information.")
        return 1

    cluster_size = int(sys.argv[1])
    print((
        f"INFO: Generating 4 clusters of strings with {cluster_size} "
        f"members each."
    ))

    alphabets = [
        "abcdeABC",
        "fghijABC",
        "klmnoABC",
        "pqrstABC"
    ]
    string_data = string_clusters(alphabets, 80, cluster_size)

    start_memory_usage = psutil.Process().memory_full_info().uss
    start_time = time.time()

    all_strings = list(itertools.chain(*string_data))

    max_neighbor_distance = 25
    min_cluster_size = 5
    cluster_labels = metric_dbscan.cluster_items(all_strings,
                                                 Levenshtein.distance,
                                                 min_cluster_size,
                                                 max_neighbor_distance)

    cluster_label_counts = collections.Counter(cluster_labels)
    end_memory_usage = metric_dbscan.dbscan.SAVED_USS
    end_time = time.time()

    total_time = end_time - start_time
    memory_increase = end_memory_usage - start_memory_usage

    print("Cluster sizes: {}".format(cluster_label_counts))
    print("Net execution time: {} seconds".format(total_time))
    print("Net memory usage: {} kilobytes".format(memory_increase / 1024))

    return 0

if __name__ == '__main__':
    sys.exit(main())


