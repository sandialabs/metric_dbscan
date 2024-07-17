# Metric DBSCAN

This repository contains an implementation of the DBSCAN clustering algorithm that works in general metric spaces.

## What's DBSCAN?

DBSCAN is a non-parametric density-based clustering algorithm.  Clustering algorithms take collections of objects and assign them to groups.  A density-based clustering algorithm looks for areas where objects are crowded together and calls that a cluster.  A non-parametric clustering algorithm does not require that you tell it in advance how many clusters to look for.

DBSCAN was first described in "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise" by Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Wu.  It was published in the proceedings of KDD 1996.  It is one of the most popular and widely used clustering algorithms: in 2014, it was awarded the Test of Time award.

## What's a metric space?

A metric space is one where the distance between two points `x` and `y` can be measured by a *metric function* `d`.  This function must have the following properties:

- The distance from a point to itself is zero: `d(x, x) = 0` for all `x`.
- The distance between two distinct points is always positive: if `x != y`, then `d(x, y) > 0`.
- The distance from `x` to `y` is always the same as the distance from `y` to `x`: `d(x, y) = d(y, x)`.
- The triangle inequality holds: `d(x, y) <= d(x, z) + d(y, z)`

Ordinary Euclidean space is the most familiar metric space.  However, Euclidean space is also a *vector space*: given two distinct points `x` and `y`, vector operations like addition, subtraction, inner product, and outer product are all defined.

There are metric spaces that are not vector spaces.  Words (strings of characters) are one such.  We can define a metric function such as [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) that meets all four properties above, but there is no sensible way to add or subtract two words, let alone take their inner product.

Most DBSCAN implementations out there assume that the objects you want to cluster live in a vector space.  That's why we wrote this package.

## How do I install this package?

The easiest way is to use `pip`:

```
pip install metric_dbscan
```

If you want to install from source, download or clone this repository, install [Poetry](https://python-poetry.org), and then run `poetry install` from the directory containing this README and `pyproject.toml`.

## How do I use it?

The function you want is `metric_dbscan.cluster_items()`.  It takes the following arguments:

1. A list of items to cluster.

2. A distance function.  This function must take two arguments (each of which will be one of the items to cluster) and return a non-negative distance.  It must obey all of the properties of a metric listed above.

3. A minimum cluster size.  You get to choose this -- what's the smallest group of objects
that you would consider a cluster?  It must be at least two.

4. A maximum neighbor distance.  Objects closer together than this distance are neighbors and can belong to the same cluster.  Objects further apart are not neighbors and belong to different clusters.

You'll get back a list of integers with the same length as the list of items.  Each entry in this list is the cluster ID for the corresponding item.  A cluster ID of -1, also known as `metric_dbscan.OUTLIER`, indicates that the corresponding item is not part of any cluster.

You can find an example in the file `example.py` at the top of this repository.

## Can I do this in scikit-learn?

Yes, but it's expensive.  The [DBSCAN implementation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) in scikit-learn accepts a precomputed distance matrix as one of its arguments.  You could fill out that matrix using your distance function of choice and pass it in.  The expensive part here is that you will have to evaluate your distance function once for every pair of items, which is DBSCAN's worst-case behavior.  We use a [vantage-point tree](https://en.wikipedia.org/wiki/Vantage-point_tree) to avoid that by speeding up neighborhood queries.

Side note: if you browse Stack Overflow or look in the scikit-learn FAQ, you'll see suggestions to just pass in your own metric function despite what the documentation for DBSCAN says.  This currently works because the library doesn't enforce some of the requirements in the documentation, but it's very slow.

## Who's the point of contact?

Send email to Andy Wilson (atwilso at sandia dot gov) with questions, comments, suggestions, and complaints, or open up an issue on the Github repository.


