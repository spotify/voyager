The ``voyager`` Python API
==========================

This module provides classes and functions for creating indices of vector data.

A quick example on how to get started::

   from voyager import Index, Space

   # Create an empty Index object that can store vectors:
   index = Index(Space.Euclidean, num_dimensions=5)
   id_a = index.add_item([1, 2, 3, 4, 5])
   id_b = index.add_item([6, 7, 8, 9, 10])

   print(id_a)  # => 0
   print(id_b)  # => 1

   # Find the two closest elements:
   neighbors, distances = index.query([1, 2, 3, 4, 5], k=2)
   print(neighbors)  # => [0 1]
   print(distances)  # => [  0. 125.]

   # Save the index to disk to reload later (or in Java!)
   index.save("output_file.voy")


For larger collections, generate vectors with NumPy and insert them in a batch::

   import numpy as np
   from voyager import Index, Space

   # Generate 1,000 synthetic vectors with 32 dimensions.
   rng = np.random.default_rng(seed=42)
   vectors = rng.random((1_000, 32), dtype=np.float32)

   # Build the index. IDs default to the vectors' row numbers.
   index = Index(Space.Euclidean, num_dimensions=32)
   index.add_items(vectors)

   # Find the five nearest neighbors of the first vector.
   neighbors, distances = index.query(vectors[0], k=5)
   print("Neighbor IDs:", neighbors)
   print("Squared Euclidean distances:", distances)

Example output::

   Neighbor IDs: [  0 549 866 545  29]
   Squared Euclidean distances: [0.        1.9799423 2.1809483 2.459416  2.4805002]

The first result is vector 0 itself, with distance 0.
Results may vary slightly across platforms or index builds.


.. autoclass:: voyager.Index
   :members:
   :inherited-members:
   :special-members: __contains__, __len__

Enums
-----
.. autoclass:: voyager.Space
   :members:
   :inherited-members:

.. autoclass:: voyager.StorageDataType
   :members:
   :inherited-members:

Utilities
---------
.. autoclass:: voyager.LabelSetView
   :members:
   :inherited-members:
