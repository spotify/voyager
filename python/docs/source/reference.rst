The ``voyager`` Python API
==========================

This module provides classes and functions for creating indices of vector data.

A quick example on how to get started::

   import numpy as np
   from voyager import Index, Space

   # Create an empty Index object that can store vectors:
   index = Index(Space.Euclidean, num_dimensions=5)
   id_a = index.add_item([1, 2, 3, 4, 5])
   id_b = index.add_item([6, 7, 8, 9, 10])

   print(id_a)  # => 0
   print(id_b)  # => 1

   # Find the two closest elements:
   neighbors, distances = index.query([1, 2, 3, 4, 5], k=2) 
   print(neighbors)  # => [0, 1]
   print(distances)  # => [0.0, 125.0]

   # Save the index to disk to reload later (or in Java!)
   index.save("output_file.voy")


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
