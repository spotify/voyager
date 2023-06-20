:og:title: Voyager Documentation
:og:description: 🛰️ Documentation for Voyager: A nearest-neighbor search library.

.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

|br|

.. image:: voyager-black-roundrect-rectangle.png
  :width: 700
  :alt: The word "Voyager" in white on a black background.

**Voyager** is a library for performing fast approximate nearest-neighbor searches on an in-memory collection of vectors.

Voyager features bindings to both Python and Java, with feature parity and index compatibility between both languages.
It uses the HNSW algorithm, based on `the open-source hnswlib package <https://github.com/nmslib/hnswlib>`_,
with numerous features added for convenience and speed. Voyager is used extensively in production at Spotify,
and is queried hundreds of millions of times per day to power numerous user-facing features.

Think of Voyager like `Sparkey <https://github.com/spotify/sparkey>`_, but for vector/embedding data;
or like `Annoy <https://github.com/spotify/annoy>`_, but with `much higher recall <https://ann-benchmarks.com>`_.
It got its name because it searches through (embedding) space(s), much like
`the Voyager interstellar probes <https://en.wikipedia.org/wiki/Voyager_program>`_ launched by NASA in 1977.


.. toctree::
   :maxdepth: 1

   Python API Reference <reference>

.. toctree::
   Java Documentation </voyager/java#http://>
   GitHub Repo <http://github.com/spotify/voyager>
