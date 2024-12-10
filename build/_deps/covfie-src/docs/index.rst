covfie: vector fields made easy
===============================

covfie (pronounced *coffee*) is a **co**-processor **v**\ ector **fie**\ ld
library. This header-only C++ library is designed to make the use of vector
fields as easy as possible, both on traditional CPUs as well as a variety of
co-processors. covfie is provided in the hope that it is useful for scientists
in the domain of high-energy physics, but also other fields where vector fields
are common, such as atmospheric sciences.

.. figure:: static/bfield.png
   :figwidth: 40 %
   :alt: map to buried treasure
   :align: right

   The `ATLAS <https://atlas.cern/>`_ magnetic field, rendered entirely on a GPU with *covfie*.

The covfie documentation consists of three major components. The first is the
user guide, which establishes many of the core ideas behind covfie and details
-- in a global sense -- how it can be used in applications. The second
component details the benchmarking part of covfie, which is a major part of the
software. The third component is the API reference, which describes in depth
the different types and methods that covfie exposes.

User guide
----------

The user guide is a mostly prosaic introduction to the covfie library and its
use. The user guide describes the main design goals of the library, as well as
a step by step introduction to using it. If you are an end user and you are
looking to use covfie in a wider project, this is for you.

.. toctree::
   :maxdepth: 3

   user/design
   user/installation
   user/integration
   user/quickstart
   user/backend
   user/types
   user/ownership
   user/conversion
   user/io

Developer guide
---------------

The developer guide is a mostly prosaic summary of the internal workings of
covfie, which should be useful to users who want to extend covfie with their
own backends.

.. toctree::
   :maxdepth: 3

   developer/build
   developer/backend
   developer/concepts
   developer/contributing

Benchmarking guide
------------------

In addition to being a header-only library, covfie is also -- in part -- a
benchmark. The modular design of covfie makes it easy to rapidly and thoroughly
test all kinds of new representations and storage strategies.

API reference
-------------

Coming soon...
