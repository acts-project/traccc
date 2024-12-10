.. _installation:

Installation
============

The library part of covfie is a header-only library, leaving the user with many
ways to install it -- including simply copying the headers to a location
accessible by the dependent project. However, the recommended way to install
(and use) covfie is through CMake. In order to install it, obtain the source
code from Github:

.. code-block:: console

    $ git clone git@github.com:acts-project/covfie.git

Alternatively, if you would like to obtain a specific version of the library,
you can specify this desire as follows:

.. code-block:: console

    $ git clone git@github.com:acts-project/covfie.git --branch v0.1.0

It is also possible to obtain the source code through other means, of course,
but we will not cover those methods here.

You will need to determine where you want to install covfie. In the following
instructions, the installation path will be denoted :code:`[prefix]`. To
install covfie, proceed with the following commands:

.. code-block:: console

    $ cmake -S covfie -B covfie_build -DCMAKE_INSTALL_PREFIX=[prefix] -DCOVFIE_PLATFORM_CPU=On -DCOVFIE_PLATFORM_CUDA=On
    $ cmake --build build
    $ cmake --install build

Configuration flags
-------------------

If all went according to plan, covfie is now installed! Please note that we
only install code for platforms that the user requests. The following flags can
be passed to CMake (through the :code:`-D[FLAG]=On` flags):

:code:`COVFIE_PLATFORM_CPU`
    Install the relevant headers for the CPU-specific parts of the library
    (enabled by default)

:code:`COVFIE_PLATFORM_CUDA`
    Install the relevant headers for the CUDA-specific parts of the library
    (disabled by default)
