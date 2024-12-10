Integration
===========

The covfie installer automatically installs the necessary files to include it
in a downstream project, which should make it relatively painless to use. In
the CMake configuration of a downstream project, simply write:

.. code-block:: cmake

    find_package(covfie)

If the availability of covfie is critical for the rest of your build system
(what an honour!), you can enforce this as follows:

.. code-block:: cmake

    find_package(covfie REQUIRED)

Following this directive, the covfie targets should be available for use. The
target for the core library is called :code:`covfie::core`, while the
platform-specific targets are found under the names :code:`covfie::cpu` and
:code:`covfie::cuda`. As an example of a complete CMake configuration that
includes covfie, consider the following:

.. code-block:: cmake

    project("my_application")
    find_package(covfie REQUIRED)
    add_executable(main main.cpp)
    target_link_libraries(main PUBLIC covfie::covfie_core)

In order for CMake to be able to find the necessary setup files, it must be
instructed to look in the installation prefix chosen during :ref:`installation
<installation>`, unless this is a standard prefix. To do so, add the following
flag to the configuration of the *downstream* project:

.. code-block:: console

    $ cmake ... -DCMAKE_PREFIX_PATH=[path]

The CMake setup exposes some of the flags which were used to installed covfie
in the first place, which allows user to make certain decision based on the way
in which covfie was installed. Variables set are as follows:

:code:`COVFIE_PLATFORM_CPU`
    True iff the CPU-specific headers were installed.

:code:`COVFIE_PLATFORM_CUDA`
    True iff the CUDA-specific headers were installed.

:code:`COVFIE_QUIET`
    Silences warnings about missing compiler features.
