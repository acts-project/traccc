# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/slobod/Documents/Work/traccc/build/_deps/dfelibs-src"
  "/home/slobod/Documents/Work/traccc/build/_deps/dfelibs-build"
  "/home/slobod/Documents/Work/traccc/build/_deps/dfelibs-subbuild/dfelibs-populate-prefix"
  "/home/slobod/Documents/Work/traccc/build/_deps/dfelibs-subbuild/dfelibs-populate-prefix/tmp"
  "/home/slobod/Documents/Work/traccc/build/_deps/dfelibs-subbuild/dfelibs-populate-prefix/src/dfelibs-populate-stamp"
  "/home/slobod/Documents/Work/traccc/build/_deps/dfelibs-subbuild/dfelibs-populate-prefix/src"
  "/home/slobod/Documents/Work/traccc/build/_deps/dfelibs-subbuild/dfelibs-populate-prefix/src/dfelibs-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/slobod/Documents/Work/traccc/build/_deps/dfelibs-subbuild/dfelibs-populate-prefix/src/dfelibs-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/slobod/Documents/Work/traccc/build/_deps/dfelibs-subbuild/dfelibs-populate-prefix/src/dfelibs-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
