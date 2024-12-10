# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/slobod/Documents/Work/traccc/build/_deps/covfie-src"
  "/home/slobod/Documents/Work/traccc/build/_deps/covfie-build"
  "/home/slobod/Documents/Work/traccc/build/_deps/covfie-subbuild/covfie-populate-prefix"
  "/home/slobod/Documents/Work/traccc/build/_deps/covfie-subbuild/covfie-populate-prefix/tmp"
  "/home/slobod/Documents/Work/traccc/build/_deps/covfie-subbuild/covfie-populate-prefix/src/covfie-populate-stamp"
  "/home/slobod/Documents/Work/traccc/build/_deps/covfie-subbuild/covfie-populate-prefix/src"
  "/home/slobod/Documents/Work/traccc/build/_deps/covfie-subbuild/covfie-populate-prefix/src/covfie-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/slobod/Documents/Work/traccc/build/_deps/covfie-subbuild/covfie-populate-prefix/src/covfie-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/slobod/Documents/Work/traccc/build/_deps/covfie-subbuild/covfie-populate-prefix/src/covfie-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
