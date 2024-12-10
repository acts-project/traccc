# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/slobod/Documents/Work/traccc/build/_deps/vecmem-src"
  "/home/slobod/Documents/Work/traccc/build/_deps/vecmem-build"
  "/home/slobod/Documents/Work/traccc/build/_deps/vecmem-subbuild/vecmem-populate-prefix"
  "/home/slobod/Documents/Work/traccc/build/_deps/vecmem-subbuild/vecmem-populate-prefix/tmp"
  "/home/slobod/Documents/Work/traccc/build/_deps/vecmem-subbuild/vecmem-populate-prefix/src/vecmem-populate-stamp"
  "/home/slobod/Documents/Work/traccc/build/_deps/vecmem-subbuild/vecmem-populate-prefix/src"
  "/home/slobod/Documents/Work/traccc/build/_deps/vecmem-subbuild/vecmem-populate-prefix/src/vecmem-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/slobod/Documents/Work/traccc/build/_deps/vecmem-subbuild/vecmem-populate-prefix/src/vecmem-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/slobod/Documents/Work/traccc/build/_deps/vecmem-subbuild/vecmem-populate-prefix/src/vecmem-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
