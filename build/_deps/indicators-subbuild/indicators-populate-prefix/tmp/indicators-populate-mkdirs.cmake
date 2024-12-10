# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/slobod/Documents/Work/traccc/build/_deps/indicators-src"
  "/home/slobod/Documents/Work/traccc/build/_deps/indicators-build"
  "/home/slobod/Documents/Work/traccc/build/_deps/indicators-subbuild/indicators-populate-prefix"
  "/home/slobod/Documents/Work/traccc/build/_deps/indicators-subbuild/indicators-populate-prefix/tmp"
  "/home/slobod/Documents/Work/traccc/build/_deps/indicators-subbuild/indicators-populate-prefix/src/indicators-populate-stamp"
  "/home/slobod/Documents/Work/traccc/build/_deps/indicators-subbuild/indicators-populate-prefix/src"
  "/home/slobod/Documents/Work/traccc/build/_deps/indicators-subbuild/indicators-populate-prefix/src/indicators-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/slobod/Documents/Work/traccc/build/_deps/indicators-subbuild/indicators-populate-prefix/src/indicators-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/slobod/Documents/Work/traccc/build/_deps/indicators-subbuild/indicators-populate-prefix/src/indicators-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
