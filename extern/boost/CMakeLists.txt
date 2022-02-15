# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.14 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building Boost as part of the TRACCC project" )

# Declare where to get Boost from.
set( TRACCC_BOOST_SOURCE
   "GIT_REPOSITORY;https://github.com/boostorg/boost.git;GIT_TAG;boost-1.75.0"
   CACHE STRING "Source for Boost, when built as part of this project" )
mark_as_advanced( TRACCC_BOOST_SOURCE )
FetchContent_Declare( Boost ${TRACCC_BOOST_SOURCE} )

# Get it into the current directory.
FetchContent_GetProperties(Boost)
FetchContent_Populate(Boost)

# Set up an alias target that mimicks the imported target.
message(${BOOST_LIBRARIES})
add_library( program_options ALIAS Boost::program_options )