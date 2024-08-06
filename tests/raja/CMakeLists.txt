# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0
find_package( raja REQUIRED PATHS /home/xola/raja/install)
traccc_add_test( raja
   raja_example.cpp
   LINK_LIBRARIES
   GTest::gtest
   RAJA
)
