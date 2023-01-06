# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Declare the io library test(s).
traccc_add_test( io 
   "test_binary.cpp" 
   "test_csv.cpp" 
   "test_mapper.cpp" 
   "test_event_map.cpp"
   LINK_LIBRARIES GTest::gtest_main traccc_tests_common
                  traccc::core traccc::io )
