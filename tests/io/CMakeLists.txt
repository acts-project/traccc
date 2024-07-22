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

target_compile_definitions(
  traccc_test_io
  PRIVATE
  TRACCC_TEST_IO_MOCK_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/mock_data"
)
