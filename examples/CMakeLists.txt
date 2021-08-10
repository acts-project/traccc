add_subdirectory(cpu)

find_package(OpenMP COMPONENTS CXX)
if (OpenMP_CXX_FOUND)
    add_subdirectory(openmp)
    message(STATUS "OpenMP found! Include /examples/openmp subdirectory")
endif()
