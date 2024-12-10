#Detray library, part of the ACTS project(R& D line)
#
#(c) 2024 CERN for the benefit of the ACTS project
#
#Mozilla Public License Version 2.0

message(STATUS "Building detray device command line tools")

# Enable CUDA as a language.
enable_language(CUDA)

include(CMakeFindDependencyMacro)

find_dependency(Boost COMPONENTS program_options REQUIRED)

# Build the detector validation executable.
detray_add_executable(detector_validation_cuda
                      "detector_validation_cuda.cpp"
                      LINK_LIBRARIES GTest::gtest GTest::gtest_main
                      Boost::program_options detray::test_cuda detray::tools
)

# Build the material cuda validation executable.
detray_add_executable(material_validation_cuda
                      "material_validation_cuda.cpp"
                      LINK_LIBRARIES GTest::gtest GTest::gtest_main
                      Boost::program_options detray::test_cuda detray::tools
)
