#Detray library, part of the ACTS project(R& D line)
#
#(c) 2023-2024 CERN for the benefit of the ACTS project
#
#Mozilla Public License Version 2.0

message(STATUS "Building detray host command line tools")

include(CMakeFindDependencyMacro)

find_dependency(Boost COMPONENTS program_options)

# Generate test detectors
detray_add_executable(generate_toy_detector
                      "generate_toy_detector.cpp"
                      LINK_LIBRARIES Boost::program_options detray::tools
                      detray::io detray::test_utils detray::core_array
)

detray_add_executable(generate_wire_chamber
                      "generate_wire_chamber.cpp"
                      LINK_LIBRARIES Boost::program_options detray::tools
                      detray::io detray::test_utils detray::core_array
)

detray_add_executable(generate_telescope_detector
                      "generate_telescope_detector.cpp"
                      LINK_LIBRARIES Boost::program_options detray::tools
                      detray::io detray::test_utils detray::core_array
)

# Build the visualization executable.
detray_add_executable(detector_display
                      "detector_display.cpp"
                      LINK_LIBRARIES Boost::program_options detray::tools
                      detray::svgtools
)

# Build the detector validation executable.
detray_add_executable(detector_validation
                      "detector_validation.cpp"
                      LINK_LIBRARIES GTest::gtest GTest::gtest_main
                      Boost::program_options detray::tools detray::test_utils
                      detray::svgtools
)

# Build the material validation executable.
detray_add_executable(material_validation
                      "material_validation.cpp"
                      LINK_LIBRARIES GTest::gtest GTest::gtest_main
                      Boost::program_options detray::tools detray::test_utils
                      detray::svgtools
)
