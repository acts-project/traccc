# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

traccc_add_executable(simulate_telescope "simulate_telescope.cpp"
    LINK_LIBRARIES vecmem::core traccc::io traccc::core
    traccc::options detray::core detray::utils covfie::core 
    Boost::filesystem)

traccc_add_executable(simulate_toy_detector "simulate_toy_detector.cpp"
    LINK_LIBRARIES vecmem::core traccc::io traccc::core
    traccc::options detray::core detray::utils covfie::core 
    Boost::filesystem)