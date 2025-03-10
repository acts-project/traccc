# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

traccc_add_executable( benchmark_cuda
   "toy_detector_cuda.cpp"
   LINK_LIBRARIES benchmark::benchmark
   vecmem::core vecmem::cuda detray::detectors
   traccc::core traccc::device_common
   traccc::cuda traccc_benchmarks_common )
