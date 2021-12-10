# We need the SYCL language in this directory.
enable_language( SYCL )

add_library(traccc_sycl SHARED 
# header files
include/sycl/seeding/detail/doublet_counter.hpp
include/sycl/seeding/detail/triplet_counter.hpp
include/sycl/seeding/detail/multiplet_estimator.hpp
include/sycl/seeding/detail/sycl_helper.hpp
include/sycl/seeding/doublet_counting.hpp
include/sycl/seeding/doublet_finding.hpp
include/sycl/seeding/seed_finding.hpp
include/sycl/seeding/seed_selecting.hpp
include/sycl/seeding/triplet_counting.hpp
include/sycl/seeding/triplet_finding.hpp
include/sycl/seeding/weight_updating.hpp
# implementation files
src/seeding/doublet_counting.sycl
src/seeding/doublet_finding.sycl
src/seeding/seed_selecting.sycl
src/seeding/triplet_counting.sycl
src/seeding/triplet_finding.sycl
src/seeding/weight_updating.sycl)

# target_compile_options(
# 	traccc_sycl
# 	PRIVATE -fsycl-targets=nvptx64-nvidia-cuda)

target_include_directories(traccc_sycl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> )
	
target_include_directories(traccc_sycl PRIVATE "/atlas/software/cuda/11.4.1/x86_64-ubuntu1804/include")
target_link_libraries(traccc_sycl PUBLIC traccc::core vecmem::sycl)

add_library(traccc::sycl ALIAS traccc_sycl)
