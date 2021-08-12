

if (TRACCC_ARRAY_PLUGIN)
    add_subdirectory(array)
endif()

if (TRACCC_EIGEN_PLUGIN)
    add_subdirectory(eigen)
endif()

if (TRACCC_SMATRIX_PLUGIN)
    add_subdirectory(smatrix)
endif()

if (TRACCC_VC_ALGEBRA_PLUGIN)
    add_subdirectory(vc)
endif()
