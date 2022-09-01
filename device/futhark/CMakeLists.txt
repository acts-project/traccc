# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# The C language is not enabled in our project by default, but we need it to
# compile Futhark's code.
enable_language(C)

# Try to find our dependencies, which are of course Futhark itself (required)
# and the CUDA toolkit for compiling CUDA projects. Also notify the user of
# what we find.
find_program(Futhark futhark REQUIRED)
find_package(CUDAToolkit)

message(STATUS "Found Futhark compiler at ${Futhark}")

# This function describes how to turn a Futhark source into a CMake library.
# The function creates a target called `TARGET_NAME` from a source file
# `_SOURCE_NAME`. The Futhark target platform (CUDA or C) is given
# (case-insensitively) by `_LANGUAGE_TARGET`
function(add_futhark_library TARGET_NAME _LANGUAGE_TARGET _SOURCE_NAME)
    # Set a few useful variables.
    set(FUTHARK_LANGUAGES c cuda)
    set(SOURCE_NAME ${CMAKE_CURRENT_SOURCE_DIR}/${_SOURCE_NAME})
    string(TOLOWER ${_LANGUAGE_TARGET} LANGUAGE_TARGET)
    string(TOUPPER ${_LANGUAGE_TARGET} LANGUAGE_TARGET_FANCY)

    # Notify the user of what we are doing.
    message(STATUS "Building ${LANGUAGE_TARGET_FANCY} Futhark target ${TARGET_NAME} from ${SOURCE_NAME}")

    # Make sure that the source file actually exists.
    if(NOT EXISTS ${SOURCE_NAME})
        message(FATAL_ERROR "File ${SOURCE_NAME} does not exist!")
    endif()

    # Ensure that the requested language is actually a supported one, namely C
    # or CUDA.
    list(FIND FUTHARK_LANGUAGES ${LANGUAGE_TARGET} index)
    if(index EQUAL -1)
        message(FATAL_ERROR "Futhark language must be one of ${FUTHARK_LANGUAGES}, not ${LANGUAGE_TARGET}")
    endif()

    # Define some needed directory and file names.
    get_filename_component(FILE_ROOT ${SOURCE_NAME} NAME_WE)
    set(FUTHARK_OUTPUT_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src)
    set(FUTHARK_OUTPUT_HEADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/include/traccc/futhark)
    set(FUTHARK_OUTPUT_SOURCE ${FUTHARK_OUTPUT_SOURCE_DIR}/${FILE_ROOT}.c)
    set(FUTHARK_OUTPUT_HEADER ${FUTHARK_OUTPUT_HEADER_DIR}/${FILE_ROOT}.h)

    # Create the destination directories.
    file(MAKE_DIRECTORY ${FUTHARK_OUTPUT_SOURCE_DIR})
    file(MAKE_DIRECTORY ${FUTHARK_OUTPUT_HEADER_DIR})

    # Define the custom command for generating .c and .h files from .fut files.
    add_custom_command(
        # Generate two files, a source file and a header file.
        OUTPUT ${FUTHARK_OUTPUT_SOURCE} ${FUTHARK_OUTPUT_HEADER}

        # Invoke the Futhark compiler in library mode.
        COMMAND ${Futhark}
        ARGS ${LANGUAGE_TARGET} --library -o ${FILE_ROOT} ${SOURCE_NAME}

        # Move the generated source and header file to their destinations.
        COMMAND mv
        ARGS ${FILE_ROOT}.c ${FUTHARK_OUTPUT_SOURCE}
        COMMAND mv
        ARGS ${FILE_ROOT}.h ${FUTHARK_OUTPUT_HEADER}

        # Register the generated JSON interface file as a byproduct.
        BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/${FILE_ROOT}.json

        # Mark the Futhark file as the main dependency.
        MAIN_DEPENDENCY ${SOURCE_NAME}

        # Execute this from the binary directory, to ensure that no dirty files
        # are generated in the user's work directory.
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}

        # Notify the user that we are generating code from this Futhark file.
        COMMENT "Generating ${LANGUAGE_TARGET_FANCY} file ${FILE_ROOT}.c from ${SOURCE_NAME}"
        VERBATIM
    )

    # With the output source now generated, we can create a new library from
    # it.
    add_library(
        ${TARGET_NAME}
        ${FUTHARK_OUTPUT_SOURCE}
    )

    # If we are genering a CUDA target, we will need to link against the CUDA
    # runtime to get the right headers.
    if(${LANGUAGE_TARGET} STREQUAL "cuda")
        target_link_libraries(
            ${TARGET_NAME}
            PUBLIC
            CUDA::cudart
            PRIVATE
            CUDA::cuda_driver
            CUDA::nvrtc
        )
    endif()

    # Add the newly generated header file as an include directory of our newly
    # generated target.
    target_include_directories(
        ${TARGET_NAME}
        PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}/include/
    )
endfunction()

# Create a new CUDA core library for CCA from our Futhark sources.
add_futhark_library(
    traccc_futhark_cca_cuda_core
    C
    src/cca_core.fut
)

# Create a new CUDA core library for SP creation from our Futhark sources.
add_futhark_library(
    traccc_futhark_spc_cuda_core
    C
    src/spc_core.fut
)

# Create a new traccc library which is built from the CCA CUDA core, plus some
# extra code.
traccc_add_library(
    traccc_futhark
    futhark
    TYPE SHARED
    "src/cca.cpp"
    "include/traccc/futhark/cca.hpp"
    "src/spc.cpp"
    "include/traccc/futhark/spc.hpp"
)

# Link the new outward-facing library against our code librar(y|ies)
target_link_libraries(
    traccc_futhark
    PUBLIC
    traccc::core
    PRIVATE
    vecmem::core
    traccc_futhark_cca_cuda_core
    traccc_futhark_spc_cuda_core
)
