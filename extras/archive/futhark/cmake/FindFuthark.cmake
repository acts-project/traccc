# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Necessary CMake includes
include(FindPackageHandleStandardArgs)
include(CheckLanguage)

# The C language is not enabled in our project by default, but we need it to
# compile Futhark's code.
check_language(C)
if(CMAKE_C_COMPILER)
    enable_language(C)
endif()

# Try to find our dependencies, which are of course Futhark itself (required)
# and the CUDA toolkit for compiling CUDA projects. Also notify the user of
# what we find.
find_program(Futhark_EXECUTABLE futhark)
mark_as_advanced(Futhark_EXECUTABLE)
find_package(CUDAToolkit)

# Report the status of loading Futhark to the user.
find_package_handle_standard_args(
    Futhark
    REQUIRED_VARS
    Futhark_EXECUTABLE
    CMAKE_C_COMPILER
)

# This function describes how to add a Futhark source to a CMake library. The
# function adds sources generated from a source file `_SOURCE_NAME` to a target
# called `TARGET_NAME`. The Futhark target platform (CUDA or C) is given
# (case-insensitively) by `_LANGUAGE_TARGET`
function(add_futhark_to_library TARGET_NAME _LANGUAGE_TARGET _SOURCE_NAME)
    # Parse the arguments that we get.
    cmake_parse_arguments(
        ARGS
        ""
        ""
        "DEPENDENCIES"
        ${ARGN}
    )

    # Set a few useful variables.
    set(FUTHARK_LANGUAGES c cuda)
    set(SOURCE_NAME "${CMAKE_CURRENT_SOURCE_DIR}/${_SOURCE_NAME}")
    string(TOLOWER ${_LANGUAGE_TARGET} LANGUAGE_TARGET)
    string(TOUPPER ${_LANGUAGE_TARGET} LANGUAGE_TARGET_FANCY)

    # Notify the user of what we are doing.
    message(
        STATUS
        "Adding ${LANGUAGE_TARGET_FANCY} sources generated from Futhark source ${SOURCE_NAME} to ${TARGET_NAME}."
    )

    # Make sure that the source file actually exists.
    if(NOT EXISTS ${SOURCE_NAME})
        message(
            FATAL_ERROR
            "File ${SOURCE_NAME} does not exist!"
        )
    endif()

    # Ensure that the requested language is actually a supported one, namely C
    # or CUDA.
    list(FIND FUTHARK_LANGUAGES ${LANGUAGE_TARGET} index)
    if(index EQUAL -1)
        message(
            FATAL_ERROR
            "Futhark language must be one of ${FUTHARK_LANGUAGES}, not ${LANGUAGE_TARGET}."
        )
    endif()

    # Define some needed directory and file names.
    get_filename_component(FILE_ROOT "${SOURCE_NAME}" NAME_WE)
    set(FUTHARK_OUTPUT_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/src")
    set(FUTHARK_OUTPUT_HEADER_DIR "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/include/traccc/futhark")
    set(FUTHARK_OUTPUT_SOURCE "${FUTHARK_OUTPUT_SOURCE_DIR}/${FILE_ROOT}.c")
    set(FUTHARK_OUTPUT_HEADER "${FUTHARK_OUTPUT_HEADER_DIR}/${FILE_ROOT}.h")

    # Create the destination directories.
    file(MAKE_DIRECTORY "${FUTHARK_OUTPUT_SOURCE_DIR}")
    file(MAKE_DIRECTORY "${FUTHARK_OUTPUT_HEADER_DIR}")

    # Define the custom command for generating .c and .h files from .fut files.
    add_custom_command(
        # Generate two files, a source file and a header file.
        OUTPUT "${FUTHARK_OUTPUT_SOURCE}" "${FUTHARK_OUTPUT_HEADER}"

        # Invoke the Futhark compiler in library mode.
        COMMAND ${Futhark_EXECUTABLE}
        ARGS ${LANGUAGE_TARGET} --library -o "${FILE_ROOT}" "${SOURCE_NAME}"

        # Move the generated source and header file to their destinations.
        COMMAND "${CMAKE_COMMAND}" -E copy "${FILE_ROOT}.c" "${FUTHARK_OUTPUT_SOURCE}"
        COMMAND "${CMAKE_COMMAND}" -E copy "${FILE_ROOT}.h" "${FUTHARK_OUTPUT_HEADER}"

        # Register the generated JSON interface file as a byproduct.
        BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${FILE_ROOT}.json"

        # Mark the Futhark file as the main dependency.
        MAIN_DEPENDENCY "${SOURCE_NAME}"

        # Mark the auxiliary Futhark files as dependencies.
        DEPENDS ${ARGS_DEPENDENCIES}

        # Execute this from the binary directory, to ensure that no dirty files
        # are generated in the user's work directory.
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}"

        # Notify the user that we are generating code from this Futhark file.
        COMMENT "Generating ${LANGUAGE_TARGET_FANCY} file ${FILE_ROOT}.c from ${SOURCE_NAME}"
        VERBATIM
    )

    # With the output source now generated, we can add it to the target
    # library.
    target_sources(
        ${TARGET_NAME}
        PRIVATE
        "${FUTHARK_OUTPUT_SOURCE}"
    )

    # If we are genering a CUDA target, we will need to link against the CUDA
    # runtime to get the right headers.
    if(${LANGUAGE_TARGET} STREQUAL "cuda")
        if(NOT CUDAToolkit_FOUND)
            message(
                FATAL_ERROR
                "CUDA compilation of Futhark sources ${SOURCE_NAME} for ${TARGET_NAME} requested, but CUDA toolkit is not available!"
            )
        endif()

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
        PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/include/"
    )
endfunction()
