# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
#
# SPDX-License-Identifier: MPL-2.0

# Helper function testing the covfie public headers.
#
# It can be used to test that public headers would include everything
# that they need to work, and that the CMake library targets would take
# care of declaring all of their dependencies correctly for the public
# headers to work.
#
# Usage: covfie_test_public_headers( covfie_core
#                                    include/header1.hpp ... )
#
function(covfie_test_public_headers library)
    # All arguments are treated as header file names.
    foreach(_headerName ${ARGN})
        # Make the header filename into a "string".
        string(REPLACE "/" "_" _headerNormName "${_headerName}")
        string(REPLACE "." "_" _headerNormName "${_headerNormName}")

        # Write a small source file that would test that the public
        # header can be used as-is.
        set(_testFileName
            "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/test_${library}_${_headerNormName}.cpp"
        )
        if(NOT EXISTS "${_testFileName}")
            file(
                WRITE
                "${_testFileName}"
                "#include \"${_headerName}\"\n"
                "int main() { return 0; }"
            )
        endif()

        # Set up an executable that would build it. But hide it, don't put it
        # into ${CMAKE_BINARY_DIR}/bin.
        add_executable("test_${library}_${_headerNormName}" "${_testFileName}")
        target_link_libraries(
            "test_${library}_${_headerNormName}"
            PRIVATE
                ${library}
        )
        set_target_properties(
            "test_${library}_${_headerNormName}"
            PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY
                    "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}"
        )
    endforeach()
endfunction(covfie_test_public_headers)

# Helper function for adding individual flags to "flag variables".
#
# Usage: covfie_add_flag( CMAKE_CXX_FLAGS "-Wall" )
#
function(covfie_add_flag name value)
    # Escape special characters in the value:
    set(matchedValue "${value}")
    foreach(
        c
        "*"
        "."
        "^"
        "$"
        "+"
        "?"
    )
        string(REPLACE "${c}" "\\${c}" matchedValue "${matchedValue}")
    endforeach()

    # Check if the variable already has this value in it:
    if("${${name}}" MATCHES "${matchedValue}")
        return()
    endif()

    # If not, then let's add it now:
    set(${name} "${${name}} ${value}" PARENT_SCOPE)
endfunction(covfie_add_flag)
