# Detray library, part of the ACTS project (R&D line)
#
# (c) 2022-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

cmake_minimum_required(VERSION 3.11)
include(FetchContent)

# Silence FetchContent warnings with CMake >=3.24.
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# Tell the user what's happening.
message(STATUS "Building nlohmann_json as part of the detray project")

# Declare where to get nlohmann from.
set(DETRAY_NLOHMANN_JSON_GIT_TAG
    "v3.11.3"
    CACHE STRING
    "Version of nlohmann_json to build"
)
set(DETRAY_NLOHMANN_JSON_SHA1
    "2074caa675f8097d9b03c0f4976ffc3410170937"
    CACHE STRING
    "SHA1 hash of the downloaded zip"
)
mark_as_advanced(
    DETRAY_NLOHMANN_JSON_GIT_REPOSITORY
    DETRAY_NLOHMANN_JSON_GIT_TAG
)

# Mark the import as a system library on modern CMake versions
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.25.0)
    set(DETRAY_NLOHMANN_JSON_SOURCE_FULL
        "URL;https://github.com/nlohmann/json/archive/refs/tags/${DETRAY_NLOHMANN_JSON_GIT_TAG}.tar.gz;URL_HASH;SHA1=${DETRAY_NLOHMANN_JSON_SHA1};SYSTEM"
    )
else()
    set(DETRAY_NLOHMANN_JSON_SOURCE_FULL
        "URL;https://github.com/nlohmann/json/archive/refs/tags/${DETRAY_NLOHMANN_JSON_GIT_TAG}.tar.gz;URL_HASH;SHA1=${DETRAY_NLOHMANN_JSON_SHA1}"
    )
endif()
mark_as_advanced(DETRAY_NLOHMANN_JSON_SOURCE_FULL)

FetchContent_Declare(nlohmann_json ${DETRAY_NLOHMANN_JSON_SOURCE_FULL})

# Now set up its build.
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install ON CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)
