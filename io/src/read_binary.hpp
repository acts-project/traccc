/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <cstddef>
#include <fstream>
#include <string_view>
#include <type_traits>
#include <vector>

namespace traccc::io::details {

/// Function for reading a container from a binary file
///
/// @param filename The full input filename
/// @param mr Is the memory resource to create the result container with
///
template <typename container_t>
container_t read_binary_container(std::string_view filename,
                                  vecmem::memory_resource* mr = nullptr) {

    // Make sure that the chosen types work.
    static_assert(std::is_standard_layout_v<typename container_t::header_type>,
                  "Container header type must be standard layout.");
    static_assert(std::is_standard_layout_v<typename container_t::item_type>,
                  "Container item type must be standard layout.");

    // Open the input file.
    std::ifstream in_file(filename.data(), std::ios::binary);

    // Read the size of the header vector.
    std::size_t headers_size;
    in_file.read(reinterpret_cast<char*>(&headers_size), sizeof(std::size_t));

    // Read the sizes of the item vector.
    std::vector<std::size_t> items_size(headers_size);
    in_file.read(reinterpret_cast<char*>(items_size.data()),
                 headers_size * sizeof(typename std::size_t));

    // Create the result container, and set it to the correct (outer) size right
    // away.
    container_t result(headers_size, mr);

    // Read the header payload into memory.
    in_file.read(reinterpret_cast<char*>(result.get_headers().data()),
                 headers_size * sizeof(typename container_t::header_type));

    // Read the items in multiple steps.
    for (std::size_t i = 0; i < headers_size; ++i) {
        result.get_items().at(i).resize(items_size.at(i));
        in_file.read(
            reinterpret_cast<char*>(result.get_items().at(i).data()),
            items_size.at(i) * sizeof(typename container_t::item_type));
    }

    // Return the newly created container.
    return result;
}

/// Function for reading a collection from a binary file
///
/// @param filename The full input filename
/// @param mr Is the memory resource to create the result collection with
///
template <typename collection_t>
collection_t read_binary_collection(std::string_view filename,
                                    vecmem::memory_resource* mr = nullptr) {

    // Make sure that the chosen types work.
    static_assert(std::is_standard_layout_v<typename collection_t::value_type>,
                  "Collection item type must be standard layout.");

    // Open the input file.
    std::ifstream in_file(filename.data(), std::ios::binary);

    // Read the size of the header vector.
    std::size_t size;
    in_file.read(reinterpret_cast<char*>(&size), sizeof(std::size_t));

    // Create the result collection, and set it to the correct size.
    collection_t result(size, mr);

    // Read the items into memory.
    in_file.read(reinterpret_cast<char*>(result.data()),
                 size * sizeof(typename collection_t::value_type));

    // Return the newly created container.
    return result;
}

}  // namespace traccc::io::details
