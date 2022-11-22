/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <fstream>
#include <string_view>
#include <type_traits>
#include <vector>

namespace traccc::io::details {

/// Function for writing a container into a binary file
///
/// @param filename is the output filename which includes the path
/// @param container is the traccc container to write
///
template <typename container_t>
void write_binary_container(std::string_view filename,
                            const container_t& container) {

    // Make sure that the chosen types work.
    static_assert(std::is_standard_layout_v<typename container_t::header_type>,
                  "Container header type must have standard layout.");
    static_assert(std::is_standard_layout_v<typename container_t::item_type>,
                  "Container item type must have standard layout.");

    // Open the output file.
    std::ofstream out_file(filename.data(), std::ios::binary);

    // Write the size of the header vector.
    const std::size_t headers_size = container.size();
    out_file.write(reinterpret_cast<const char*>(&headers_size),
                   sizeof(std::size_t));

    // Write the sizes of the item vectors.
    std::vector<std::size_t> item_sizes;
    item_sizes.reserve(headers_size);
    for (const typename container_t::item_vector::value_type& i :
         container.get_items()) {
        item_sizes.push_back(i.size());
    }
    out_file.write(reinterpret_cast<const char*>(item_sizes.data()),
                   headers_size * sizeof(std::size_t));

    // Write header elements.
    out_file.write(
        reinterpret_cast<const char*>(container.get_headers().data()),
        container.get_headers().size() *
            sizeof(typename container_t::header_type));

    // Write the items.
    for (const typename container_t::item_vector::value_type& i :
         container.get_items()) {
        out_file.write(reinterpret_cast<const char*>(i.data()),
                       i.size() * sizeof(typename container_t::item_type));
    }
}

}  // namespace traccc::io::details
