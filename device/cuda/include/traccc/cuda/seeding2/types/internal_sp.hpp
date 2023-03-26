/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <traccc/edm/internal_spacepoint.hpp>
#include <traccc/edm/spacepoint.hpp>
#include <traccc/utils/array_wrapper.hpp>
#include <vecmem/memory/unique_ptr.hpp>

namespace traccc::cuda {
template <template <typename> typename F>
struct sp_t {
    using tuple_t = std::tuple<float, float, float, float, float, unsigned int>;

    F<float> x;
    F<float> y;
    F<float> z;
    F<float> phi;
    F<float> radius;
    F<unsigned int> link;
};

using internal_sp = array_wrapper<soa, sp_t>;

using internal_sp_owning_t = internal_sp::owner;

using internal_sp_t = internal_sp::handle;
}  // namespace traccc::cuda
