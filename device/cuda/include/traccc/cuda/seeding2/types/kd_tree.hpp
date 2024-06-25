/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <traccc/cuda/seeding2/types/range3d.hpp>
#include <traccc/utils/array_wrapper.hpp>
#include <vecmem/memory/unique_ptr.hpp>

namespace traccc::cuda {
enum class nodetype_e { LEAF, INTERNAL, NON_EXTANT };

enum class pivot_e { Phi, R, Z };

template <template <typename> typename F>
struct node_t {
    using tuple_t =
        std::tuple<nodetype_e, range3d, uint32_t, uint32_t, pivot_e, float>;

    F<nodetype_e> type;
    F<range3d> range;
    F<uint32_t> begin, end;
    F<pivot_e> dim;
    F<float> mid;
};

using kd_tree = array_wrapper<soa, node_t>;

using kd_tree_owning_t = kd_tree::owner;

using kd_tree_t = kd_tree::handle;
}  // namespace traccc::cuda
