/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <CL/sycl.hpp>

namespace traccc{
namespace sycl {

// inline
// unsigned int& atomic_add(unsigned int& address,int& val)
// {
//   ::sycl::ext::oneapi::atomic_ref<unsigned int, ::sycl::memory_order::seq_cst,
//                                    ::sycl::memory_scope::device,
//                                    ::sycl::access::address_space::global_space> obj (address);

//   unsigned int old_val = obj.fetch_add(val);

//   return old_val;
// }

// inline
// void reduceInShared(int* array, ::sycl::nd_item<1> &item)
// {
//   const auto& workItemIdx = item.get_local_id(0);
//   const auto& groupDim = item.get_local_range(0);
//   auto sg = item.get_sub_group();
//   auto workGroup = item.get_group();

//   array[workItemIdx] += ::sycl::shift_group_left(sg, array[workItemIdx], 4);
//   array[workItemIdx] += ::sycl::shift_group_left(sg, array[workItemIdx], 2);
//   // array[workItemIdx] += ::sycl::shift_group_left(sg, array[workItemIdx], 4);
//   // array[workItemIdx] += ::sycl::shift_group_left(sg, array[workItemIdx], 2);
//   array[workItemIdx] += ::sycl::shift_group_left(sg, array[workItemIdx], 1);

//   ::sycl::group_barrier(workGroup);

//   if (workItemIdx == 0) {
//       for (int i = 1; i < groupDim / 8; i++) {
//           array[workItemIdx] += array[i * 8];
//     }
//   }
// }

} // namespace sycl
} // namespace traccc