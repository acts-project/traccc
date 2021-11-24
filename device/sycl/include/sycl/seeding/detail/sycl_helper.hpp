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

inline
unsigned int atomic_add(unsigned int* address, unsigned int val)
{
  ::sycl::ext::oneapi::atomic_ref<unsigned int, ::sycl::memory_order::relaxed,
                                   ::sycl::memory_scope::device,
                                   ::sycl::access::address_space::global_space> obj (*address);

  unsigned int old_val = *address;
  while(true)
  {
    const unsigned int new_val = old_val + val;
    if(obj.compare_exchange_strong(old_val, new_val))
      break;
  }
  return old_val;
}

inline
void reduceInShared(int *const v, ::sycl::nd_item<1> &item)
{
  int lid = item.get_local_id(0);
  if(lid<64) v[lid] = v[lid] + v[lid+64];
  item.barrier(::sycl::access::fence_space::local_space);
  if(lid<32) v[lid] = v[lid] + v[lid+32];
  item.barrier(::sycl::access::fence_space::local_space);
  if(lid<32) v[lid] = v[lid] + v[lid+16];
  item.barrier(::sycl::access::fence_space::local_space);
  if(lid<32) v[lid] = v[lid] + v[lid+8];
  item.barrier(::sycl::access::fence_space::local_space);
  if(lid<32) v[lid] = v[lid] + v[lid+4];
  item.barrier(::sycl::access::fence_space::local_space);
  if(lid<32) v[lid] = v[lid] + v[lid+2];
  item.barrier(::sycl::access::fence_space::local_space);
  if(lid<32) v[lid] = v[lid] + v[lid+1];
  item.barrier(::sycl::access::fence_space::local_space);
}

} // namespace sycl
} // namespace traccc