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
unsigned int atomic_add(unsigned int *address, unsigned int val)
{
  ::sycl::atomic<unsigned int, ::sycl::access::address_space::global_space> obj ((
    ::sycl::multi_ptr<unsigned int, ::sycl::access::address_space::global_space>(
      address)));

  unsigned int old_val = *address;
  while(true)
  {
    const unsigned int new_val = old_val + val;
    if(obj.compare_exchange_strong(old_val, new_val))
      break;
  }
  return old_val;
}

} // namespace sycl
} // namespace traccc