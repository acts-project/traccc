/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

namespace traccc::alpaka {

using Dim = ::alpaka::DimInt<1>;
using Idx = uint32_t;
using WorkDiv = ::alpaka::WorkDivMembers<Dim, Idx>;

using Acc = ::alpaka::ExampleDefaultAcc<Dim, Idx>;
using Host = ::alpaka::DevCpu;
using Queue = ::alpaka::Queue<Acc, ::alpaka::Blocking>;

}  // namespace traccc::alpaka
