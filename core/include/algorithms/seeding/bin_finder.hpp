/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "algorithms/seeding/spacepoint_grid.hpp"

namespace traccc{

/// @class BinFinder
/// The BinFinder is used by the ISPGroupSelector. It can be
/// used to find both bins that could be bottom bins as well as bins that could
/// be top bins, which are assumed to be the same bins. Does not take
/// interaction region into account to limit z-bins.
class bin_finder {
 public:
  /// destructor
  ~bin_finder() = default;

  /// Return all bins that could contain space points that can be used with the
  /// space points in the bin with the provided indices to create seeds.
  /// @param phiBin phi index of bin with middle space points
  /// @param zBin z index of bin with middle space points
  /// @param binnedSP phi-z grid containing all bins
  std::vector<size_t> findBins(
      size_t phiBin, size_t zBin,
      const spacepoint_grid& binnedSP){
      return binnedSP.neighborhood_indices({phiBin, zBin}).collect();
  }
};
   
} // namespace traccc
