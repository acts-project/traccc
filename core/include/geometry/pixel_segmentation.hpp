/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "definitions/primitives.hpp"

// CUDA macros
#ifdef __CUDACC__      
#include <cuda.h>
#include <cuda_runtime.h>
#include "definitions/cuda_macros.hpp"
#define __CUDA_QUALIFIER__ inline __device__ __host__
#else
#define __CUDA_QUALIFIER__
#endif      

namespace traccc {

    /// A very basic pixel segmentation with 
    /// a minimum corner and ptich x/y
    ///
    /// No checking on out of bounds done
    struct pixel_segmentation {

      scalar min_center_x;
      scalar min_center_y;
      scalar pitch_x;
      scalar pitch_y;

      /// Translate @param ch0, @param ch1 
      /// into a vector2 at @return for a pixel segemntation
      __CUDA_QUALIFIER__
      vector2 operator()(channel_id ch0,channel_id ch1) 
      { 
          return { min_center_x + ch0 * pitch_x, min_center_y + ch1 * pitch_y };
      };

    };

}

