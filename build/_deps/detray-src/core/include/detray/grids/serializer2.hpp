/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <algorithm>

#include "detray/definitions/detail/indexing.hpp"
#include "detray/definitions/detail/qualifiers.hpp"

namespace detray {
/** Struct that helps taking a two-dimensional binning into
 * a serial binning for data storage.
 *
 * Serializers allow to create a memory local environment if
 * advantegeous.
 *
 **/
struct serializer2 {
    /** Create a serial bin from two individual bins
     *
     * @tparam faxis_t is the type of the first axis
     * @tparam saxis_t is the type of the second axis
     *
     * @param faxis the first axis
     * @param saxis the second axis, unused here
     * @param fbin first bin
     * @param sbin second bin
     *
     * @return a dindex for the memory storage
     */
    template <typename faxis_t, typename saxis_t>
    DETRAY_HOST_DEVICE dindex serialize(const faxis_t &faxis,
                                        const saxis_t & /*saxis*/, dindex fbin,
                                        dindex sbin) const {

        dindex offset = sbin * faxis.bins();
        return offset + fbin;
    }

    /** Create a bin tuple from a serialized bin
     *
     * @tparam faxis_t is the type of the first axis
     * @tparam saxis_t is the type of the second axis
     *
     * @param faxis the first axis
     * @param saxis the second axis, unused here
     * @param serialbin the serial bin
     *
     * @return a 2-dim array of dindex
     */
    template <typename faxis_t, typename saxis_t,
              template <typename, std::size_t> class array_t = darray>
    DETRAY_HOST_DEVICE array_t<dindex, 2> deserialize(const faxis_t &faxis,
                                                      const saxis_t & /*saxis*/,
                                                      dindex serialbin) const {
        auto sbin = static_cast<dindex>(serialbin / faxis.bins());
        dindex fbin = serialbin - sbin * faxis.bins();
        return {fbin, sbin};
    }
};

}  // namespace detray
