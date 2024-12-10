/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem::hip::details {

/**
 * @brief Class with RAII mechanism for selecting a HIP device.
 *
 * This class can be used to select HIP devices in a modern C++ way, with
 * scope safety. When an object of this class is constructed, it will switch
 * the thread-local device selector to the device number specified in the
 * constructor argument. When this object goes out of scope or gets
 * destructed in any other way, it will restore the device that was set
 * before the object was constructed. This allows us to easily write methods
 * with few side-effects.
 *
 * @warning The behaviour of this class is not well-defined if you construct
 * more than one in the same scope.
 */
class select_device {

public:
    /**
     * @brief Constructs the object, switching the current HIP device
     * to the requested number.
     *
     * @param[in] device The HIP device number to switch to.
     */
    select_device(int device);

    /**
     * @brief Deconstructs the object, returning to the device that was
     * selected before constructing this object.
     */
    ~select_device();

private:
    /**
     * @brief The old device number, this is what we restore when the
     * object goes out of scope.
     */
    int m_device;

};  // class select_device

}  // namespace vecmem::hip::details
