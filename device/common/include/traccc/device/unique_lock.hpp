/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <mutex>

#include "traccc/definitions/qualifiers.hpp"

namespace traccc::device {
template <typename Mutex>
class unique_lock {
    public:
    using mutex_type = Mutex;

    /*
     * Construct a unique lock without locking.
     */
    TRACCC_HOST_DEVICE
    unique_lock(mutex_type& m, std::defer_lock_t);

    /*
     * Construct a unique lock, attempting to lock it.
     */
    TRACCC_HOST_DEVICE
    unique_lock(mutex_type& m, std::try_to_lock_t);

    /*
     * Construct a unique lock which was previously locked.
     */
    TRACCC_HOST_DEVICE
    unique_lock(mutex_type& m, std::adopt_lock_t);

    /*
     * Destroy a lock, freeing the underlying mutex.
     */
    TRACCC_HOST_DEVICE
    ~unique_lock();

    /*
     * Lock the lock, blocking until the operation succeeds.
     */
    TRACCC_HOST_DEVICE
    void lock();

    /*
     * Try to lock the lock without blocking.
     */
    TRACCC_HOST_DEVICE
    bool try_lock();

    /*
     * Explicitly lock the underlying lock.
     */
    TRACCC_HOST_DEVICE
    void unlock();

    /*
     * Check if the lock is locked by this object.
     */
    TRACCC_HOST_DEVICE
    bool owns_lock() const;

    private:
    mutex_type* m_mutex_ptr = nullptr;
    bool m_owns_lock;
};
}  // namespace traccc::device

#include "impl/unique_lock.ipp"
