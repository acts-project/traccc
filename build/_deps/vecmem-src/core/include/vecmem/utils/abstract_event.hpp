/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem {

/// Interface that language specific "events" need to implement
///
/// "Events" provide a way to have synchronization points in a program's
/// execution flow. Asynchronous memory operations in different languages
/// provide ways of explicitly waiting for the completion of certain operations.
/// This abstract interface is used in this project to provide a uniform
/// interface to language specific "event" objects that would provide such
/// synchronization points.
///
struct abstract_event {

    /// Virtual destructor to make vtable happy
    virtual ~abstract_event() {}

    /// Function that would block the current thread until the event is
    /// complete
    virtual void wait() = 0;

    /// Function telling the object not to wait for the underlying event
    virtual void ignore() = 0;

};  // struct abstract_event

}  // namespace vecmem
