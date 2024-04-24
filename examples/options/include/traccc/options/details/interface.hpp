/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Boost include(s).
#include <boost/program_options.hpp>

// System include(s).
#include <iosfwd>
#include <string>
#include <string_view>

namespace traccc::opts {

/// Common base class / interface for all of the program option classes
class interface {

    public:
    /// Constructor on top of a common @c boost::program_options object
    ///
    /// @param description The description of this program option group
    ///
    interface(std::string_view description);
    /// Virtual destructor
    virtual ~interface() = default;

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    virtual void read(const boost::program_options::variables_map& vm);

    /// Helper for printing the options to an output stream
    ///
    /// @param out The output stream to print to
    /// @return The output stream
    ///
    std::ostream& print(std::ostream& out) const;

    /// Get the description of this program option group
    const boost::program_options::options_description& options() const;

    protected:
    /// Print the specific options of the derived class
    virtual std::ostream& print_impl(std::ostream& out) const;

    /// (Boost) Description of this program option group
    boost::program_options::options_description m_desc;

    private:
    /// (String) Description of this program option group
    std::string m_description;

};  // class interface

/// Printout helper for @c traccc::opts::interface and types derived from it
std::ostream& operator<<(std::ostream& out, const interface& opt);

}  // namespace traccc::opts
