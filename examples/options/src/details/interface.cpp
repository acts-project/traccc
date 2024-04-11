/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/details/interface.hpp"

namespace traccc::opts {

interface::interface(std::string_view group_desc)
    : m_desc(std::string{group_desc}), m_description(group_desc) {}

void interface::read(const boost::program_options::variables_map&) {}

std::ostream& interface::print(std::ostream& out) const {

    out << ">>> " << m_description << " <<<\n";
    return print_impl(out);
}

const boost::program_options::options_description& interface::options() const {

    return m_desc;
}

std::ostream& interface::print_impl(std::ostream& out) const {

    return (out << "  None");
}

std::ostream& operator<<(std::ostream& out, const interface& opt) {

    return opt.print(out);
}

}  // namespace traccc::opts
