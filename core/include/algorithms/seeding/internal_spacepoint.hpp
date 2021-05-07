/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/spacepoint.hpp"

namespace traccc{

template< typename spacepoint >
class internal_spacepoint{
public:

    internal_spacepoint() = delete;
    internal_spacepoint(const spacepoint& sp, const vector3& globalPos,
			const vector2& offsetXY,
			const vector2& variance);
    internal_spacepoint(const internal_spacepoint<spacepoint>& sp);

    internal_spacepoint<spacepoint>& operator=(const internal_spacepoint<spacepoint>&);
    
    const float& x() const { return m_x; }
    const float& y() const { return m_y; }
    const float& z() const { return m_z; }
    const float& radius() const { return m_r; }
    float phi() const { return atan2f(m_y, m_x); }
    const float& varianceR() const { return m_varianceR; }
    const float& varianceZ() const { return m_varianceZ; }
    const spacepoint& sp() const { return m_sp; }
    
private:
    float m_x;               // x-coordinate in beam system coordinates
    float m_y;               // y-coordinate in beam system coordinates
    float m_z;               // z-coordinate in beam system coordinetes
    float m_r;               // radius       in beam system coordinates
    float m_varianceR;       //
    float m_varianceZ;       //
    const spacepoint& m_sp;  // external space point

};
    
template < typename spacepoint >
inline internal_spacepoint<spacepoint>::internal_spacepoint(
    const spacepoint& sp, const vector3& globalPos,
    const vector2& offsetXY,
    const vector2& variance)
    : m_sp(sp) {
	m_x = globalPos[0] - offsetXY[0];
	m_y = globalPos[1] - offsetXY[1];
	m_z = globalPos[2];
	m_r = std::sqrt(m_x * m_x + m_y * m_y);
	m_varianceR = variance[0];
	m_varianceZ = variance[1];
    }
    

template <typename spacepoint>
inline internal_spacepoint<spacepoint>::internal_spacepoint(
    const internal_spacepoint<spacepoint>& sp)
    : m_sp(sp.sp()) {
    m_x = sp.m_x;
    m_y = sp.m_y;
    m_z = sp.m_z;
    m_r = sp.m_r;
    m_varianceR = sp.m_varianceR;
    m_varianceZ = sp.m_varianceZ;
}
    

} // namespace traccc
