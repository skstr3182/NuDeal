#pragma once
#include "Defines.h"
#include "Array.hpp"
#include "PhysicalDomain.h"

namespace Transport {
class DriverSCMOC{
	template <typename T> using Array = LinPack::Array_t<T>;
	using RaySegment = PhysicalDomain::RaySegmentDomain;
	using Dimension = Geometry::Dimension;
private:
	Dimension mode;
	int nangle_oct;

	int3 Nxyz, nxyz;
	double3 lxyz0;
	int nnode;
	int divlevel;
	Array<double> trackL0; // trackL[nangle_oct]
	Array<double> wtIF0; // weights from interfaces along x,y,z axes, wtIF[nangle_oct][3]

public:
	void Initialize(const RaySegment& rhs);

	DriverSCMOC(int nangle_oct) { this->nangle_oct = nangle_oct; }

	DriverSCMOC(int nangle_oct, const RaySegment& rhs) { this->nangle_oct = nangle_oct; Initialize(rhs); }
};
}