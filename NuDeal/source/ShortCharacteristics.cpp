#include "ShortCharacteristics.h"

namespace Transport {
void DriverSCMOC::Initialize(const RaySegment& rhs){
	mode = rhs.GetDimension();
	rhs.GetBaseSizes(Nxyz, nxyz, nnode, divlevel);
	rhs.GetNodeSizes(lxyz0);
}
}