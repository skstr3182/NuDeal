#include "Geometry.h"
#include "Input.h"

namespace Geometry {

void Master_t::SetBasicUnitGeom(const IO::InputManager_t *Input)
{
	//const auto& unitVolumes = Input->GetUnitVolumeInfo();
	//const auto& unitComps = Input->GetUnitCompInfo();

	//for (const auto& iter : unitVolumes) {
	//	auto name = iter.first;
	//	const auto& obj = iter.second;	
	//	auto& vol = basicVolumes.emplace_back(name);
	//	for (const auto& eq : obj.equations) {
	//		vol.Append(UnitSurf(eq));
	//	}
	//	vol.Finalize();
	//}

}

}