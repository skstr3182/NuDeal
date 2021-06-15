#pragma once
#include "Defines.h"
#include "UnitGeo.h"
#include "GeoHandle.h"
#include "IODeclare.h"

namespace Geometry {

	class Master_t {
	private:
		
		vector<UnitVol> basicVolumes;
		vector<UnitComp> basicComps;

	public:
		
		void SetBasicUnitGeom(const IO::InputManager_t *Input);
	};
}