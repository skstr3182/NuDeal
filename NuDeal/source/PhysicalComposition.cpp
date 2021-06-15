#include "PhysicalComposition.h"
namespace PhysicalComposition {
void MatComp::PushBoundary(int BaseType, double lossrate){
	int i = Boundary.size(); Boundary.emplace_back();
	Boundary[i].BaseType = BaseType; Boundary[i].lossrate = lossrate;
}
}