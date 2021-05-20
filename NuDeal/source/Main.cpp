#include "Defines.h"
#include "UnitGeo.h"
#include "GeoHandle.h"
#include "Input.h"
#include "HardCodeParam.h"

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	IO::InputManager_t Parser;
	std::string file = std::string(argv[1]);
	//Geometry::DebugUnitGeo();
	Geometry::DebugGeomHandle();

	MPI_Finalize();

	return EXIT_SUCCESS;
}