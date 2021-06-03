#include "Defines.h"
#include "UnitGeo.h"
#include "GeoHandle.h"
#include "Input.h"
#include "HardCodeParam.h"
#include "Array.hpp"
#include "MPIBind.h"
#include "OpenMP.h"

int main(int argc, char *argv[])
{
	MPI::Init(&argc, &argv);	
	MPI::Configure_cuda_types();

	int arr[5] = {0, };
	int arr2[5];

	std::copy(arr, arr + 5, arr2);


	/*IO::InputManager_t Parser;
	std::string file = std::string(argv[1]);
	Parser.ReadInput(file);*/
	//Geometry::DebugUnitGeo();
	//Geometry::DebugGeomHandle();

	MPI::Finalize();

	return EXIT_SUCCESS;
}