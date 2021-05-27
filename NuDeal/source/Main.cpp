#include "Defines.h"
#include "UnitGeo.h"
#include "GeoHandle.h"
#include "Input.h"
#include "HardCodeParam.h"
#include "Array.hpp"

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);	

	LinPack::Array_t<double> a(100);
	LinPack::Array_t<string> str_array(10);

	str_array[0] = "Some String";

	a.ResizeDevice();
	a.Fill(2.0);
	a.FillDevice(2.0);

	vector<double> doubleVec(100);

	IO::InputManager_t Parser;
	std::string file = std::string(argv[1]);
	Parser.ReadInput(file);
	//Geometry::DebugUnitGeo();
	//Geometry::DebugGeomHandle();

	MPI_Finalize();

	return EXIT_SUCCESS;
}