#include "Defines.h"
#include "UnitGeo.h"
#include "GeoHandle.h"
#include "Input.h"
#include "HardCodeParam.h"
#include "Array.hpp"
#include "Array_v2.hpp"
#include "Vector.hpp"

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);	

	LinPack_v2::Array_t<double> a(10, 10, 10);

	a.ResizeDevice();

	//thrust::host_vector<int> ah(100);
	//thrust::device_vector<int> ad(100);

	//LinPack_v2::Array_t<string> h;
	//LinPack_v2::Array_t<string> hh;

	//thrust::device_vector<int> dddd(1000);

	//h.Resize(hh.GetDims());
	//a.ResizeDevice(b.GetDims());

	//LinPack::Array_t<double> a(100);
	//LinPack::Array_t<map<int, int>> str_array(10);

	//str_array[0] = "Some String";

	//a.ResizeDevice();
	//a.Fill(2.0);
	//a.FillDevice(2.0);

	vector<double> doubleVec(100);

	IO::InputManager_t Parser;
	std::string file = std::string(argv[1]);
	Parser.ReadInput(file);
	//Geometry::DebugUnitGeo();
	//Geometry::DebugGeomHandle();

	MPI_Finalize();

	return EXIT_SUCCESS;
}