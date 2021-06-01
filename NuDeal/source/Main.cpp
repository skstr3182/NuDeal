#include "Defines.h"
#include "UnitGeo.h"
#include "GeoHandle.h"
#include "Input.h"
#include "HardCodeParam.h"
#include "Array.hpp"
#include "MPIBind.h"

int main(int argc, char *argv[])
{
	MPI::Init(&argc, &argv);	
	MPI::Configure_cuda_types();


	LinPack::Array_t<float> arr(100);

	arr.Fill(200.0);

	//std::fill(a, a + 100, make_int2(200, 300));

	//MPI::Send(a, 100, 0, 0, MPI_COMM_WORLD);
	//MPI::Recv(b, 100, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	//MPI::Gather(a, 100, b, 100, 0, MPI_COMM_WORLD);

	/*IO::InputManager_t Parser;
	std::string file = std::string(argv[1]);
	Parser.ReadInput(file);*/
	//Geometry::DebugUnitGeo();
	//Geometry::DebugGeomHandle();

	MPI::Finalize();

	return EXIT_SUCCESS;
}