#include "Defines.h"
#include "UnitGeo.h"
#include "GeoHandle.h"
#include "Input.h"
#include "HardCodeParam.h"
#include "XS.h"
#include "Array.h"
#include "MPI.hpp"
#include "OMP.hpp"

template class LinPack::Array_t<int>;
template <typename T> using Array_t = LinPack::Array_t<T>;

int main(int argc, char *argv[])
{
	MPI::Init(&argc, &argv);	
	MPI::Configure_cuda_types();

	//LinPack::Array_t<double> a;
	//LinPack::Array_t<double> b;
	//LinPack::Array_t<int> c;

	//vector<double> doubleVec(100);

	//IO::InputManager_t Parser;
	//std::string file = std::string(argv[1]);
	//Parser.ReadInput(file);

	//Geometry::DebugUnitGeo();
	//Geometry::DebugGeomHandle();
	//PhysicalDomain::DebugPhysicalDomain();

	//XS::XSLib C5G7Data(false, ng, nXSset, 0);
	//C5G7Data.UploadXSData(typeXS, XSSet, XSSM);
	
	MPI::Finalize();
	return EXIT_SUCCESS;
}