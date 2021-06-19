#include "NuDEAL.h"
#include "MPI.hpp"

namespace NuDEAL
{

void Master_t::Initialize(int argc, char *argv[])
{
	MPI::Init(&argc, &argv);


}


void Master_t::Finalize()
{


	MPI::Finalize();
}

}