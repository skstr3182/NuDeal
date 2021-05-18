#include "Defines.h"
#include "UnitGeo.h"
#include "Input.h"
#include "HardCodeParam.h"

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	//IO::InputManager_t Parser;
	//std::string file = std::string(argv[1]);

	//Parser.ReadInput(file);

	//for (int i = 0; i < 100; i++){
		//cout << i + 1 << "-th iteration" << endl;
		Geometry::DebugUnitGeo();
	//}

	MPI_Finalize();

	return EXIT_SUCCESS;
}