#include "Defines.h"
#include "Input.h"

int main(int argc, char *argv[]) 
{
	MPI_Init(&argc, &argv);

	IO::InputManager_t Parser;
	std::string file = std::string(argv[1]);

	Parser.ReadInput(file);


	MPI_Finalize();

	return EXIT_SUCCESS;
}