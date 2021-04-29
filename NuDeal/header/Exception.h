#pragma once
#include "Defines.h"

namespace IO
{

class Exception_t
{
public :

	enum class Code
	{
		FILE_NOT_FOUND,
		INVALID_INTEGER,
		INVALID_FLOATING_POINT,
		INVALID_LOGICAL,
		INVALID_INPUT_BLOCK,
		INVALID_INPUT_CARD,
		SEMICOLON_MISSED,
		UNSPECIFIED_ERROR,
		COUNTS,
	};

	static constexpr char *error_messages[] = {
		"File not found!",
		"Invalid integer expression!",
		"Invalid floating point expression!",
		"Invalid logical exrpession!",
		"Invalid input block!",
		"Invalid input card!",
		"Semicolon missed!",
		"Unspecified error!"
	};

	static size_t CountCurrentLine(istream& in)
	{
		size_t pos = in.tellg();
		
		in.clear(stringstream::goodbit);
		in.seekg(0, ios::beg);

		size_t count = 0;

		for (size_t i = 0; i < pos; ++i) {
			if (in.get() == '\n') ++count;
		}

		return count;
	}

	static void Abort(string message)
	{
		int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if (rank) return;

		cout << "                                                                              " << endl;
		cout << "========================== NuDEAL Exception Handler ==========================" << endl;
		cout << "                                                                              " << endl;
		cout << message << endl;
		cout << "                                                                              " << endl;
		cout << "==============================================================================" << endl;
		cout << "                                                                              " << endl;

		exit(EXIT_FAILURE);
	}

	static void Abort(Code error, string info = "", string message = "")
	{
		int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if (rank) return;

		cout << "                                                                              " << endl;
		cout << "========================== NuDEAL Exception Handler ==========================" << endl;
		cout << "                                                                              " << endl;

		int e = static_cast<int>(error);

		cout << info << endl;
		cout << "Exception : " << error_messages[e];
		cout << " " << message << endl;

		cout << "                                                                              " << endl;
		cout << "==============================================================================" << endl;
		cout << "                                                                              " << endl;

		exit(EXIT_FAILURE);
	}

};


}
