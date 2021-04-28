#pragma once
#include "Defines.h"

namespace IO
{

using namespace std;

class InputManager_t
{
public :

	static const char BLANK = ' ';
	static const char BANG = '!';
	static const char BRACKET = '>';
	static const char TAB = '\t';
	static const char CR = '\r';
	static const char LBRACE = '{';
	static const char RBRACE = '}';

	static const int num_blocks = 2;
	static const int num_cards = 30;

	const string BlockNames[num_blocks] =
	{
		"GEOMETRY",
		"MACROXS"
	};
	const string CardNames[num_blocks][num_cards] = 
	{
		{ "UNITVOLUME", "UNITCOMP", "DISAPLCE" },
		{ "NG", "FORMAT" }
	};

	enum class Blocks {
		GEOMETRY,
		MACROXS,
		INVALID = -1
	};

	enum class GeometryCards {
		UNITVOLUME,
		UNITCOMP,
		DISPLACE,
		INVALID = -1
	};

	enum class MacroXsCards {
		NG,
		FORMAT,
		INVALID = -1
	};

private :

	string contents;
	vector<string> lines;
	unsigned int line = 0;
	static const int INVALID = -1;

	// IO Utility
	/// Parser
	void Uppercase(string& line) const;
	int Repeat(string& field) const;
	int Integer(string field) const;
	double Float(string field) const;
	bool Logical(string field) const;
	// Input Parser
	Blocks GetBlockID(string oneline) const;
	template <typename T>
	T GetCardID(Blocks block, string oneline) const;

public :

	void ReadInput(string file);

};


}
