#pragma once
#include "Defines.h"

namespace IO
{

using namespace std;

class InputManager_t
{
private :

	struct SpecialCharacters
	{
		static constexpr char BLANK = ' ';
		static constexpr char BANG = '!';
		static constexpr char BRACKET = '>';
		static constexpr char TAB = '\t';
		static constexpr char CR = '\r';
		static constexpr char LF = '\n';
		static constexpr char LBRACE = '{';
		static constexpr char RBRACE = '}';
		static constexpr char SEMICOLON = ';';
		static constexpr char COMMENT[] = "//";
		static constexpr char DAMPERSAND[] = "&&";
	};

public :

	using SC = SpecialCharacters;
	
private :

	static constexpr int num_blocks = 2;
	static constexpr int num_cards = 30;

	const string BlockNames[num_blocks] =
	{
		"GEOMETRY",
		"MACROXS"
	};
	const string CardNames[num_blocks][num_cards] = 
	{
		{ "UNITVOLUME", "UNITCOMP", "DISPLACE" },
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
	unsigned int line = 0;
	static const int INVALID = -1;

	// IO Utility
	/// Parser
	void Uppercase(string& line) const;
	int Repeat(string& field) const;
	int Integer(string field) const;
	double Float(string field) const;
	bool Logical(string field) const;
	unsigned int CountCurrentLine(istream& in) const;
	string GetLine(istream& fin, const char delimiter = SC::LF) const;
	string GetScriptBlock(istream& in) const;
	// Input Parser
	Blocks GetBlockID(string oneline) const;
	template <typename T> T GetCardID(Blocks block, string oneline) const;
	stringstream ExtractInput(istream& fin, string& contents) const;

public :

	void ReadInput(string file);

};


}
