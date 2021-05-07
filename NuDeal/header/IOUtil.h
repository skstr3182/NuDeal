#pragma once
#include "Defines.h"
#include "IODeclare.h"

namespace IO
{
// Special Characters
struct SpecialCharacters
{
	static constexpr char LeftParen = '(';
	static constexpr char RightParen = ')';
	static constexpr char LeftBracket = '[';
	static constexpr char RightBracket = ']';
	static constexpr char LeftBrace = '{';
	static constexpr char RightBrace = '}';
	static constexpr char LeftAngle = '<';
	static constexpr char RightAngle = '>';
	static constexpr char Equal = '=';
	static constexpr char Plus = '+';
	static constexpr char Minus = '-';
	static constexpr char Asterisk = '*';
	static constexpr char Slash = '/';
	static constexpr char Caret = '^';
	static constexpr char Hash = '#';
	static constexpr char Dot = '.';
	static constexpr char Comma = ',';
	static constexpr char SemiColon = ';';
	static constexpr char Colon = ':';
	static constexpr char BackSlash = '\\';
	static constexpr char Blank = ' ';
	static constexpr char Tab = '\t';
	static constexpr char CR = '\r';
	static constexpr char LF = '\n';
};

// Block
const vector<string> BlockNames = {
	"GEOMETRY",
	"MATERIAL",
	"OPTION"
};

// Card
const vector<vector<string>> CardNames = {
	{ "UNITVOLUME", "UNITCOMP", "DISPLACE" },
	{ "NG", "FORMAT" },
	{ "CRITERIA" }
};

enum class Blocks {
	GEOMETRY,
	MATERIAL,
	OPTION,
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

class Util_t
{
public :

	using SC = SpecialCharacters;
	using size_type = string::size_type;

	// String Parsing Utility
	/// Type Conversion
	static string Uppercase(string& line);
	static int Integer(string field);
	static double Float(string field);
	static bool Logical(string field);
	/// String Manipulation
	static string Trim(const string& field, const string& delimiter = "\n ");
	static int LineCount(const string& line, size_type count = string::npos);
	static string EraseSpace(const string& line, const string& delimiter = "\n ");
	/// Read Input
	static string GetLine(stringstream& in, const char delimiter = SC::LF);
	static size_type FindEndPoint(const string& contents, size_type& pos);
	static vector<string> SplitFields(string line, const string& delimiter);
	
	// Check Synax Error
	static bool IsClosed(const string& s);

	static Blocks GetBlockID(string line)
	{
		int pos_end = line.find(SC::Blank, 0);
		string block = line.substr(0, pos_end);

		block = Uppercase(block);
		for (int i = 0; i < BlockNames.size(); ++i)
			if (!block.compare(BlockNames[i]))
				return static_cast<Blocks>(i);
		return Blocks::INVALID;
	}

	template <typename T> 
	static T GetCardID(Blocks block, string line)
	{
		static constexpr T INVALID = static_cast<T>(-1);

		int pos_beg = line.find_first_not_of(SC::Blank);
		if (pos_beg == string::npos) return INVALID;

		int pos_end = line.find(SC::Blank, pos_beg);
		string card = line.substr(pos_beg, pos_end - pos_beg);

		card = Uppercase(card);
		int b = static_cast<int>(block);
		for (int i = 0; i < CardNames[b].size(); ++i)
			if (!card.compare(CardNames[b][i]))
				return static_cast<T>(i);
		return INVALID;
	}
};



}