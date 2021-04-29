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
		static constexpr char COLON = ':';
		static constexpr char SEMICOLON = ';';
		static constexpr char COMMENT[] = "//";
		static constexpr char DAMPERSAND[] = "&&";
		static constexpr char RDBRACKET[] = ">>";
	};

	struct InputTree_t
	{
		size_t line_begin = 0;
		string contents = "";
		map<string, InputTree_t> children;
		string GetLineInfo() { return "Line : " + to_string(line_begin); }
	};

	using Tree_t = map<string, InputTree_t>;

public :

	using SC = SpecialCharacters;
	
private :

	static constexpr int num_blocks = 5;
	static constexpr int num_cards = 30;

	const string BlockNames[num_blocks] =
	{
		"GEOMETRY",
		"MATERIAL",
		"OPTION"
	};
	const string CardNames[num_blocks][num_cards] = 
	{
		{ "UNITVOLUME", "UNITCOMP", "DISPLACE" },
		{ "NG", "FORMAT" }
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

private :
	
	string contents;
	size_t line = 0;
	static constexpr int INVALID = -1;
	Tree_t TreeHead;

	// IO Utility
	/// Parser
	void Uppercase(string& line) const;
	int Repeat(string& field) const;
	int Integer(string field) const;
	double Float(string field) const;
	bool Logical(string field) const;
	static string Trim(const string& field, const string& delimiter = "\n ");
	static size_t LineCount(const string& line);
	static string EraseSpace(const string& line, const string& delimiter = "\n ");
	string GetLine(istream& fin, const char delimiter = SC::LF) const;
	void DeleteComments(string& line) const;
	vector<string> SplitFields(string line, const string& delimiter);
	string GetScriptBlock(istream& in) const;
	/// Input Parser
	Blocks GetBlockID(string oneline) const;
	template <typename T> T GetCardID(Blocks block, string oneline) const;
	stringstream ExtractInput(istream& fin);
	void ConfigureTree(stringstream& in, Tree_t& Tree, size_t offset = 1);
	/// Block Parser
	void ParseGeometryBlock(InputTree_t& Tree);
	void ParseMaterialBlock(InputTree_t& Tree);
	void ParseOptionBlock(InputTree_t& Tree);
	/// Geometry Card Parser
	void ParseUnitVolumeCard(InputTree_t& Tree);
	void ParseUnitCompCard(InputTree_t& Tree);

public :

	void ReadInput(string file);

public :
	
	struct UnitVolume_t
	{
		string origin = "(0, 0, 0)";
		vector<string> equations;
	};

	struct UnitComp_t
	{
		string origin = "(0, 0, 0)";
		string background;
		vector<string> unitvols;
		vector<vector<string>> displace;
	};

private :

	map<string, UnitVolume_t> unitVolumes;
	map<string, UnitComp_t> unitComps;

};


}
