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
		static constexpr char LPAREN = '(';
		static constexpr char RPAREN = ')';
		static constexpr char HASHTAG = '#';
	};


	struct Tree_t
	{
		size_t line_info = 0;
		string contents = "";
		map<string, string> define;
		map<string, Tree_t> children;
		const Tree_t *parent = NULL;
		string GetLineInfo() { return "Line : " + to_string(line_info); }
		void MakeInputTree(stringstream& in, size_t offset = 1);
		void MakeInputTree(const vector<string>& in, vector<string>::const_iterator iter);
	};

public :

	using SC = SpecialCharacters;
	using HashTree_t = map<string, Tree_t>;

	
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
	
	size_t line = 0;
	vector<string> line_contents;
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
	static string GetLine(stringstream& in, const char delimiter = SC::LF);
	static string GetLine(stringstream& in, const string& delimiter = "\n");
	static string GetContentsBlock(stringstream& fin);
	static void DeleteComments(string& line);
	static vector<string> SplitFields(string line, const string& delimiter);
	string GetScriptBlock(stringstream& in) const;
	/// Input Parser
	Blocks GetBlockID(string oneline) const;
	template <typename T> T GetCardID(Blocks block, string oneline) const;
	stringstream ExtractInput(istream& fin);
	/// Block Parser
	void ParseGeometryBlock(Tree_t& Tree);
	void ParseMaterialBlock(Tree_t& Tree);
	void ParseOptionBlock(Tree_t& Tree);
	/// Geometry Card Parser
	void ParseUnitVolumeCard(Tree_t& Tree);
	void ParseUnitCompCard(Tree_t& Tree);

public :

	void ReadInput(string file);

public :
	
	struct UnitVolume_t
	{
		string origin = "0, 0, 0";
		vector<string> equations;
	};

	struct UnitComp_t
	{
		string origin = "0, 0, 0";
		string background;
		vector<string> unitvols;
		vector<vector<string>> displace;
	};

private :

	map<string, UnitVolume_t> unitVolumes;
	map<string, UnitComp_t> unitComps;

};


}
