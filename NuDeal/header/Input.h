#pragma once
#include "Defines.h"

namespace IO
{

using namespace std;

class InputManager_t
{

public :

	struct Tree_t
	{
		size_t line_info = 0;
		string contents = "";
		map<string, string> define;
		map<string, Tree_t> children;
		const Tree_t *parent = NULL;
		string GetLineInfo() { return "Line : " + to_string(line_info); }
		void ProcessMacro(stringstream& in);
		void Make(stringstream& in, size_t offset = 0);
	};

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
	
	string file, contents;
	static constexpr int INVALID = -1;
	Tree_t TreeHead;

private :

	/// Input Parser
	Blocks GetBlockID(string oneline) const;
	template <typename T> T GetCardID(Blocks block, string oneline) const;
	stringstream ExtractInput(istream& fin);
	void InspectSyntax(stringstream& file);
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
