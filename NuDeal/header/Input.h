#pragma once
#include "Defines.h"
#include "IODeclare.h"

namespace IO
{

using namespace std;

class InputManager_t
{

public :

	struct HashTree_t
	{
		using size_type = string::size_type;
		int line_info = 0, num_lines = 0;
		string contents = "";
		map<string, map<string, string>> macro;
		const HashTree_t *parent = NULL;
		map<string, HashTree_t> children;
		string GetLineInfo() { return "Line : " + to_string(line_info); }
		void ProcessMacro(const string& contents);
		void Make(const string& file, size_type Beg = 0, size_type End = string::npos);
		void CountLine(const string& name, const string& contents);
	};

private :

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
	
	string file;
	string original, modified;

	Lexer_t *Lexer;

	HashTree_t HashTree;

private :

	/// Input Parser
	Blocks GetBlockID(string oneline) const;
	template <typename T> T GetCardID(Blocks block, string oneline) const;
	void ExtractInput(istream& fin);
	void InspectSyntax(const string& contents);
	/// Block Parser
	void ParseGeometryBlock(HashTree_t& Tree);
	void ParseMaterialBlock(HashTree_t& Tree);
	void ParseOptionBlock(HashTree_t& Tree);
	/// Geometry Card Parser
	void ParseUnitVolumeCard(HashTree_t& Tree);
	void ParseUnitCompCard(HashTree_t& Tree);

	// Preprocessing
	void Preprocess();

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
