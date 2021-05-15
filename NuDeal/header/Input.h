#pragma once
#include "Defines.h"
#include "IODeclare.h"

namespace IO
{

using namespace std;

class InputManager_t
{

public :

	using SC = SpecialCharacters;
	using Util = Util_t;

	struct HashTree_t
	{
		using size_type = string::size_type;

		int line_info = 0, num_lines = 0;
		string name, contents;
		const HashTree_t *parent = NULL;
		//map<string, HashTree_t> children;
		vector<HashTree_t> children;
		string GetLineInfo() { return "Line : " + to_string(line_info); }
		void Make(const string& file, size_type Beg = 0, size_type End = string::npos);
		void CountLine(const string& name, const string& contents);
	};

private :



private :
	
	string file;
	string contents;

	HashTree_t HashTree;

private :

	void ExtractInput(istream& fin);
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
		double3 origin;
		vector<array<double, 10>> equations;
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
