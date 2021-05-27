#pragma once
#include "Defines.h"
#include "IODeclare.h"

namespace IO
{

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
		vector<HashTree_t> children;
		string GetLineInfo() { return "Line : " + to_string(line_info); }
		void Make(const string& file, size_type Beg = 0, size_type End = string::npos);
		void CountLine(const string& name, const string& contents);
	};

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

	void ReadInput(const string& file);

public :
	
	using Equation_t = array<double, 10>;

	struct UnitVolume_t
	{
		double3 origin = {0.0, 0.0, 0.0};
		vector<Equation_t> equations;
	};

	struct Displace_t
	{
		enum class Type { Rotation, Translation };
		enum class Axis { X, Y, Z, INVALID = -1 };
		Type type = Type::Translation;
		Axis axis = Axis::INVALID;
		array<double, 3> move = {0.0, };

		bool IsRotation() const noexcept { return type == Type::Rotation; }
		bool IsTranslation() const noexcept { return type == Type::Translation; }
		Axis GetAxis() const noexcept { return axis; }
		const array<double, 3>& GetTrans() const noexcept { return move; }
		double GetRot() const noexcept { return move[static_cast<int>(axis)]; }
	};

	struct UnitComp_t
	{
		double3 origin = {0.0, 0.0, 0.0};
		vector<string> unitvols;
		vector<vector<Displace_t>> displace;
	};

	struct _UnitComp_t
	{
		double3 origin = {0.0, 0.0, 0.0};
		string unitvol, material;
	};

	struct UnitCell_t
	{
		enum class Type {General, Rect, Hex};
		Type type = Type::General;
		double3 origin = {0.0, 0.0, 0.0};
		double2 pitch = {0.0, 0.0};
		vector<vector<string>> array;
	};

private :

	map<string, UnitVolume_t> unitVolumes;
	map<string, UnitComp_t> unitComps;

public :

	const auto& GetUnitVolumeInfo() const { return unitVolumes; }
	const auto& GetUnitCompInfo() const { return unitComps; }

};


}
