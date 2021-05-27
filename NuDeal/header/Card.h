#pragma once
#include "Defines.h"

namespace IO
{

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

template <typename T>
class InputCard_t
{
public:
	static const vector<string> names;
	static T GetID(const string& line);
};

const vector<string> InputCard_t<Blocks>::names = {
	"GEOMETRY",
	"MATERIAL",
	"OPTION"
};
const vector<string> InputCard_t<GeometryCards>::names = {
	"UNITVOLU",
	"UNITCOMP",
	"DISPLACE"
};
const vector<string> InputCard_t<MacroXsCards>::names = {
	"NG",
	"FORMAT"
};

template <typename T>
T InputCard_t<T>::GetID(const string& line)
{
	static constexpr T INVALID = static_cast<T>(-1);

	string card;
	std::transform(line.begin(), line.end(), card.begin(), ::toupper);
	auto iter = std::find(names.begin(), names.end(), card);

	if (iter == names.end()) return INVALID;
	else return static_cast<T>(iter - names.begin());
}

}