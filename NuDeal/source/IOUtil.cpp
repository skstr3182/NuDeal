#include "Input.h"
#include "Exception.h"

namespace IO
{

using Except = Exception_t;
using Code = Except::Code;

void InputManager_t::Uppercase(string& line) const
{
	std::transform(line.begin(), line.end(), line.begin(), ::toupper);
}

int InputManager_t::Repeat(string& field) const
{
	if (field.find('*') == string::npos) return 1;
	else {
		int pos = field.find('*');
		int rpt = stoi(field);
		field = field.substr(pos + 1, field.size() - pos - 1);
		return rpt;
	}
}

int InputManager_t::Integer(string field) const
{
	int val;

	try {
		val = stoi(field);
	}
	catch (invalid_argument) {
		Except::Abort(Code::INVALID_INTEGER, field, "Line : " + to_string(line));
	}

	if (val != stod(field)) Except::Abort(Code::INVALID_INTEGER, field, "Line : " + to_string(line));

	return val;
}

double InputManager_t::Float(string field) const
{
	double val;

	try {
		val = stod(field);
	}
	catch (std::invalid_argument) {
		Except::Abort(Code::INVALID_FLOATING_POINT, "Line : " + to_string(line));
	}

	return val;
}

bool InputManager_t::Logical(string field) const
{
	Uppercase(field);
	if (!field.compare("T")) return true;
	else if (!field.compare("TRUE")) return true;
	else if (!field.compare("F")) return false;
	else if (!field.compare("FALSE")) return false;

	Except::Abort(Code::INVALID_LOGICAL, "Line : " + to_string(line));
}

InputManager_t::Blocks InputManager_t::GetBlockID(string line) const
{
	int pos_end = line.find(BLANK, 0);
	string block = line.substr(0, pos_end);

	Uppercase(block);
	for (int i = 0; i < num_blocks; ++i) 
		if (!block.compare(BlockNames[i])) 
			return static_cast<Blocks>(i);
	
	return Blocks::INVALID;
}

template <typename T>
T InputManager_t::GetCardID(Blocks block, string line) const
{
	int pos_beg = line.find_first_not_of(BLANK);
	if (pos_beg == string::npos) return static_cast<T>(INVALID);


}

}