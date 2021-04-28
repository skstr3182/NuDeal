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

string InputManager_t::GetLine(istream& fin, const char delimiter) const
{
	string oneline;
	
	std::getline(fin, oneline, delimiter);

	std::replace(oneline.begin(), oneline.end(), SC::TAB, SC::BLANK);
#ifdef __linux__
	std::replace(oneline.begin(), oneline.end(), SC::CR, SC::BLANK);
#endif
	
	auto pos = static_cast<string::size_type>(0);

	// Treat C-style Comment

	do {
		pos = oneline.find(SC::COMMENT);
		if (pos == string::npos) break;
		auto LF = oneline.find(SC::LF, pos);
		if (LF != string::npos) oneline.erase(pos, LF - pos);
	} while (pos != string::npos);

	// Treat Fortran-style Comment

	do {
		pos = oneline.find(SC::BANG);
		if (pos == string::npos) break;
		auto LF = oneline.find(SC::LF, pos);
		if (LF != string::npos) oneline.erase(pos, LF - pos);
	} while (pos != string::npos);

	oneline.erase(std::remove(oneline.begin(), oneline.end(), SC::LF), oneline.end());

	oneline = oneline.erase(oneline.find_last_not_of(SC::BLANK) + 1);
	oneline = oneline.erase(0, oneline.find_first_not_of(SC::BLANK));

	return static_cast<string&&>(oneline);
}

string InputManager_t::GetScriptBlock(istream& in) const
{
	
	string block = "";
	bool stop = false;
	auto pos = in.tellg();

	do {
		auto pos = in.tellg();
		string oneline;
		std::getline(in, oneline);
		block += oneline;

		auto pos_LBRACE = oneline.find(SC::LBRACE);
		auto pos_RBRACE = oneline.find(SC::RBRACE);
		auto pos_SEMI = oneline.find(SC::SEMICOLON);

		stop = pos_LBRACE != string::npos;
		stop = stop && pos_RBRACE != string::npos;
		stop = stop && pos_SEMI != string::npos;
		if (stop) {
			auto m = min(pos_LBRACE, pos_RBRACE);
			m = min(m, pos_SEMI);

		}
	} while (!stop);


	return static_cast<string&&>(block);
}

}