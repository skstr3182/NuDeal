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

string InputManager_t::GetLine(istream& in, const char delimiter) const
{
	string oneline;
	
	std::getline(in, oneline, delimiter);

	std::replace(oneline.begin(), oneline.end(), SC::TAB, SC::BLANK);
#ifdef __linux__
	std::replace(oneline.begin(), oneline.end(), SC::CR, SC::BLANK);
#endif
	
	string::size_type pos = 0;

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

vector<string> InputManager_t::SplitFields(string line, const char *delimiter)
{
	vector<string> splitted;
	stringstream ss(line);
	
	string::size_type beg, pos = 0;

	while ((beg = line.find_first_not_of(delimiter, pos)) != string::npos)
	{
		pos = line.find_first_of(delimiter, beg + 1);
		splitted.push_back(line.substr(beg, pos - beg));
	}

	return static_cast<vector<string>&&>(splitted);
}

string InputManager_t::GetScriptBlock(istream& in) const
{
	string oneline;

	bool lstop = false;

	do {
		
		oneline += GetLine(in, SC::RBRACE);
		if (oneline.empty()) break;
		oneline.push_back(SC::RBRACE);

		auto lcount = std::count(oneline.begin(), oneline.end(), SC::LBRACE);
		auto rcount = std::count(oneline.begin(), oneline.end(), SC::RBRACE);

		lstop = lcount == rcount;

	} while (!lstop && !in.eof());

	return static_cast<string&&>(oneline);
}

}