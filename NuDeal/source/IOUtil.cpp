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

string InputManager_t::Trim(const string& field, const string& delimiter)
{
	string s = field;
	auto pos = s.find_first_not_of(delimiter);
	s.erase(0, pos);
	pos = s.find_last_not_of(delimiter) + 1;
	s.erase(pos);

	return static_cast<string&&>(s);
}

size_t InputManager_t::LineCount(const string& line)
{
	return static_cast<size_t>(std::count(line.begin(), line.end(), SC::LF));
}

string InputManager_t::EraseSpace(const string& line, const string& delimiter)
{
	string s = line;
	for (const auto& i : delimiter)
		s.erase(std::remove(s.begin(), s.end(), i), s.end());
	return static_cast<string&&>(s);
}

string InputManager_t::GetLine(istream& in, const char delimiter) const
{
	string oneline;
	
	std::getline(in, oneline, delimiter);

	std::replace(oneline.begin(), oneline.end(), SC::TAB, SC::BLANK);
	std::replace(oneline.begin(), oneline.end(), SC::CR, SC::BLANK);

	DeleteComments(oneline);

	return static_cast<string&&>(oneline);
}

void InputManager_t::DeleteComments(string& line) const
{
	
	string::size_type pos = 0;

	// C-style Comment

	do {
		pos = line.find(SC::COMMENT);
		if (pos == string::npos) break;
		auto LF = line.find(SC::LF, pos);
		line.erase(pos, LF - pos);
	} while (pos != string::npos);

	// Fortran-style Comment

	do {
		pos = line.find(SC::BANG);
		if (pos == string::npos) break;
		auto LF = line.find(SC::LF, pos);
		line.erase(pos, LF - pos);
	} while (pos != string::npos);

}

vector<string> InputManager_t::SplitFields(string line, const string& delimiter)
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
		auto lcount = std::count(oneline.begin(), oneline.end(), SC::LBRACE);

		if (lcount == 0) break;

		oneline.push_back(SC::RBRACE);

		auto rcount = std::count(oneline.begin(), oneline.end(), SC::RBRACE);

		lstop = lcount == rcount;

	} while (!lstop && !in.eof());

	return static_cast<string&&>(oneline);
}

}