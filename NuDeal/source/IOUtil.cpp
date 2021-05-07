#include "IOUtil.h"
#include "Exception.h"

namespace IO
{

using Except = Exception_t;
using Code = Except::Code;

string Util_t::Uppercase(string& line)
{
	string l = line;
	std::transform(l.begin(), l.end(), l.begin(), ::toupper);
	return static_cast<string&&>(l);
}

int Util_t::Integer(string field)
{
	int val;

	try {
		val = stoi(field);
	}
	catch (invalid_argument) {
		Except::Abort(Code::INVALID_INTEGER, field);
	}

	if (val != stod(field)) Except::Abort(Code::INVALID_INTEGER, field);

	return val;
}

double Util_t::Float(string field)
{
	double val;

	try {
		val = stod(field);
	}
	catch (std::invalid_argument) {
		Except::Abort(Code::INVALID_FLOATING_POINT, field);
	}

	return val;
}

bool Util_t::Logical(string field)
{
	Uppercase(field);
	if (!field.compare("T")) return true;
	else if (!field.compare("TRUE")) return true;
	else if (!field.compare("F")) return false;
	else if (!field.compare("FALSE")) return false;

	Except::Abort(Code::INVALID_LOGICAL, field);
}

string Util_t::Trim(const string& field, const string& delimiter)
{
	string s = field;
	auto pos = s.find_first_not_of(delimiter);
	s.erase(0, pos);
	pos = s.find_last_not_of(delimiter) + 1;
	s.erase(pos);

	return static_cast<string&&>(s);
}

int Util_t::LineCount(const string& line, size_type count)
{
	auto beg = line.begin();
	auto end = line.end();

	if (count != string::npos) end = beg + count;

	return static_cast<int>(std::count(beg, end, SC::LF));
}

string Util_t::EraseSpace(const string& line, const string& delimiter)
{
	string s = line;
	for (const auto& i : delimiter)
		s.erase(std::remove(s.begin(), s.end(), i), s.end());
	return static_cast<string&&>(s);
}

string Util_t::GetLine(stringstream& in, const char delimiter)
{
	string s;

	std::getline(in, s, delimiter);

	std::replace(s.begin(), s.end(), SC::Tab, SC::Blank);
	std::replace(s.begin(), s.end(), SC::CR, SC::Blank);

	return static_cast<string&&>(s);
}

Util_t::size_type Util_t::FindEndPoint(const string& contents, size_type& pos)
{
	pos = contents.find_first_of(SC::LeftBrace, pos);

	size_type end, beg = pos;

	while ((end = contents.find_first_of(SC::RightBrace, beg)) != string::npos) {
		if (IsClosed(contents.substr(pos, end - pos + 1))) break;
		beg = end + 1;
	}
	return end;
}

vector<string> Util_t::SplitFields(string line, const string& delimiter)
{
	vector<string> splitted;

	size_type beg, pos = 0;

	while ((beg = line.find_first_not_of(delimiter, pos)) != string::npos) {
		pos = line.find_first_of(delimiter, beg + 1);
		splitted.push_back(line.substr(beg, pos - beg));
	}

	return static_cast<vector<string>&&>(splitted);
}

bool Util_t::IsClosed(const string& s) 
{
	auto lcount = std::count(s.begin(), s.end(), SC::LeftBrace);
	auto rcount = std::count(s.begin(), s.end(), SC::RightBrace);

	return (lcount == rcount) && (lcount > 0);
}

}