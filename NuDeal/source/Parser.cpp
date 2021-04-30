#include "Parser.h"
#include "Exception.h"

namespace IO
{

using Except = Exception_t;
using Code = Except::Code;

string Parser_t::Uppercase(string& line)
{
	string l = line;
	std::transform(l.begin(), l.end(), l.begin(), ::toupper);
	return static_cast<string&&>(l);
}

int Parser_t::Integer(string field)
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

double Parser_t::Float(string field)
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

bool Parser_t::Logical(string field)
{
	Uppercase(field);
	if (!field.compare("T")) return true;
	else if (!field.compare("TRUE")) return true;
	else if (!field.compare("F")) return false;
	else if (!field.compare("FALSE")) return false;

	Except::Abort(Code::INVALID_LOGICAL, field);
}

string Parser_t::Trim(const string& field, const string& delimiter)
{
	string s = field;
	auto pos = s.find_first_not_of(delimiter);
	s.erase(0, pos);
	pos = s.find_last_not_of(delimiter) + 1;
	s.erase(pos);

	return static_cast<string&&>(s);
}

size_t Parser_t::LineCount(const string& line)
{
	return static_cast<size_t>(std::count(line.begin(), line.end(), SC::LF));
}

string Parser_t::EraseSpace(const string& line, const string& delimiter)
{
	string s = line;
	for (const auto& i : delimiter)
		s.erase(std::remove(s.begin(), s.end(), i), s.end());
	return static_cast<string&&>(s);
}

string Parser_t::GetLine(stringstream& in, const char delimiter)
{
	string s;

	std::getline(in, s, delimiter);

	std::replace(s.begin(), s.end(), SC::TAB, SC::BLANK);
	std::replace(s.begin(), s.end(), SC::CR, SC::BLANK);

	return static_cast<string&&>(s);
}

string Parser_t::GetBlock(stringstream& in)
{
	string section;

	while (!in.eof()) {
		auto s = GetLine(in, SC::LF) + string(1, SC::LF);
		if (Trim(s).empty()) {
			section += s;
			continue;
		}
		
		NullifyMacro(s, in);
		
		section += s;

		if (!IsClosed(section)) continue;

		auto pos = in.tellg();
		decltype(pos) backspace = s.size() - s.find_last_of(SC::RBRACE);
		in.seekg(pos - backspace + 1);

		cout << in.str().substr(0, in.tellg()) << endl;

		break;
	}

	DeleteComments(section);

	return static_cast<string&&>(section);
}

void Parser_t::DeleteComments(string& line)
{
	
	string::size_type pos;

	// C-style Comment

	do {
		pos = line.find(SC::COMMENT);
		if (pos == string::npos) break;
		auto LF = line.find(SC::LF, pos);
		line.replace(pos, LF - pos, string(1, SC::BLANK));
	} while (pos != string::npos);

	// Fortran-style Comment

	do {
		pos = line.find(SC::BANG);
		if (pos == string::npos) break;
		auto LF = line.find(SC::LF, pos);
		line.replace(pos, LF - pos, string(1, SC::BLANK));
	} while (pos != string::npos);

}

vector<string> Parser_t::SplitFields(string line, const string& delimiter)
{
	vector<string> splitted;

	string::size_type beg, pos = 0;

	while ((beg = line.find_first_not_of(delimiter, pos)) != string::npos) {
		pos = line.find_first_of(delimiter, beg + 1);
		splitted.push_back(line.substr(beg, pos - beg));
	}

	return static_cast<vector<string>&&>(splitted);
}

vector<string> Parser_t::ExtractMacro(const string& line)
{
	vector<string> macro;

	string::size_type beg = 0, pos;

	pos = line.find(SC::BLANK);
	macro.push_back(Trim(line.substr(0, pos)));

	beg = pos + 1;

	if ((pos = line.find(SC::RPAREN, beg)) != string::npos) {
		macro.push_back(Trim(line.substr(beg, pos - beg)));
		macro.push_back(Trim(line.substr(pos + 1)));
	}
	else {
		pos = line.find(SC::BLANK, beg);
		macro.push_back(Trim(line.substr(beg, pos - beg)));
		macro.push_back(Trim(line.substr(pos + 1)));
	}

	return static_cast<vector<string>&&>(macro);
}


bool Parser_t::AreBracketsMatched(const string& contents)
{
	auto beg = contents.begin(), end = contents.end();

	auto lcount = std::count(beg, end, SC::LBRACE);
	auto rcount = std::count(beg, end, SC::RBRACE);

	if (lcount != rcount) return false;

	lcount = std::count(beg, end, SC::LPAREN);
	rcount = std::count(beg, end, SC::RPAREN);

	if (lcount != rcount) return false;

	return true;
}

bool Parser_t::IsMacroValid(const string& contents)
{
	const set<string> Directive = { "#define" };
	map<string, string> defined;

	string::size_type macro_pos, pos = 0;

	while ((macro_pos = contents.find(SC::HASHTAG, pos)) != string::npos) {
		auto macro_end = macro_pos;
		string::size_type backslash, LF;

		do {
			LF = contents.find(SC::LF, macro_end);
			backslash = std::find(contents.begin() + macro_end,
				contents.begin() + LF, SC::BACKSLASH) - contents.begin();
			macro_end = LF + 1;
		} while (backslash < LF);

		auto macro = contents.substr(macro_pos, macro_end - macro_pos);
		auto v = ExtractMacro(macro);
		
		if (Directive.find(v.front()) == Directive.end()) return false;
		if (defined.find(v[1]) != defined.end()) return false;
		defined[v[1]] = v[2];
		pos = LF;
	}

	return true;
}

void Parser_t::NullifyMacro(string& s, stringstream& in)
{
	if (Trim(s).front() == SC::HASHTAG) {
		while (Trim(s).back() == SC::BACKSLASH) {
			s += GetLine(in, SC::LF) + string(1, SC::LF);
		}
		std::replace_if(s.begin(), s.end(), [](char c) { return c != SC::LF; }, SC::BLANK);
	}
}

bool Parser_t::IsClosed(string& s) 
{
	size_t count = 0;
	bool lopened = false;
	auto iter = s.begin();

	for (iter = s.begin(); iter != s.end(); ++iter) {
		if (*iter == SC::LBRACE) {
			++count; lopened = true;
		}
		else if (*iter == SC::RBRACE) {
			--count;
		}
		if (lopened && !count) break;
	}

	if (iter < s.end() - 1)
		s.replace(iter + 1, s.end(), string(1, SC::BLANK));
	return iter != s.end();
}


}