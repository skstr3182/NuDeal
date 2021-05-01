#include "Parser.h"
#include "Exception.h"

namespace IO
{

using Except = Exception_t;
using Code = Except::Code;

string Parse_t::Uppercase(string& line)
{
	string l = line;
	std::transform(l.begin(), l.end(), l.begin(), ::toupper);
	return static_cast<string&&>(l);
}

int Parse_t::Integer(string field)
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

double Parse_t::Float(string field)
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

bool Parse_t::Logical(string field)
{
	Uppercase(field);
	if (!field.compare("T")) return true;
	else if (!field.compare("TRUE")) return true;
	else if (!field.compare("F")) return false;
	else if (!field.compare("FALSE")) return false;

	Except::Abort(Code::INVALID_LOGICAL, field);
}

string Parse_t::Trim(const string& field, const string& delimiter)
{
	string s = field;
	auto pos = s.find_first_not_of(delimiter);
	s.erase(0, pos);
	pos = s.find_last_not_of(delimiter) + 1;
	s.erase(pos);

	return static_cast<string&&>(s);
}

int Parse_t::LineCount(const string& line, size_type count)
{
	auto beg = line.begin();
	auto end = line.end();

	if (count != string::npos) end = beg + count;

	return static_cast<int>(std::count(beg, end, SC::LF));
}

string Parse_t::EraseSpace(const string& line, const string& delimiter)
{
	string s = line;
	for (const auto& i : delimiter)
		s.erase(std::remove(s.begin(), s.end(), i), s.end());
	return static_cast<string&&>(s);
}

Parse_t::size_type Parse_t::FindEndOfMacro(const string& line, size_type pos)
{
	if (line[pos] != SC::HASHTAG) return string::npos;

	size_type backslash, LF;
	size_type macro_end = pos;

	do {
		LF = line.find(SC::LF, macro_end);
		backslash = line.find(SC::BACKSLASH, macro_end);
		macro_end = LF + 1;
	} while (backslash < LF);

	return macro_end;
}

string Parse_t::ReplaceMacro(const string& line, char replace)
{
	string s = line;
	size_type macro_beg, pos = 0;

	while ((macro_beg = s.find_first_of(SC::HASHTAG, pos)) != string::npos) {
		auto macro_end = FindEndOfMacro(s, macro_beg);
		std::replace_if(s.begin() + macro_beg, s.begin() + macro_end, 
			[] (char c) { return c != SC::LF; }, replace);
		pos = macro_end;
	}

	return static_cast<string&&>(s);
}

vector<string> Parse_t::ExtractMacro(const string& line)
{
	vector<string> macro;

	size_type beg = 0, pos;

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


string Parse_t::GetLine(stringstream& in, const char delimiter)
{
	string s;

	std::getline(in, s, delimiter);

	std::replace(s.begin(), s.end(), SC::TAB, SC::BLANK);
	std::replace(s.begin(), s.end(), SC::CR, SC::BLANK);

	return static_cast<string&&>(s);
}

string Parse_t::GetBlock(stringstream& in)
{
	string section;
	auto file_pos = in.tellg();

	while (!in.eof()) {
		auto s = GetLine(in, SC::RBRACE);
		s += in.str()[in.tellg()];
		section += s;

		cout << section << endl << "123" << endl;

		if (Trim(s).empty()) continue;
		if (!IsClosed(section)) continue;

		break;
	}

	return static_cast<string&&>(section);
}

Parse_t::size_type Parse_t::FindEndPoint(const string& contents, size_type& pos)
{
	pos = contents.find_first_of(SC::LBRACE, pos);

	size_type end, beg = pos;

	while ((end = contents.find_first_of(SC::RBRACE, beg)) != string::npos) {
		if (IsClosed(contents.substr(pos, end - pos + 1))) break;
		beg = end + 1;
	}
	return end;
}

string Parse_t::GetBlock(const string& contents, size_type pos)
{
	string section;

	size_type LF_beg, LF_end = pos;

	while (LF_end != string::npos) {
		
	}

	return static_cast<string&&>(section);
}

void Parse_t::ReplaceComments(string& line)
{
	
	size_type pos;

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

vector<string> Parse_t::SplitFields(string line, const string& delimiter)
{
	vector<string> splitted;

	size_type beg, pos = 0;

	while ((beg = line.find_first_not_of(delimiter, pos)) != string::npos) {
		pos = line.find_first_of(delimiter, beg + 1);
		splitted.push_back(line.substr(beg, pos - beg));
	}

	return static_cast<vector<string>&&>(splitted);
}


void Parse_t::AreBracketsMatched(const string& contents)
{
	vector<size_type> pos;

	for (auto i = contents.begin(); i < contents.end(); ++i) {
		if (*i == SC::LBRACE)
			pos.push_back(i - contents.begin());
		else if (*i == SC::RBRACE) {
			if (pos.empty()) throw runtime_error("Not closed }");
			pos.erase(pos.end() - 1);
		}
	}

	if (!pos.empty()) throw runtime_error("Not closed }");

	pos.clear();

	for (auto i = contents.begin(); i < contents.end(); ++i) {
		if (*i == SC::LPAREN)
			pos.push_back(i - contents.begin());
		else if (*i == SC::RPAREN) {
			if (pos.empty()) throw runtime_error("Not closed )");
			pos.erase(pos.end() - 1);
		}
	}

	if (!pos.empty()) throw runtime_error("Not closed )");
}

void Parse_t::IsMacroValid(const string& contents)
{
	const set<string> directives = { "#define" };
	map<string, string> macro_table;

	size_type macro_beg, pos = 0;

	while ((macro_beg = contents.find(SC::HASHTAG, pos)) != string::npos) {
		auto macro_end = FindEndOfMacro(contents, macro_beg);
		auto macro = contents.substr(macro_beg, macro_end - macro_beg);
		auto v = ExtractMacro(macro);
		
		if (directives.find(v[0]) == directives.end()) 
			throw runtime_error("Invalid directive");
		if (macro_table.find(v[1]) != macro_table.end()) 
			throw runtime_error("Redefined macro");
		macro_table[v[1]] = v[2];
		pos = macro_end;
	}
}

void Parse_t::IsVariableCorrect(const string& contents)
{
	auto s = ReplaceMacro(contents);

	size_type beg, pos = 0;
	string line;
	set<string> variables;
	
	while ((beg = s.find_first_not_of(SC::LBRACE, pos)) != string::npos) {
		pos = s.find_first_of(SC::LBRACE, beg + 1);
		if (pos == string::npos) continue;
		auto sub = Trim(s.substr(beg, pos - beg));
		size_type p;
		if ((p = sub.find_last_of(SC::RBRACE)) != string::npos)
			sub = Trim(sub.substr(p + 1));
		if ((p = sub.find_first_of(SC::LPAREN)) != string::npos)
			sub = Trim(sub.substr(0, p));
		if ((p = sub.find(SC::COLON)) != string::npos)
			sub.erase(std::remove(sub.begin(), sub.end(), SC::BLANK), sub.end());
		cout << sub << endl;
		if (sub.find(SC::BLANK) != string::npos || sub.empty()) 
			throw runtime_error("Invalid variable name!");
		if (variables.find(sub) != variables.end()) 
			throw runtime_error("Redfeind variable!");
		variables.insert(sub);
	}

	for (auto var : variables) {
		size_type pos;
		if ((pos = var.find(SC::COLON)) != string::npos) {
			auto parent = var.substr(pos + 1);
			if (variables.find(parent) == variables.end())
				throw runtime_error("Undefined variable! : " + parent);
		}
	}

}

bool Parse_t::IsClosed(const string& s) 
{
	auto lcount = std::count(s.begin(), s.end(), SC::LBRACE);
	auto rcount = std::count(s.begin(), s.end(), SC::RBRACE);

	return (lcount == rcount) && (lcount > 0);
}


}