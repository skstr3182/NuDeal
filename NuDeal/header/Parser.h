#pragma once
#include "Defines.h"

namespace IO
{

using namespace std;

struct SpecialCharacters
{
	static constexpr char BLANK = ' ';
	static constexpr char BANG = '!';
	static constexpr char BRACKET = '>';
	static constexpr char TAB = '\t';
	static constexpr char CR = '\r';
	static constexpr char LF = '\n';
	static constexpr char LBRACE = '{';
	static constexpr char RBRACE = '}';
	static constexpr char COLON = ':';
	static constexpr char SEMICOLON = ';';
	static constexpr char COMMENT[] = "//";
	static constexpr char DAMPERSAND[] = "&&";
	static constexpr char RDBRACKET[] = ">>";
	static constexpr char LPAREN = '(';
	static constexpr char RPAREN = ')';
	static constexpr char HASHTAG = '#';
	static constexpr char BACKSLASH = '\\';
};

class Parser_t
{
public :

	using SC = SpecialCharacters;
	
	// String Parsing Utility
	static string Uppercase(string& line);
	static int Integer(string field);
	static double Float(string field);
	static bool Logical(string field);
	static string Trim(const string& field, const string& delimiter = "\n ");
	static size_t LineCount(const string& line);
	static string EraseSpace(const string& line, const string& delimiter = "\n ");
	static string GetLine(stringstream& in, const char delimiter = SC::LF);
	static string GetBlock(stringstream& fin);
	static void DeleteComments(string& line);
	static vector<string> SplitFields(string line, const string& delimiter);
	static vector<string> ExtractMacro(const string& line);
	
	// Check Synax Error
	static bool AreBracketsMatched(const string& contents);
	static bool IsMacroValid(const string& contents);

private :

	static void NullifyMacro(string& s, stringstream& in);
	static bool IsClosed(string& s);

};



}