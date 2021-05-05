#pragma once
#include "Defines.h"

namespace IO
{

// Standard Namespace
using namespace std;

// Input.h
class InputManager_t;

// Lexer.h
class Lexer_t;

// Parser.h
class Parser_t;

// Special Characters
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

// Block
const vector<string> BlockNames = {
	"GEOMETRY",
	"MATERIAL",
	"OPTION"
};

// Card
const vector<vector<string>> CardNames = {
	{ "UNITVOLUME", "UNITCOMP", "DISPLACE" },
	{ "NG", "FORMAT" },
	{ "CRITERIA" }
};

}