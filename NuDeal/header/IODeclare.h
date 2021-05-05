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
	static constexpr char LeftParen = '(';
	static constexpr char RightParen = ')';
	static constexpr char LeftBracket = '[';
	static constexpr char RightBracket = ']';
	static constexpr char LeftBrace = '{';
	static constexpr char RightBrace = '}';
	static constexpr char LeftAngle = '<';
	static constexpr char RightAngle = '>';
	static constexpr char Equal = '=';
	static constexpr char Plus = '+';
	static constexpr char Minus = '-';
	static constexpr char Asterisk = '*';
	static constexpr char Slash = '/';
	static constexpr char Caret = '^';
	static constexpr char Hash = '#';
	static constexpr char Dot = '.';
	static constexpr char Comma = ',';
	static constexpr char SemiColon = ';';
	static constexpr char Colon = ':';
	static constexpr char SingleQuote = '\'';
	static constexpr char DoubleQuote = '\"';
	static constexpr char BackSlash = '\\';
	static constexpr char Blank = ' ';
	static constexpr char Tab = '\t';
	static constexpr char CR = '\r';
	static constexpr char LF = '\n';
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