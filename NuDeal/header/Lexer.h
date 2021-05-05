#pragma once
#include "Defines.h"
#include "IODeclare.h"
#include <regex>

namespace IO
{

class Lexer_t
{
private :
	
	enum class TokenType
	{
		KEYWORD,
		NUMBER,
		VARIABLE,
		FUNCTION,
		COUNT,
	};
	
	vector<vector<string>> tokens = {
		{ },
		{ R"([0-9]*)" },
		{ R"([a-zA-Z][a-zA-Z0-9_]*)" },
		{ "[abcd]" },
		{ "[efgh]" }
	};

	vector<vector<regex>> v_regex;

public :

	Lexer_t();


};

}