#pragma once
#include "Defines.h"
#include "IODeclare.h"
#include "IOUtil.h"

namespace IO
{

class Parser_t
{
public :

	using Util = Util_t;
	using Except = Exception_t;
	using SC = SpecialCharacters;

private :
	
	static constexpr char operators[] = {
		SC::Caret,
		SC::Asterisk,
		SC::Slash,
		SC::Plus,
		SC::Minus,
		SC::LeftAngle,
		SC::RightAngle
	};
	static constexpr int precedence[] = {
		4, 
		3, 
		3, 
		2, 
		2, 
		1, 
		1
	};

	static const regex numeric;
	static const regex variable;
	static const regex ops;

	static bool IsDigit(char c) noexcept;
	static bool IsVariable(char c) noexcept;
	static bool IsOperator(char c) noexcept;
	static void TreatNumeric(string::const_iterator& pos);
	static void TreatVariable(string::const_iterator& pos);

	static vector<string> Tokenize(const string& line) noexcept;

public :
	
	static array<double, 10> ParseEquation(const string& line);

};


}