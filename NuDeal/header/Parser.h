#pragma once
#include "Defines.h"
#include "IODeclare.h"
#include "IOUtil.h"

namespace IO
{

class Token_t
{
public :

	using SC = SpecialCharacters;
	using Except = Exception_t;
	using siterator = string::iterator;

	static const initializer_list<char> operators;

	enum class Type
	{
		Variable,
		Function,
		Number,
		Operator,
		LeftParen,
		RightParen,
		Angle,
		Invalid = -1
	};

	Token_t(Type t, const string& s, int prec = -1, bool ra = false) noexcept;
	static bool IsNumeric(siterator s) noexcept { return isdigit(*s); }
	static bool IsFunction(siterator s, siterator end) noexcept;
	static bool IsVariable(siterator s, siterator end) noexcept;
	static bool IsParen(siterator s) noexcept 
	{ return *s == SC::LeftParen || *s == SC::RightParen; }
	static bool IsOperator(siterator s) noexcept 
	{ return std::find(operators.begin(), operators.end(), *s) != operators.end(); }
	static bool IsAngle(siterator s) noexcept
	{ return *s == SC::LeftAngle || *s == SC::RightAngle; }

	static Token_t GetNumeric(siterator& pos, siterator end);
	static Token_t GetFunction(siterator& pos, siterator end);
	static Token_t GetVariable(siterator& pos);
	static Token_t GetParenthesis(siterator& pos);
	static Token_t GetOperator(siterator& pos);
	static Token_t GetAngle(siterator& pos);

	Type TokenType() const noexcept { return type; }
	bool RightAssociative() const noexcept { return rightAssociative; }
	int Precedence() const noexcept { return precedence; }
	string GetString() const noexcept { return str; }

private :

	const Type type;
	const string str;
	const int precedence;
	const bool rightAssociative;

	// For Numeric
	double v = 0.0;
	// For Variable
	double coefficient = 0.0; int exponent = 0;
	// For Function
	double (*Do) (double) = NULL;
};

class Parser_t
{
public :

	using Util = Util_t;
	using Except = Exception_t;
	using SC = SpecialCharacters;

private :

	static deque<Token_t> Tokenize(const string& line) noexcept;
	static deque<Token_t> ShuntingYard(const deque<Token_t>& tokens) noexcept;
	static void Calculator(deque<string>& tokens) noexcept;

public :
	
	static array<double, 10> ParseEquation(const string& line);

};


}