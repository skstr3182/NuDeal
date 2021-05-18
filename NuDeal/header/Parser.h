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
	using siterator = string::const_iterator;

	static const initializer_list<char> operators;

	enum class Type
	{
		Variable,
		Function,
		Number,
		UnaryOp,
		BinaryOp,
		LeftParen,
		RightParen,
	};

	Token_t(Type t, const string& s, int prec = -1, bool ra = false) noexcept;
	static bool IsNumeric(siterator s) noexcept { return isdigit(*s); }
	static bool IsFunction(siterator s, siterator end) noexcept;
	static bool IsVariable(siterator s, siterator end) noexcept;
	static bool IsParen(siterator s) noexcept 
	{ return *s == SC::LeftParen || *s == SC::RightParen; }
	static bool IsOperator(siterator s) noexcept 
	{ return std::find(operators.begin(), operators.end(), *s) != operators.end(); }

	static Token_t GetNumeric(siterator& pos, const string& container);
	static Token_t GetFunction(siterator& pos, const string& container);
	static Token_t GetVariable(siterator& pos);
	static Token_t GetParenthesis(siterator& pos);
	static Token_t GetOperator(siterator& pos, const string& container);

	Type TokenType() const noexcept { return type; }
	bool RightAssociative() const noexcept { return rightAssociative; }
	int Precedence() const noexcept { return precedence; }
	string GetString() const noexcept { return str; }

private :

	const Type type;
	const string str;
	const int precedence;
	const bool rightAssociative;

public :

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

	using Variable_t = map<string, double>;
	static Variable_t Aggregate(const deque<Token_t>& queue) noexcept;
	static Variable_t TreatAdd(const Variable_t& lhs, const Variable_t& rhs);
	static Variable_t TreatSub(const Variable_t& lhs, const Variable_t& rhs);
	static Variable_t TreatMult(const Variable_t& lhs, const Variable_t& rhs);
	static Variable_t TreatDiv(const Variable_t& lhs, const Variable_t& rhs);
	static Variable_t TreatPow(const Variable_t& lhs, const Variable_t& rhs);

public :
	
	static array<double, 10> ParseEquation(const string& line);

};

}