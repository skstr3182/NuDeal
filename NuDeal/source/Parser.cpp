#include "Parser.h"
#include "Exception.h"
// https://gist.github.com/t-mat/b9f681b7591cdae712f6

namespace IO
{

const initializer_list<char> Token_t::operators = {
	SC::Caret,
	SC::Asterisk,
	SC::Slash,
	SC::Plus,
	SC::Minus,
};

Token_t::Token_t(Type t, const string& s, int prec, bool ra) noexcept
	: type{ t }, str{ s }, precedence{ prec }, rightAssociative{ ra }
{
	if (type == Type::Function) {
		if (str == "sqrt") Do = &sqrt;
		else if (str == "exp") Do = &exp;
		else if (str == "sin") Do = &sin;
		else if (str == "cos") Do = &cos;
		else
			Except::Abort(Except::Code::INVALID_FLOATING_POINT);
	}
	else if (type == Type::Number) {
		v = Util_t::Float(str);
		coefficient = 1.0; exponent = 1;
	}
	else if (type == Type::Variable) { 
		if (s != "x" && s != "y" && s != "z")
			Except::Abort(Except::Code::INVALID_VARIABLE);
		coefficient = 1.0; exponent = 1; 
	}
}

bool Token_t::IsFunction(siterator s, siterator end) noexcept
{
	auto k = std::find_if_not(s, end, isalnum);

	return isalpha(*s) && *k == SC::LeftParen;
}

bool Token_t::IsVariable(siterator s, siterator end) noexcept
{
	if (!isalpha(*s)) return false;
	if (*s != 'x' && *s != 'y' && *s != 'z') return false;
	return true;
}

Token_t Token_t::GetNumeric(siterator& pos, siterator end)
{
	auto beg = pos;
	int dot = 0, exp = 0, sign = 0;
	
	for (; pos < end; ++pos) {
		if (isdigit(*pos)) continue;
		else if (*pos == SC::Dot) ++dot;
		else if (toupper(*pos) == 'E') ++exp;
		else if (*pos == SC::Plus || *pos == SC::Minus) ++sign;
		else break;

		if (dot > 1 || exp > 1 || sign > 1)
			Except::Abort(Except::Code::INVALID_FLOATING_POINT);
	}

	return Token_t(Type::Number, string(beg, pos));
}

Token_t Token_t::GetFunction(siterator& pos, siterator end)
{
	auto beg = pos;
	pos = std::find(pos, end, SC::LeftParen);

	return Token_t(Type::Function, string(beg, pos), 5);
}

Token_t Token_t::GetVariable(siterator& pos)
{
	auto beg = pos;

	if (*pos != 'x' && *pos != 'y' && *pos != 'z')
		Except::Abort(Except::Code::INVALID_VARIABLE);
	
	return Token_t(Type::Variable, string(beg, ++pos));
}

Token_t Token_t::GetParenthesis(siterator& pos)
{
	auto beg = pos;
	if (*pos == SC::LeftParen)
		return Token_t(Type::LeftParen, string(beg, ++pos));
	else if (*pos == SC::RightParen)
		return Token_t(Type::RightParen, string(beg, ++pos));
}

Token_t Token_t::GetOperator(siterator& pos)
{
	auto beg = pos;
	int pr = -1; bool ra = false;

	switch (*pos) {
	case SC::Caret :
		pr = 4; ra = true; break;
	case SC::Asterisk :
	case SC::Slash :
		pr = 3; break;
	case SC::Plus :
	case SC::Minus :
		pr = 2; break;
	case SC::LeftAngle :
	case SC::RightAngle :
		pr = 5; break;
	}

	return Token_t(Type::Operator, string(beg, ++pos), pr, ra);
}

Token_t Token_t::GetAngle(siterator& pos)
{
	auto beg = pos;
	return Token_t(Type::Angle, string(beg, ++pos));
}

deque<Token_t> Parser_t::Tokenize(const string& line) noexcept
{
	using Type = Token_t::Type;

	auto eq = line;
	eq = regex_replace(eq, regex(R"(y\*x)"), R"(x\*y)");
	eq = regex_replace(eq, regex(R"(z\*y)"), R"(y\*z)");
	eq = regex_replace(eq, regex(R"(x\*z)"), R"(z\*x)");

	deque<Token_t> tokens;

	for (auto i = eq.begin(); i < eq.end(); ) {
		if (Token_t::IsNumeric(i))
			tokens.push_back(Token_t::GetNumeric(i, eq.end()));
		else if (Token_t::IsFunction(i, eq.end())) 
			tokens.push_back(Token_t::GetFunction(i, eq.end()));
		else if (Token_t::IsVariable(i, eq.end()))
			tokens.push_back(Token_t::GetVariable(i));
		else if (Token_t::IsParen(i))
			tokens.push_back(Token_t::GetParenthesis(i));
		else if (Token_t::IsOperator(i))
			tokens.push_back(Token_t::GetOperator(i));
		else if (Token_t::IsAngle(i))
			tokens.push_back(Token_t::GetAngle(i));
		else 
			Except::Abort(Except::Code::INVALID_FLOATING_POINT);
	}

	return static_cast<deque<Token_t>&&>(tokens);
}

deque<Token_t> Parser_t::ShuntingYard(const deque<Token_t>& tokens) noexcept
{
	using Type = Token_t::Type;

	deque<Token_t> queue;
	vector<Token_t> stack;

	for (const auto& token : tokens) {

		switch (token.TokenType()) {

			case Type::Number :
			case Type::Variable :
			case Type::Angle :
				queue.push_back(token); break;

			case Type::Function :
			case Type::Operator: {
				const auto o1 = token;
				while (!stack.empty()) {
					const auto o2 = stack.back();
					if ((!o1.RightAssociative() && o1.Precedence() <= o2.Precedence()) ||
						(o1.RightAssociative() && o1.Precedence() < o2.Precedence())) {
						stack.pop_back();
						queue.push_back(o2);
						continue;
					}
					break;
				}
				stack.push_back(o1);
			}
			break;

			case Type::LeftParen :
				stack.push_back(token); break;
			case Type::RightParen: {
				bool match = false;
				while (!stack.empty() && stack.back().TokenType() != Type::LeftParen) {
					queue.push_back(stack.back());
					stack.pop_back();
					match = true;
				}
				stack.pop_back();
			}
			break;
		}

	}

	while (!stack.empty()) {
		queue.push_back(std::move(stack.back()));
		stack.pop_back();
	}

	return queue;
}

array<double, 10> Parser_t::ParseEquation(const string& line)
{
	auto tokens = Tokenize(line);
	decltype(tokens) lhs, rhs;

	auto queue = ShuntingYard(tokens);

	return array<double, 10>();
}

}