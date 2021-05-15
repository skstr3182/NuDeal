#include "Parser.h"
#include "Exception.h"
#include "Lexer.h"

namespace IO
{

bool Parser_t::IsDigit(char c) noexcept
{
	static const auto numeric = regex(R"([0-9])");
	return regex_match(&c, &c + 1, numeric);
}

bool Parser_t::IsVariable(char c) noexcept
{
	static const auto variable = regex(R"([xyz])");
	return regex_match(&c, &c + 1, variable);
}

bool Parser_t::IsOperator(char c) noexcept
{
	static const auto ops = regex(Parser_t::operators);
	return regex_match(&c, &c + 1, ops);
}

void Parser_t::TreatNumeric(string::const_iterator& pos)
{
	int dot = 0, exponent = 0;
	static const regex re(R"((\d)|(\.)|[Ee]|[+-])");

	while (regex_match(pos, pos + 1, re)) {
		if (*pos == SC::Dot) ++dot;
		if (toupper(*pos) == 'E') ++exponent;

		if (dot > 1 || exponent > 1) 
			Except::Abort(Except::Code::INVALID_FLOATING_POINT);
		++pos;
	}
}

void Parser_t::TreatVariable(string::const_iterator& pos)
{
	int asterisk = 0, caret = 0;
	static const regex re(R"((\^)|(\d)|[xyz])");

	while (regex_match(pos, pos + 1, re)) {
		if (*pos == SC::Asterisk) ++asterisk;
		if (*pos == SC::Caret) ++ caret;

		if (caret > 1 || asterisk > 1)
			Except::Abort(Except::Code::INVALID_FLOATING_POINT);
		++pos;
	}

}

vector<string> Parser_t::Tokenize(const string& line) noexcept
{
	static const auto ops = regex(R"([)" + string(Parser_t::operators) + R"(])");

	auto eq = line;
	eq = regex_replace(eq, regex(R"(y\*x)"), R"(x\*y)");
	eq = regex_replace(eq, regex(R"(z\*y)"), R"(y\*z)");
	eq = regex_replace(eq, regex(R"(x\*z)"), R"(z\*x)");

	auto pos = eq.begin();
	vector<string> tokens;

	while (pos != eq.end()) {
		auto beg = pos++;
		while (!regex_match(pos, pos + 1, ops)) ++pos;
		tokens.emplace_back(beg, pos);
	}

	return static_cast<vector<string>&&>(tokens);
}

array<double, 10> Parser_t::ParseEquation(const string& line)
{
	
	
	auto tokens = Tokenize(line);

	return array<double, 10>();
}

}