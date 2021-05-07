#include "Lexer.h"
#include "IOUtil.h"
#include "Exception.h"

namespace IO
{

const map<char, Token_t::Type> Token_t::special_table =
{
	{SC::LeftParen,			Type::LeftParen		 },
	{SC::RightParen,		Type::RightParen	 },
	{SC::LeftBracket,		Type::LeftBracket	 },
	{SC::RightBracket,	Type::RightBracket },
	{SC::LeftBrace,			Type::LeftBrace		 },
	{SC::RightBrace,		Type::RightBrace	 },
	{SC::LeftAngle,			Type::LeftAngle		 },
	{SC::RightAngle,		Type::RightAngle	 },
	{SC::Equal,					Type::Equal				 },
	{SC::Plus,					Type::Plus				 },
	{SC::Minus,					Type::Minus				 },
	{SC::Asterisk,			Type::Asterisk		 },
	{SC::Slash,					Type::Slash				 },
	{SC::Caret,					Type::Caret				 },
	{SC::Dot,						Type::Dot					 },
	{SC::Comma,					Type::Comma				 },
	{SC::Colon,					Type::Colon				 },
	{SC::SemiColon,			Type::SemiColon		 }
};

regex const Lexer_t::number = regex(R"([0-9])");
regex const Lexer_t::word = regex(R"([a-zA-Z])");
regex const Lexer_t::special = regex(R"([\(\)\{\}\[\]\<\>\=\+\-\*\^\/\,\:\;])");

void Lexer_t::Lex(const string& input)
{
	using Except = Exception_t;
	using Code = Except::Code;
	contents = input;
	m_pos = this->contents.begin();

	for (auto next = Next(); m_pos < contents.end();  next = Next()) {
		if (next.Is(TokenType::INVALID)) Except::Abort(Code::INVALID_VARIABLE, string(1, *m_pos));
		tokens.push_back(std::move(next));
	}

	regex re(R"(.*;)");
	auto t = sregex_token_iterator(contents.begin(), contents.end(), re);
	sregex_token_iterator end;

	for (; t != end; ++t) {
		cout << string(t->first, t->second) << endl;
	}
}

Token_t Lexer_t::Next() noexcept
{
	while (isspace(Peek())) Get();

	if (m_pos == contents.end())
		return Token_t(TokenType::END);
	else if (regex_match(m_pos, m_pos + 1, word)) 
		return Identifier();
	else if (regex_match(m_pos, m_pos + 1, number)) 
		return Number();
	else if (regex_match(m_pos, m_pos + 1, special))
		return Atom(GetEscapeName(*m_pos));
	else
		return Token_t(TokenType::INVALID, m_pos, 1);
}

void Lexer_t::March() noexcept
{
	static const string stop_signal = "{};";
	while (stop_signal.find_first_of(Peek()) != string::npos) Get();
}

bool Lexer_t::IsDigit(char c) noexcept
{
	static const regex re(R"([0-9])");
	return regex_match(&c, &c + 1, re);
}

bool Lexer_t::IsIdentifierChar(char c) noexcept
{
	static const regex re(R"([a-zA-Z0-9_])");
	return regex_match(&c, &c + 1, re);
}

Token_t Lexer_t::Identifier() noexcept
{
	const auto beg = m_pos;

	Get();
	while (IsIdentifierChar(Peek())) Get();
	auto l = string(beg, m_pos);
	
	return Token_t(TokenType::Identifier, beg, m_pos);
}

Token_t Lexer_t::Number() noexcept
{
	static const regex re(R"((\d)|(\.)|[Ee]|[+-])");
	const auto beg = m_pos;

	int dot = 0, exponent = 0;;

	Get(); 
	while (regex_match(m_pos, m_pos + 1, re)) {
		if (Peek() == SC::Dot) ++dot;
		if (toupper(Peek()) == 'E') ++exponent;

		if (dot > 1) break;
		if (exponent > 1) break;
		Get();
	}
	return Token_t(TokenType::Number, beg, m_pos);
}

Token_t::Type Lexer_t::GetEscapeName(char c) noexcept
{
	auto iter = Token_t::special_table.find(c);
	return iter == Token_t::special_table.end() ? TokenType::INVALID : iter->second;
}

}