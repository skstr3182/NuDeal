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

void Lexer_t::Lex(const string& contents)
{
	using Except = Exception_t;
	using Code = Except::Code;

	this->contents = contents;
	m_pos = this->contents.begin();

	for (auto next = Next(); m_pos < this->contents.end();  next = Next()) {
		if (next.Is(Token_t::Type::INVALID)) Except::Abort(Code::INVALID_VARIABLE, string(1, *m_pos));
		tokens.push_back(std::move(next));
	}
}

Token_t Lexer_t::Next() noexcept
{
	while (isspace(Peek())) Get();

	using Type = Token_t::Type;

	smatch m;

	if (m_pos == contents.end())
		return Token_t(Token_t::Type::END, m_pos, 1);
	else if (regex_match(m_pos, m_pos + 1, word)) 
		return Identifier();
	else if (regex_match(m_pos, m_pos + 1, number)) 
		return Number();
	else if (regex_match(m_pos, m_pos + 1, m, special))
		return Atom(GetEscapeName(*m_pos));
	else
		return Token_t(Token_t::Type::INVALID, m_pos, 1);
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
	return Token_t(Token_t::Type::Identifier, beg, m_pos);
}

Token_t Lexer_t::Number() noexcept
{
	const auto beg = m_pos;
	Get();
	while (IsDigit(Peek())) Get();
	return Token_t(Token_t::Type::Number, beg, m_pos);
}

Token_t::Type Lexer_t::GetEscapeName(char c) noexcept
{
	using T = Token_t::Type;
	auto iter = Token_t::special_table.find(c);
	return iter == Token_t::special_table.end() ? T::INVALID : iter->second;
}

}