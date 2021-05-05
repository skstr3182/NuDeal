#include "Lexer.h"

namespace IO
{

bool Lexer_t::IsSpace(char c) noexcept
{
	static const regex re(R"([\s\t\r\n])");
	return regex_match(&c, &c + 1, re);
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

Token_t Lexer_t::Next() noexcept
{
	while (IsSpace(Peek())) Get();

	static const regex word(R"([a-zA-Z])");
	static const regex number(R"([0-9])");
	static const regex special(R"~([\(\)\[\]\{\}\<\>\=\+\-\*\^\/\#\.\,\:\;\|])~");
	static const regex end(R"(\0)");

	using Type = Token_t::Type;
	

	cmatch m;

	if (regex_match(m_beg, m_beg + 1, end))
		return Token_t(Token_t::Type::END, m_beg, 1);
	else if (Peek() == SC::Hash) 
		return Macro();
	else if (Peek() == SC::Slash)
		return SlashOrComment();
	else if (regex_match(m_beg, m_beg + 1, word)) 
		return Identifier();
	else if (regex_match(m_beg, m_beg + 1, number)) 
		return Number();
	else if (regex_match(m_beg, m_beg + 1, m, special))
		return Atom(Token_t::Type::LEFTPAREN);
	else
		return Token_t(Token_t::Type::INVALID, m_beg, 1);
}

Token_t Lexer_t::Identifier() noexcept
{
	const auto *beg = m_beg;
	Get();
	while (IsIdentifierChar(Peek())) Get();
	return Token_t(Token_t::Type::IDENTIFIER, beg, m_beg);
}

Token_t Lexer_t::Number() noexcept
{
	const auto *beg = m_beg;
	Get();
	while (IsDigit(Peek())) Get();
	return Token_t(Token_t::Type::NUMBER, beg, m_beg);
}

Token_t Lexer_t::SlashOrComment() noexcept
{
	const auto *beg = m_beg;
	Get();
	if (Peek() == SC::Slash) {
		Get();
		beg = m_beg;
		while (Peek() != NULL) {
			if (Get() == SC::LF) 
				return Token_t(Token_t::Type::COMMENT, beg, std::distance(beg, m_beg) - 1);
		}
		return Token_t(Token_t::Type::INVALID, m_beg, 1);
	}
	else {
		return Token_t(Token_t::Type::SLASH, beg, 1);
	}
}

Token_t Lexer_t::Macro() noexcept
{
	const auto *beg = m_beg;
	auto linecount = 1;
	while (linecount) {
		Get();
		if (Peek() == SC::LF) --linecount;
		else if (Peek() == SC::BackSlash) ++linecount;
	}
	return Token_t(Token_t::Type::HASH, beg, std::distance(beg, m_beg));
}

Token_t::Type Lexer_t::GetEscapeName(char c) noexcept
{
	using T = Token_t::Type;
	static const map<char, T> table = 
	{
		{SC::LeftParen, T::LEFTPAREN},
		{SC::RightParen, T::RIGHTPAREN},
		
	};

	auto iter = table.find(c);
	return iter == table.end() ? T::INVALID : iter->second;
}

}