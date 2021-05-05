#pragma once
#include "Defines.h"
#include "IODeclare.h"
#include <regex>

namespace IO
{

class Token_t
{
public :
	enum class Type 
	{
		NUMBER,
		IDENTIFIER,
		LEFTPAREN,
		RIGHTPAREN,
		LEFTSQUARE,
		RIGHTSQUARE,
		LEFTCURLY,
		RIGHTCURLY,
		LESSTHAN,
		GREATERTHAN,
		EQUAL,
		PLUS,
		MINUS,
		ASTERISK,
		SLASH,
		CARET,
		HASH,
		DOT,
		COMMA,
		COLON,
		SEMICOLON,
		SINGLEQUOTE,
		DOUBLEQUOTE,
		COMMENT,
		PIPE,
		END,
		INVALID = -1
	};


private :

	Type m_type{};
	string_view m_lexeme{};

public :

	Token_t(Type kind) noexcept : m_type{kind} {};
	Token_t(Type kind, const char *beg, size_t len) noexcept
		: m_type{kind}, m_lexeme(beg, len) {}
	Token_t(Type kind, const char *beg, const char *end) noexcept
		: m_type{kind}, m_lexeme(beg, std::distance(beg, end)) {}
	Type GetType() const noexcept { return m_type; }
	void SetType(Type kind) noexcept { m_type = kind; }
	bool Is(Type kind) const noexcept { return m_type == kind; }
	bool IsNot(Type kind) const noexcept { return m_type != kind; }
	bool IsOneOf(Type k1, Type k2) const noexcept { return Is(k1) || Is(k2); }
	template <typename... Ts>
	bool IsOneOf(Type k1, Type k2, Ts... ks) const noexcept
	{
		return Is(k1) || IsOneOf(k2, ks...);
	}
	string_view GetLexeme() const noexcept { return m_lexeme; }
	void SetLexeme(string_view lexeme) noexcept 
	{
		m_lexeme = lexeme;
	}

};

class Lexer_t
{

private :

	using SC = SpecialCharacters;
	const char *m_beg = NULL;

public :

	Lexer_t(const char *beg) noexcept : m_beg{beg} {}
	
	Token_t Next() noexcept;

private :

	Token_t Identifier() noexcept;
	Token_t Number() noexcept;
	Token_t SlashOrComment() noexcept;
	Token_t Macro() noexcept;
	Token_t Atom(Token_t::Type kind) noexcept { return Token_t(kind, m_beg++, 1); };
	static Token_t::Type GetEscapeName(char c) noexcept;

	char Peek() const noexcept { return *m_beg; }
	char Get() noexcept { return *m_beg++; }

public :

	static bool IsSpace(char c) noexcept;
	static bool IsDigit(char c) noexcept;
	static bool IsIdentifierChar(char c) noexcept;

};

}