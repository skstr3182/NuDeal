#pragma once
#include "Defines.h"
#include "IODeclare.h"

namespace IO
{

class Token_t
{
public :

	enum class Type 
	{
		Number,
		Identifier,
		LeftParen,
		RightParen,
		LeftBracket,
		RightBracket,
		LeftBrace,
		RightBrace,
		LeftAngle,
		RightAngle,
		Equal,
		Plus,
		Minus,
		Asterisk,
		Slash,
		Caret,
		Hash,
		Dot,
		Comma,
		Colon,
		SemiColon,
		END,
		INVALID = -1
	};

	static const map<char, Type> special_table;

private :

	Type m_type{};
	string_view m_lexeme{};

public :

	using SC = SpecialCharacters;
	using iterator = string::const_iterator;

	Token_t() {}
	Token_t(Type kind) noexcept : m_type{kind} {}
	Token_t(Type kind, iterator beg, size_t len) noexcept
		: m_type{kind}, m_lexeme(&(*beg), len) {}
	Token_t(Type kind, iterator beg, iterator end) noexcept
		: m_type{kind}, m_lexeme(&(*beg), std::distance(beg, end)) {}
	Type Kind() const noexcept { return m_type; }
	bool Is(Type kind) const noexcept { return m_type == kind; }
	bool IsNot(Type kind) const noexcept { return m_type != kind; }
	bool IsOneOf(Type k1, Type k2) const noexcept { return Is(k1) || Is(k2); }
	template <typename... Ts>
	bool IsOneOf(Type k1, Type k2, Ts... ks) const noexcept
	{ return Is(k1) || IsOneOf(k2, ks...); }
	string_view Lexeme() const noexcept { return m_lexeme; }
	void Lexeme(const string_view& s) noexcept { m_lexeme = s; }
};

class Lexer_t
{

private :

	using SC = SpecialCharacters;
	string contents;
	string::const_iterator m_pos;

public :

	void Lex(const string& contents);

private :
	
	using TokenType = Token_t::Type;

	static const regex number, word, special;

	vector<Token_t> tokens;

	Token_t Next() noexcept;
	Token_t Identifier() noexcept;
	Token_t Number() noexcept;
	Token_t Atom(Token_t::Type kind) noexcept { return Token_t(kind, m_pos++, 1); };
	static Token_t::Type GetEscapeName(char c) noexcept;

	char Peek() const noexcept { return *m_pos; }
	char Get() noexcept { return *m_pos++; }

public :

	const auto& Tokens() { return tokens; }

	static bool IsDigit(char c) noexcept;
	static bool IsIdentifierChar(char c) noexcept;

};

}