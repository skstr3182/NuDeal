#include "Parser.h"
#include "IOExcept.h"
// https://gist.github.com/t-mat/b9f681b7591cdae712f6

namespace IO
{

const initializer_list<char> Token_t::operators = {
	SC::Caret,
	SC::Asterisk,
	SC::Slash,
	SC::Plus,
	SC::Minus,
	SC::LeftAngle,
	SC::RightAngle
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
			Except::Abort(Except::Code::INVALID_EQUATION);
	}
	else if (type == Type::Variable) { 
		if (s != "x" && s != "y" && s != "z")
			Except::Abort(Except::Code::INVALID_EQUATION);
	}
}

bool Token_t::IsFunction(siterator s, siterator end) noexcept
{
	if (!isalpha(*s)) return false;
	auto k = std::find_if_not(s, end, isalnum);
	if (k == end) return false;
	return *k == SC::LeftParen;
}

bool Token_t::IsVariable(siterator s, siterator end) noexcept
{
	if (!isalpha(*s)) return false;
	auto k = std::find_if_not(s, end, isalnum);
	if (k == end) 
		return true;
	else 
		return std::find(operators.begin(), operators.end(), *k) != operators.end();
}

Token_t Token_t::GetNumeric(siterator& pos, const string& container)
{
	auto beg = pos;
	size_t sz;
	std::stod(string(pos, container.end()), &sz);
	pos += sz;

	return Token_t(Type::Number, string(beg, pos));
}

Token_t Token_t::GetFunction(siterator& pos, const string& container)
{
	auto beg = pos;
	pos = std::find(pos, container.end(), SC::LeftParen);

	return Token_t(Type::Function, string(beg, pos), 15);
}

Token_t Token_t::GetVariable(siterator& pos)
{
	auto beg = pos;
	return Token_t(Type::Variable, string(beg, ++pos));
}

Token_t Token_t::GetParenthesis(siterator& pos)
{
	auto beg = pos;
	if (*pos == SC::LeftParen)
		return Token_t(Type::LeftParen, string(beg, ++pos));
	else 
		return Token_t(Type::RightParen, string(beg, ++pos));
}

Token_t Token_t::GetOperator(siterator& pos, const string& container)
{
	auto beg = pos;
	int pr = -1; bool ra = false;
	Type t = Type::BinaryOp;

	switch (*pos) {
	case SC::Caret :
		pr = 13; ra = true; break;
	case SC::Asterisk : case SC::Slash :
		pr = 12; break;
	case SC::Plus : case SC::Minus : {
		if (pos == container.begin()) {
			pr = 14; t = Type::UnaryOp; ra = true;
		}
		else {
			auto prev = *(pos - 1);
			if (!isalnum(prev)) {
				pr = 14; t = Type::UnaryOp; ra = true;
			}
			else pr = 11;
		}
		break;
	}
	case SC::LeftAngle : case SC::RightAngle :
		pr = 10; break;
	}

	return Token_t(t, string(beg, ++pos), pr, ra);
}

deque<Token_t> Parser_t::Tokenize(const string& line) noexcept
{
	using Type = Token_t::Type;

	deque<Token_t> tokens;

	for (auto i = line.begin(); i < line.end(); ) {
		if (Token_t::IsNumeric(i))
			tokens.emplace_back(Token_t::GetNumeric(i, line));
		else if (Token_t::IsFunction(i, line.end())) 
			tokens.emplace_back(Token_t::GetFunction(i, line));
		else if (Token_t::IsVariable(i, line.end()))
			tokens.emplace_back(Token_t::GetVariable(i));
		else if (Token_t::IsParen(i))
			tokens.emplace_back(Token_t::GetParenthesis(i));
		else if (Token_t::IsOperator(i))
			tokens.emplace_back(Token_t::GetOperator(i, line));
		else 
			Except::Abort(Except::Code::INVALID_EQUATION);
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
				queue.push_back(token); break;

			case Type::Function:
			case Type::UnaryOp:
			case Type::BinaryOp: {
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
				while (!stack.empty() && stack.back().TokenType() != Type::LeftParen) {
					queue.push_back(stack.back());
					stack.pop_back();
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

	return static_cast<deque<Token_t>&&>(queue);
}

Parser_t::Variable_t Parser_t::Aggregate(const deque<Token_t>& queue) noexcept
{
	using Type = Token_t::Type;

	auto Tokens = deque<Token_t>(queue.begin(), queue.end() - 1);
	vector<Variable_t> stack;

	while (!Tokens.empty()) {
		string key; double value;
		const auto token = std::move(Tokens.front());
		Tokens.pop_front();
		switch (token.TokenType()) {
			case Type::Number: {
				auto& V = stack.emplace_back();
				key = "";
				V[key] += Util::Float(token.GetString());
				break;
			}
			case Type::Variable : {
				auto& V = stack.emplace_back();
				key = token.GetString();
				V[key] += 1.0;
				break;
			}
			case Type::Function: {
				// Pull back
				auto operand = std::move(stack.back()); stack.pop_back();

				// If operand is not a monomial
				if (operand.size() != 1)
					Except::Abort(Except::Code::INVALID_EQUATION);

				key = operand.begin()->first;
				value = operand.begin()->second;

				// If operand is a variable
				if (!key.empty())
					Except::Abort(Except::Code::INVALID_EQUATION);

				// Push result
				auto& result = stack.emplace_back();
				result[key] = token.Do(value);
				break;
			}
			case Type::UnaryOp: {
				auto op = token.GetString().front();
				auto& T = stack.back();
				switch (op) {
					case SC::Plus: break;
					case SC::Minus: 
						for (auto& t : T) t.second = -t.second;
						break;
				}
				break;
			}
			case Type::BinaryOp: {
				auto rhs = std::move(stack.back()); stack.pop_back();
				auto lhs = std::move(stack.back()); stack.pop_back();
				const auto op = token.GetString().front();
				auto& T = stack.emplace_back();
				switch (op) {
					case SC::Caret: T = TreatPow(lhs, rhs); break;
					case SC::Asterisk: T = TreatMult(lhs, rhs); break;
					case SC::Slash: T = TreatDiv(lhs, rhs); break;
					case SC::Plus: T = TreatAdd(lhs, rhs); break;
					case SC::Minus: T = TreatSub(lhs, rhs); break;
				}
				break;
			}
		}
	}

	auto rhs = std::move(stack.back()); stack.pop_back();
	auto lhs = std::move(stack.back()); stack.pop_back();
	Variable_t T;

	auto sign = queue.back().GetString().front();
	if (sign == SC::LeftAngle)
		T = TreatSub(lhs, rhs);
	else if (sign == SC::RightAngle)
		T = TreatSub(rhs, lhs);
	else
		Except::Abort(Except::Code::INVALID_EQUATION);

	for (auto iter = T.begin(); iter != T.end(); ++iter) {
		if (iter->first.empty()) continue;
		auto key = iter->first;
		if (key.size() > 2) Except::Abort(Except::Code::INVALID_EQUATION);
	}

	return static_cast<Variable_t&&>(T);
}

Parser_t::Variable_t Parser_t::TreatAdd(const Variable_t& lhs, const Variable_t& rhs)
{
	auto result = std::move(lhs);
	for (const auto& r : rhs) result[r.first] += r.second;
	return static_cast<Variable_t&&>(result);
}

Parser_t::Variable_t Parser_t::TreatSub(const Variable_t& lhs, const Variable_t& rhs)
{
	auto result = std::move(lhs);
	for (const auto& r : rhs) result[r.first] -= r.second;
	return static_cast<Variable_t&&>(result);
}

Parser_t::Variable_t Parser_t::TreatMult(const Variable_t& lhs, const Variable_t& rhs)
{
	Variable_t result;
	for (const auto& l : lhs) {
		for (const auto& r : rhs) {
			auto key = l.first + r.first;
			std::sort(key.begin(), key.end());
			result[key] += l.second * r.second;
		}
	}
	return static_cast<Variable_t&&>(result);
}

Parser_t::Variable_t Parser_t::TreatDiv(const Variable_t& lhs, const Variable_t& rhs)
{
	Variable_t result;
	if (rhs.size() != 1)
		Except::Abort(Except::Code::INVALID_EQUATION);
	auto r = rhs.begin();
	if (!r->first.empty()) {
		for (auto& l : lhs) {
			auto p = l.first.find(r->first);
			if (p == string::npos)
				Except::Abort(Except::Code::INVALID_EQUATION);
			auto key = l.first;
			key.erase(p, 1);
			result[key] += l.second / r->second;
		}
	}
	else {
		for (auto& l : lhs) {
			auto key = l.first;
			result[key] += l.second / r->second;
		}
	}
	return static_cast<Variable_t&&>(result);
}

Parser_t::Variable_t Parser_t::TreatPow(const Variable_t& lhs, const Variable_t& rhs)
{
	Variable_t result;
	
	if (rhs.size() != 1 || !rhs.begin()->first.empty()) 
		Except::Abort(Except::Code::INVALID_EQUATION);

	if (lhs.size() == 1) {
		if (lhs.begin()->first.empty()) {
			decltype(result)::key_type key = "";
			result[key] = pow(lhs.begin()->second, rhs.begin()->second);
		}
		else {
			auto p = rhs.begin()->second;
			if (trunc(p) != p)
				Except::Abort(Except::Code::INVALID_EQUATION);
			decltype(result)::key_type key = "";
			if (!trunc(p))
				result[key] = 1.0;
			else if (trunc(p) < 0)
				Except::Abort(Except::Code::INVALID_EQUATION);
			else {
				for (int i = 0; i < trunc(p); ++i) key += lhs.begin()->first;
				result[key] = pow(lhs.begin()->second, p);
			}
		}
	}
	else {
		auto p = rhs.begin()->second;
		if (trunc(p) != p)
			Except::Abort(Except::Code::INVALID_EQUATION);
		if (!trunc(p))
			result[""] = 1.0;
		else if (trunc(p) < 0)
			Except::Abort(Except::Code::INVALID_EQUATION);
		else {
			result = lhs;
			for (int i = 0; i < trunc(p); ++i) result = TreatMult(result, lhs);
		}
	}
	return static_cast<Variable_t&&>(result);
}

array<double, 10> Parser_t::ParseEquation(const string& line)
{
	static const map<string, int> order = {
		{"xx", 0},
		{"yy", 1},
		{"zz", 2},
		{"xy", 3},
		{"yz", 4},
		{"xz", 5},
		{"x",  6},
		{"y",  7},
		{"z",  8},
		{"",   9}
	};

	auto tokens = Tokenize(line);
	auto queue = ShuntingYard(tokens);
	auto eqn = Aggregate(queue);
	
	array<double, 10> result = {0.0, };

	for (auto iter = eqn.begin(); iter != eqn.end(); ++iter) {
		auto key = iter->first;
		auto l = order.find(key)->second;
		result[l] = iter->second;
	}

	return static_cast<array<double, 10>&&>(result);
}

}