#include "Input.h"
#include "IOUtil.h"
#include "Exception.h"
#include "IOUtil.h"
#include "Lexer.h"

namespace IO
{

using Util = Util_t;
using SC = Util::SC;

void InputManager_t::HashTree_t::Make(const string& file, 
	string::size_type Beg, string::size_type End)
{
	string::size_type beg = Beg, end;

	while ((end = Util::FindEndPoint(file, beg)) < End) {
		auto name = file.substr(Beg, beg - Beg);

		beg = file.find_first_not_of(SC::LeftBrace, beg);
		auto contents = file.substr(beg, end - beg);

		HashTree_t Tnew, &T = this->children[Util::Trim(name)];
		Tnew.parent = this;
		Tnew.CountLine(name, contents);
		T = std::move(Tnew);

		if (Util::IsClosed(contents)) 
			T.Make(file, beg, end);
		else 
			T.contents = contents;

		beg = end + 1; Beg = beg;
	}
}

void InputManager_t::HashTree_t::CountLine(const string& name, const string& contents)
{
	num_lines = Util::LineCount(name) + Util::LineCount(contents);
	line_info = Util::LineCount(name, name.find_first_not_of(SC::LF)) + 1;
	line_info += parent->line_info;
	for (const auto& t : parent->children) line_info += t.second.num_lines;
}

void InputManager_t::Preprocessor_t::DeleteComment(string& contents)
{
	using Except = Exception_t;
	using Code = Except::Code;

	static constexpr char open[] = "/*", close[] = "*/";
	static constexpr char comment[] = "//";

	try {
		AreParenthesesMatched(contents, { string(open) }, { string(close) });
	}
	catch (...) {
		throw runtime_error("Mismatched comments");
	}

	size_type pos = 0;

	while ((pos = contents.find(comment, pos)) != string::npos) {
		auto LF = contents.find(SC::LF, pos);
		contents.replace(contents.begin() + pos, contents.begin() + LF, LF - pos, SC::Blank);
		LF + 1;
	}
	
	pos = 0;

	while ((pos = contents.find(open, pos)) != string::npos) {
		auto end = contents.find(close, pos);
		std::replace_if(contents.begin() + pos, contents.begin() + end + strlen(close),
			[] (char c) { return c != SC::LF; }, SC::Blank);
		pos = end + strlen(close) + 1;
	}
}

void InputManager_t::Preprocessor_t::ApplyMacro(string& contents)
{
	using Except = Exception_t;
	using Code = Except::Code;

	static const string macro_str = R"~([\b]*(#[A-z]{2,})\s+(\w+)(\(.*\))?\s+((?:.*\\\r?\n)*.*))~";
	static const regex macro_re(macro_str);

	/*
	[\b]*(#[A-z]{2,}) : #define
	\s+ : Whitespace
	(\w+) : Name
	(\(.*\))? : Potential Argument
	\s+ : Whitespace
	(?:.*\\\r?\n)*.* : Contents;
	*/

	auto beg = sregex_token_iterator(contents.begin(), contents.end(), macro_re);
	sregex_token_iterator end;
	vector<string> macro;

	// Pull Out

	for (auto iter = beg; iter != end; ++iter) {
		auto& s = iter->str();
		auto pos = contents.find(s);
		std::replace_if(s.begin(), s.end(), 
			[] (char c) { return c == SC::BackSlash || c == SC::LF; }, SC::Blank);
		macro.push_back(s);
		std::replace_if(contents.begin() + pos, contents.begin() + pos + s.size(),
			[] (char c) { return c != SC::LF; }, SC::Blank);
		
	}

	// Directive

	for (const auto& i : macro) {
		auto beg = sregex_token_iterator(i.begin(), i.end(), macro_re, 1);
		if (beg->str() != "#define") throw runtime_error("Invalid directive");
	}

	// Check Redefinition

	set<string> check_redef;

	for (const auto& i : macro) {
		auto beg = sregex_token_iterator(i.begin(), i.end(), macro_re, 2);
		if (check_redef.find(beg->str()) != check_redef.end()) throw runtime_error("Redfeind macro");
		check_redef.insert(beg->str());
	}


	// Inlining

	for (const auto& i : macro) {
		auto name = sregex_token_iterator(i.begin(), i.end(), macro_re, 2)->str();
		auto arguments = sregex_token_iterator(i.begin(), i.end(), macro_re, 3)->str();
		auto func = sregex_token_iterator(i.begin(), i.end(), macro_re, 4)->str();

		arguments = Util::EraseSpace(arguments, " \n()");
		auto v_args = Util::SplitFields(arguments, string(1, SC::Comma));
		int num_args = arguments.empty() ? 0 : std::count(arguments.begin(), arguments.end(), SC::Comma) + 1;

		string reg_str = name;
		if (num_args) {
			reg_str += R"~(\()~";
			for (int i = 0; i < num_args; ++i) {
				reg_str += R"(\s*(\w+)\s*)";
				if (i < num_args - 1) reg_str += R"(,)";
			}
			reg_str += R"~(\))~";
		}
		
		regex re(reg_str);
		sregex_token_iterator beg, end;

		while ((beg = sregex_token_iterator(contents.begin(), contents.end(), re)) != end) {
			vector<sregex_token_iterator> begs(num_args);
			for (int i = 0; i < num_args; ++i) 
				begs[i] = sregex_token_iterator(contents.begin(), contents.end(), re, i + 1);
			auto replace = func;
			for (int i = 0; i < num_args; ++i) {
				auto val = begs[i]->str();
				auto arg = v_args[i];
				regex re(R"(\s*)" + arg + R"(\s*)");
				replace = regex_replace(replace, re, val);
			}
			auto& str = beg->str();
			auto pos = contents.find(str);
			contents.replace(contents.begin() + pos, contents.begin() + pos + str.size(), replace);
		}
	}
}

bool InputManager_t::Preprocessor_t::AreParenthesesMatched(const string& contents, 
	const vector<string>& open, const vector<string>& close)
{
	vector<string> S;

	auto iter = contents.begin();

	while (iter < contents.end()) {
		size_type advance = 1;
		for (int s = 0; s < open.size(); ++s) {
			auto l = open[s].size();
			auto c = contents.substr(iter - contents.begin(), l);
			if (c == open[s]) { S.push_back(open[s]); advance = l; break; }
		}
		for (int s = 0; s < close.size(); ++s) {
			auto l = close[s].size();
			auto c = contents.substr(iter - contents.begin(), l);
			if (c == close[s]) {
				if (S.empty() || S.back() != open[s]) 
					throw runtime_error("Mismatched" + open[s] + close[s]);
				else
					S.pop_back();
				advance = l;
			}
		}
		iter += advance;
	}

	if (!S.empty()) throw runtime_error("Mismatched");

	return true;
}

InputManager_t::Blocks InputManager_t::GetBlockID(string line) const
{
	int pos_end = line.find(SC::Blank, 0);
	string block = line.substr(0, pos_end);

	block = Util::Uppercase(block);
	for (int i = 0; i < BlockNames.size(); ++i)
		if (!block.compare(BlockNames[i]))
			return static_cast<Blocks>(i);
	return Blocks::INVALID;
}

template <typename T>
T InputManager_t::GetCardID(Blocks block, string line) const
{
	static constexpr T INVALID = static_cast<T>(-1);

	int pos_beg = line.find_first_not_of(SC::Blank);
	if (pos_beg == string::npos) return INVALID;

	int pos_end = line.find(SC::Blank, pos_beg);
	string card = line.substr(pos_beg, pos_end - pos_beg);

	card = Util::Uppercase(card);
	int b = static_cast<int>(block);
	for (int i = 0; i < CardNames[b].size(); ++i) 
		if (!card.compare(CardNames[b][i])) 
			return static_cast<T>(i);
	return INVALID;
}

void InputManager_t::ExtractInput(istream& fin)
{
	stringstream strstream;

	strstream << fin.rdbuf();
	
	contents = strstream.str();

	std::replace(contents.begin(), contents.end(), SC::Tab, SC::Blank);
	std::replace(contents.begin(), contents.end(), SC::CR, SC::Blank);
}

// BLOCK

void InputManager_t::ParseGeometryBlock(HashTree_t& Tree)
{
	using Except = Exception_t;
	using Code = Except::Code;
	using Cards = GeometryCards;

	for (auto& T : Tree.children) {
		
		auto& object = T.second;
		string card = Util::Trim(T.first);
		Cards ID = GetCardID<Cards>(Blocks::GEOMETRY, card);

		switch (ID)
		{
		case Cards::UNITVOLUME :
			ParseUnitVolumeCard(object); break;
		case Cards::UNITCOMP :
			ParseUnitCompCard(object); break;
		case Cards::INVALID :
			Except::Abort(Code::INVALID_INPUT_CARD, card, object.GetLineInfo());
		}

	}

}

void InputManager_t::ParseMaterialBlock(HashTree_t& Tree)
{

}

void InputManager_t::ParseOptionBlock(HashTree_t& Tree)
{

}

/// GEOMETRY CARDS

void InputManager_t::ParseUnitVolumeCard(HashTree_t& Tree)
{
	using Except = Exception_t;
	using Code = Except::Code;

	string ORIGIN = "ORIGIN";

	for (auto& T : Tree.children) {
		auto name = Util::Trim(T.first);
		auto& object = T.second;
		auto contents = Util::EraseSpace(object.contents);
		auto v = Util::SplitFields(contents, string(1, SC::SemiColon));
		
		UnitVolume_t U;

		for (auto i = v.begin(); i != v.end(); ) {
			auto& s = *i;
			if (s.find(ORIGIN) != string::npos) {
				auto delimiter = string(1, s[ORIGIN.size()]);
				auto u = Util::SplitFields(s, delimiter);

				if (u.size() != 2) 
					Except::Abort(Code::INVALID_ORIGIN_DATA, object.contents, object.GetLineInfo());
				U.origin = u.back();

				i = v.erase(i);
			}
			else ++i;
		}

		v = Util::SplitFields(v.front(), string(1, SC::SemiColon));

		U.equations = v;

		unitVolumes[name] = U;
	}

}

void InputManager_t::ParseUnitCompCard(HashTree_t& Tree)
{
	using Except = Exception_t;
	using Code = Except::Code;

	for (auto& T : Tree.children) {
		auto& object = T.second;
		auto prefix = Util::EraseSpace(T.first);
		auto v = Util::SplitFields(prefix, string(1, SC::Colon));
		if (v.size() != 2) 
			Except::Abort(Code::BACKGROUND_MISSED, object.contents, object.GetLineInfo());
		auto name = v.front();
		auto background = v.back();
		auto contents = Util::EraseSpace(object.contents);
		v = Util::SplitFields(contents, string(1, SC::SemiColon));

		UnitComp_t U;

		U.background = background;

		for (const auto& s : v) {
			auto u = Util::SplitFields(s, string(2, SC::RightAngle));
			U.unitvols.push_back(u.front());
			U.displace.push_back(vector<string>());
			u.erase(u.begin());
			for (const auto& k : u) {
				auto lpos = k.find(SC::LeftParen) + 1;
				auto rpos = k.find(SC::RightParen);
				if (lpos == string::npos && rpos == string::npos)
					U.displace.back().push_back(k);
				else if (lpos != string::npos && rpos != string::npos)
					U.displace.back().push_back(k.substr(lpos, rpos - lpos));
				else
					Except::Abort(Code::MISMATCHED_BRAKETS, object.contents, object.GetLineInfo());
			}
		}

		unitComps[name] = U;
	}
}

void InputManager_t::Preprocess()
{
	using Preprocessor = Preprocessor_t;
	using Except = Exception_t;
	using Code = Except::Code;

	try {
		Preprocessor::DeleteComment(contents);
	}
	catch (exception& e) {
		Except::Abort(Code::MISMATCHED_BRAKETS, e.what());
	}

	try {
		Preprocessor::ApplyMacro(contents);
	}
	catch (exception& e) {
		Except::Abort(Code::INVALID_MACRO, e.what());
	}

	try {
		Preprocessor::AreParenthesesMatched(contents, {"{", "[", "("}, {"}", "]", ")"} );
	}
	catch (exception& e) {
		Except::Abort(Code::MISMATCHED_BRAKETS, e.what());
	}

	Lexer = new Lexer_t;

	Lexer->Lex(contents);

	HashTree.Make(contents);
}

void InputManager_t::ReadInput(string file)
{
	using Except = Exception_t;
	using Code = Except::Code;

	ifstream fin(file);
	
	if (fin.fail()) Except::Abort(Code::FILE_NOT_FOUND, file);

	this->file = file;

	ExtractInput(fin);

	fin.close();

	Preprocess();
	
	for (auto& T : HashTree.children) {
		auto& contents = T.second;
		string block = Util::Trim(T.first);
		Blocks ID = GetBlockID(block);
		switch (ID)
		{
		case Blocks::GEOMETRY :
			ParseGeometryBlock(contents); break;
		case Blocks::MATERIAL :
			ParseMaterialBlock(contents); break;
		case Blocks::OPTION :
			ParseOptionBlock(contents); break;
		case Blocks::INVALID :
			Except::Abort(Code::INVALID_INPUT_BLOCK, block, contents.GetLineInfo());
		}
	}


}

}