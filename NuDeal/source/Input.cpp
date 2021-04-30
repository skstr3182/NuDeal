#include "Input.h"
#include "IOUtil.h"
#include "Exception.h"
#include "Parser.h"

namespace IO
{

using Parser = Parser_t;
using SC = Parser::SC;

InputManager_t::Blocks InputManager_t::GetBlockID(string line) const
{
	int pos_end = line.find(SC::BLANK, 0);
	string block = line.substr(0, pos_end);

	block = Parser::Uppercase(block);
	for (int i = 0; i < num_blocks; ++i)
		if (!block.compare(BlockNames[i]))
			return static_cast<Blocks>(i);
	return Blocks::INVALID;
}

template <typename T>
T InputManager_t::GetCardID(Blocks block, string line) const
{
	int pos_beg = line.find_first_not_of(SC::BLANK);
	if (pos_beg == string::npos) return static_cast<T>(INVALID);

	int pos_end = line.find(SC::BLANK, pos_beg);
	string card = line.substr(pos_beg, pos_end - pos_beg);

	card = Parser::Uppercase(card);
	int b = static_cast<int>(block);
	for (int i = 0; i < num_cards; ++i) 
		if (!card.compare(CardNames[b][i])) 
			return static_cast<T>(i);
	return static_cast<T>(INVALID);
}

stringstream InputManager_t::ExtractInput(istream& fin)
{
	stringstream fileStream;

	fileStream << fin.rdbuf();
	
	string contents = fileStream.str();

	std::replace(contents.begin(), contents.end(), SC::TAB, SC::BLANK);
	std::replace(contents.begin(), contents.end(), SC::CR, SC::BLANK);

	fileStream.str("");
	fileStream.str(contents);

	return static_cast<stringstream&&>(fileStream);
}

void InputManager_t::InspectSyntax(stringstream& file)
{
	using Except = Exception_t;
	using Code = Except::Code;

	if (!Parser::AreBracketsMatched(file.str()))
		Except::Abort(Code::MISMATCHED_BRAKETS);

	if (!Parser::IsMacroValid(file.str()))
		Except::Abort(Code::INVALID_MACRO);
}

void InputManager_t::Tree_t::ProcessMacro(stringstream& in)
{
	auto buffer = in.str();
	size_t define_pos, beg = 0;

	while ((define_pos = buffer.find("#define", beg)) != string::npos) {
		auto define_end = define_pos;
		string::size_type backslash, LF;
		do {
			LF = buffer.find(SC::LF, define_end);
			backslash = std::find(buffer.begin() + define_end, 
				buffer.begin() + LF, SC::BACKSLASH) - buffer.begin();
			define_end = LF + 1;
		} while (backslash < LF);
		
		auto macro = buffer.substr(define_pos, define_end - define_pos);
		auto v = Parser::ExtractMacro(macro);
		this->define[v[1]] = v[2];
		std::replace_if(macro.begin(), macro.end(), 
			[] (char c) { return c != SC::LF; }, SC::BLANK);
		buffer.replace(buffer.begin() + define_pos, buffer.begin() + define_end, 
			macro.begin(), macro.end());
		beg = LF;
	}

	in.clear(ios::goodbit);
	in.str("");
	in.str(buffer);
}

void InputManager_t::Tree_t::Make(stringstream& in, size_t offset)
{
	do {

		auto contents = Parser::GetBlock(in);
		if (Parser::Trim(contents).empty()) continue;
		auto pos = contents.find_first_of(SC::LBRACE);
		auto name = contents.substr(0, pos);

		auto& T = this->children[name];

		contents.replace(contents.find_first_of(SC::LBRACE), 1, string(1, SC::BLANK));
		contents.replace(contents.find_last_of(SC::RBRACE), 1, string(1, SC::BLANK));

		auto count = std::count(contents.begin(), contents.end(), SC::LBRACE);

		if (count == 0)
			T.contents = contents;
		else {
			stringstream next(contents);
			next.seekg(pos + 1);
			T.Make(next);
		}

	} while (!in.eof());
}

// BLOCK

void InputManager_t::ParseGeometryBlock(Tree_t& Tree)
{
	using Except = Exception_t;
	using Code = Except::Code;
	using Cards = GeometryCards;

	for (auto& T : Tree.children) {
		
		auto& object = T.second;
		string card = Parser::Trim(T.first);
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

void InputManager_t::ParseMaterialBlock(Tree_t& Tree)
{

}

void InputManager_t::ParseOptionBlock(Tree_t& Tree)
{

}

/// GEOMETRY CARDS

void InputManager_t::ParseUnitVolumeCard(Tree_t& Tree)
{
	using Except = Exception_t;
	using Code = Except::Code;

	string ORIGIN = "ORIGIN";

	for (auto& T : Tree.children) {
		auto name = Parser::Trim(T.first);
		auto& object = T.second;
		auto contents = Parser::EraseSpace(object.contents);
		auto v = Parser::SplitFields(contents, string(1, SC::SEMICOLON));
		
		UnitVolume_t U;

		for (auto i = v.begin(); i != v.end(); ) {
			auto& s = *i;
			if (s.find(ORIGIN) != string::npos) {
				auto delimiter = string(1, s[ORIGIN.size()]);
				auto u = Parser::SplitFields(s, delimiter);

				if (u.size() != 2) 
					Except::Abort(Code::INVALID_ORIGIN_DATA, object.contents, object.GetLineInfo());
				U.origin = u.back();

				i = v.erase(i);
			}
			else ++i;
		}

		v = Parser::SplitFields(v.front(), SC::DAMPERSAND);

		U.equations = v;

		unitVolumes[name] = U;
	}

}

void InputManager_t::ParseUnitCompCard(Tree_t& Tree)
{
	using Except = Exception_t;
	using Code = Except::Code;

	for (auto& T : Tree.children) {
		auto& object = T.second;
		auto prefix = Parser::EraseSpace(T.first);
		auto v = Parser::SplitFields(prefix, string(1, SC::COLON));
		if (v.size() != 2) 
			Except::Abort(Code::BACKGROUND_MISSED, object.contents, object.GetLineInfo());
		auto name = v.front();
		auto background = v.back();
		auto contents = Parser::EraseSpace(object.contents);
		v = Parser::SplitFields(contents, string(1, SC::SEMICOLON));

		UnitComp_t U;

		U.background = background;

		for (const auto& s : v) {
			auto u = Parser::SplitFields(s, SC::RDBRACKET);
			U.unitvols.push_back(u.front());
			U.displace.push_back(vector<string>());
			u.erase(u.begin());
			for (const auto& k : u) {
				auto lpos = k.find(SC::LPAREN) + 1;
				auto rpos = k.find(SC::RPAREN);
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

// PUBLIC

void InputManager_t::ReadInput(string file)
{
	using Except = Exception_t;
	using Code = Except::Code;

	ifstream fin(file);
	
	if (fin.fail()) Except::Abort(Code::FILE_NOT_FOUND, file);

	stringstream fileStream = ExtractInput(fin);

	fin.close();

	InspectSyntax(fileStream);

	TreeHead.ProcessMacro(fileStream);

	TreeHead.Make(fileStream);

	for (auto& T : TreeHead.children) {
		auto& contents = T.second;
		string block = Parser::Trim(T.first);
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