#include "Input.h"
#include "IOUtil.h"
#include "Exception.h"

namespace IO
{

InputManager_t::Blocks InputManager_t::GetBlockID(string line) const
{
	int pos_end = line.find(SC::BLANK, 0);
	string block = line.substr(0, pos_end);

	Uppercase(block);
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

	Uppercase(card);
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
	
	contents = fileStream.str();

	std::replace(contents.begin(), contents.end(), SC::TAB, SC::BLANK);
#ifdef __linux__
	std::replace(contents.begin(), contents.end(), SC::CR, SC::BLANK);
#endif

	contents.erase(contents.find_last_not_of(SC::BLANK), contents.size());

	Uppercase(contents);

	fileStream.str("");
	fileStream.str(contents);

	return static_cast<stringstream&&>(fileStream);
}

void InputManager_t::MakeInputTree(stringstream& in, Tree_t& Tree, size_t offset)
{
	using Except = Exception_t;
	using Code = Except::Code;

	static auto ONE = static_cast<std::streampos>(1);

	do {

		auto prefix = GetLine(in, SC::LBRACE); 
		if (Trim(prefix).empty()) continue;
		in.seekg(in.tellg() - ONE);
		auto body = GetScriptBlock(in);
		if (Trim(body).empty()) continue;

		prefix = Trim(prefix, string(1, SC::BLANK));

		auto count = std::count(body.begin(), body.end(), SC::LBRACE);

		if (count != std::count(body.begin(), body.end(), SC::RBRACE))
			Except::Abort(Code::MISMATCHED_BRAKETS, body);

		body.erase(0, body.find_first_of(SC::LBRACE) + 1);
		body.erase(body.find_last_of(SC::RBRACE));
		--count;

		auto& T = Tree[prefix];

		offset += std::count(prefix.begin(), 
			prefix.begin() + prefix.find_first_not_of(SC::LF), SC::LF);

		T.line_info = offset;

		offset += std::count(prefix.begin() + prefix.find_last_not_of(SC::LF) + 1, 
			prefix.end(), SC::LF);

		if (count > 0) 
			MakeInputTree(stringstream(body), T.children, offset);
		else 
			T.contents = body;

		offset += LineCount(body);

	} while (!in.eof());

}

// BLOCK

void InputManager_t::ParseGeometryBlock(InputTree_t& Tree)
{
	using Except = Exception_t;
	using Code = Except::Code;
	using Cards = GeometryCards;

	for (auto& T : Tree.children) {
		
		auto& object = T.second;
		string card = Trim(T.first);
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

void InputManager_t::ParseMaterialBlock(InputTree_t& Tree)
{

}

void InputManager_t::ParseOptionBlock(InputTree_t& Tree)
{

}

// GEOMETRY CARDS

void InputManager_t::ParseUnitVolumeCard(InputTree_t& Tree)
{
	using Except = Exception_t;
	using Code = Except::Code;

	string ORIGIN = "ORIGIN";

	for (auto& T : Tree.children) {
		auto name = Trim(T.first);
		auto& object = T.second;
		auto contents = EraseSpace(object.contents);
		auto v = SplitFields(contents, string(1, SC::SEMICOLON));
		
		UnitVolume_t U;

		for (auto i = v.begin(); i != v.end(); ) {
			auto& s = *i;
			if (s.find(ORIGIN) != string::npos) {
				auto delimiter = string(1, s[ORIGIN.size()]);
				auto u = SplitFields(s, delimiter);

				if (u.size() != 2) 
					Except::Abort(Code::INVALID_ORIGIN_DATA, object.contents, object.GetLineInfo());
				U.origin = u.back();

				i = v.erase(i);
			}
			else ++i;
		}

		v = SplitFields(v.front(), SC::DAMPERSAND);

		U.equations = v;

		unitVolumes[name] = U;
	}

}

void InputManager_t::ParseUnitCompCard(InputTree_t& Tree)
{
	using Except = Exception_t;
	using Code = Except::Code;

	for (auto& T : Tree.children) {
		auto& object = T.second;
		auto prefix = EraseSpace(T.first);
		auto v = SplitFields(prefix, string(1, SC::COLON));
		if (v.size() != 2) 
			Except::Abort(Code::BACKGROUND_MISSED, object.contents, object.GetLineInfo());
		auto name = v.front();
		auto background = v.back();
		auto contents = EraseSpace(object.contents);
		v = SplitFields(contents, string(1, SC::SEMICOLON));

		UnitComp_t U;

		U.background = background;

		for (const auto& s : v) {
			auto u = SplitFields(s, SC::RDBRACKET);
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

	MakeInputTree(fileStream, TreeHead);

	for (auto& T : TreeHead) {
		auto& contents = T.second;
		string block = Trim(T.first);
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