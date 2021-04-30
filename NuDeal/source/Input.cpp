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
	
	string contents = fileStream.str();

	std::replace(contents.begin(), contents.end(), SC::TAB, SC::BLANK);
	std::replace(contents.begin(), contents.end(), SC::CR, SC::BLANK);

	fileStream.str("");
	fileStream.str(contents);

	line_contents = SplitFields(contents, string(1, SC::LF));

	return static_cast<stringstream&&>(fileStream);
}

void InputManager_t::Tree_t::MakeInputTree(const vector<string>& in,
	vector<string>::const_iterator iter)
{
	using Except = Exception_t;
	using Code = Except::Code;

	string contents;

	while (iter != in.end()) {
		string oneline = Trim(*iter);
		contents += oneline;
		size_t line_info = iter - in.begin();
		
		if (contents[0] == SC::HASHTAG) {
			while (contents.back() == '\\' && iter != in.end()) contents += Trim(*(iter++));
			if (contents.back() == '\\') Except::Abort("");
			auto pos = contents.find_first_of(SC::BLANK);
			auto directive = contents.substr(0, pos);
			if (directive == "#define") {
				auto p = contents.find_last_of(SC::RPAREN, pos) + pos;
				auto name = contents.substr(pos, p);
				auto macro = contents.substr(p + 1);
				this->define[name] = macro;
			}
		}
		else {
			auto lcount = std::count(contents.begin(), contents.end(), SC::LBRACE);
			auto rcount = std::count(contents.begin(), contents.end(), SC::RBRACE);
			while (lcount != rcount && iter != in.end()) {
				contents += Trim(*(iter++));

			}
		}



	}
}

void InputManager_t::Tree_t::MakeInputTree(stringstream& in, size_t offset)
{
	using Except = Exception_t;
	using Code = Except::Code;

	do {
		auto contents = GetContentsBlock(in);
		
		if (Trim(contents).empty()) continue;

		if (contents[0] == SC::HASHTAG) {
			auto pos = contents.find_first_of(SC::BLANK);
			auto directive = contents.substr(0, pos);
			if (directive == "#define") {
				auto name = contents.substr(0, pos);
				auto macro = contents.substr(pos + 1);
				this->define[name] = macro;
			}

			offset += LineCount(contents);
		}
		else {
			auto next = contents;
			auto pos = contents.find_first_of(SC::LBRACE);
			auto name = contents.substr(0, pos);
			next.erase(0, pos);
			auto& T = this->children[name];
			T.parent = this;

			offset += std::count(name.begin(), name.begin() + name.find_first_not_of(SC::LF), SC::LF);;
			T.line_info = offset;
			offset += std::count(name.begin() + name.find_last_not_of(SC::LF), name.end(), SC::LF);
			
			next.erase(next.find_first_of(SC::LBRACE), 1);
			next.erase(next.find_last_of(SC::RBRACE), 1);
			
			pos = next.find_first_not_of(SC::LF);
			offset += std::count(next.begin(), next.begin() + next.find_first_not_of(SC::LF), SC::LF);
			next.erase(0, pos);

			auto count = std::count(next.begin(), next.end(), SC::LBRACE);
			
			if (count == 0)
				T.contents = next;
			else
				T.MakeInputTree(stringstream(next), offset);

			offset += LineCount(next);
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

void InputManager_t::ParseMaterialBlock(Tree_t& Tree)
{

}

void InputManager_t::ParseOptionBlock(Tree_t& Tree)
{

}

// GEOMETRY CARDS

void InputManager_t::ParseUnitVolumeCard(Tree_t& Tree)
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

void InputManager_t::ParseUnitCompCard(Tree_t& Tree)
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

	TreeHead.MakeInputTree(fileStream);

	for (auto& T : TreeHead.children) {
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