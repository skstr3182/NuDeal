#include "Parser.h"
#include "IOUtil.h"
#include "Exception.h"
#include "Lexer.h"

namespace IO
{

Parser_t::Parser_t(const Lexer_t *Lexer) noexcept : tokens(Lexer->Tokens()) {}

void Parser_t::Parse()
{
	this->depth = 0;

	iter = tokens.begin();

	while (iter++ != tokens.end()) {
		const auto& token = iter->Lexeme();
		Blocks ID = Util::GetBlockID(token);
		switch (ID)
		{
		case Blocks::GEOMETRY :
			ScanGeometryBlock(); break;
		case Blocks::MATERIAL :
			break;
		case Blocks::OPTION :
			break;
		case Blocks::INVALID :
			Except::Abort(Except::Code::INVALID_INPUT_BLOCK, token);
		}
	}

}

void Parser_t::ScanGeometryBlock()
{
	using Cards = GeometryCards;
	++iter;
	while (iter++->Lexeme() != string(1, SC::RightBrace)) {
		const auto& token = iter->Lexeme();
		Cards ID = Util::GetCardID<Cards>(Blocks::GEOMETRY, token);
		switch (ID)
		{
		case Cards::UNITVOLUME :
			ScanUnitVolumeCard(); break;
		case Cards::UNITCOMP :
			break;
		case Cards::DISPLACE :
			break;
		case Cards::INVALID :
			Except::Abort(Except::Code::INVALID_INPUT_CARD, token);
		}	
	}
}


void Parser_t::ScanUnitVolumeCard()
{
	++iter;
	while (iter++->Lexeme() != string(1, SC::RightBrace)) {
		const auto& name = iter->Lexeme();
		auto v = GetUnitVolume();
	}
}

Parser_t::UnitVolume_t Parser_t::GetUnitVolume()
{
	UnitVolume_t unitvol;

	++iter;
	while (iter++->Lexeme() != string(1, SC::RightBrace)) {
		const auto& token = iter->Lexeme();
		if (!Util::Uppercase(token).compare("ORIGIN")) {
			string s;
			auto begin = iter;
			auto end = iter;
			for (;; ++iter) {

			}
			
		}

	}

	return static_cast<UnitVolume_t&&>(unitvol);
}

}