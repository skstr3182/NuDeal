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

	while (iter != tokens.end()) {
		const auto& token = *iter++;
		Blocks ID = Util::GetBlockID(token.Lexeme());
		switch (ID)
		{
		case Blocks::GEOMETRY :
			ScanGeometryBlock(); break;
		case Blocks::MATERIAL :
			break;
		case Blocks::OPTION :
			break;
		case Blocks::INVALID :
			Except::Abort(Except::Code::INVALID_INPUT_BLOCK, token.Lexeme());
		}
	}

}

void Parser_t::ScanGeometryBlock()
{
	using Cards = GeometryCards;
	++iter;
	while (iter->Lexeme() != string(1, SC::RightBrace)) {
		const auto& token = *iter++;
		Cards ID = Util::GetCardID<Cards>(Blocks::GEOMETRY, token.Lexeme());
		switch (ID)
		{
		case Cards::UNITVOLUME :
			ScanUnitVolumeCard(); break;
		case Cards::UNITCOMP :
			break;
		case Cards::DISPLACE :
			break;
		case Cards::INVALID :
			Except::Abort(Except::Code::INVALID_INPUT_CARD, token.Lexeme());
		}	
	}
	--depth;
}


void Parser_t::ScanUnitVolumeCard()
{
	++iter;
	while (iter->Lexeme() != string(1, SC::RightBrace)) {
		const auto& token = *iter++;
		auto v = GetUnitVolume();
	}
}

Parser_t::UnitVolume_t Parser_t::GetUnitVolume()
{
	UnitVolume_t unitvol;

	++iter;
	while (iter->Lexeme() != string(1, SC::RightBrace)) {
		string s;

	}

	return static_cast<UnitVolume_t&&>(unitvol);
}

}