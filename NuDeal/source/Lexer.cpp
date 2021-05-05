#include "Lexer.h"

namespace IO
{

Lexer_t::Lexer_t()
{

	int p;

	p = static_cast<int>(TokenType::KEYWORD);

	for (const auto& block : BlockNames) 
		tokens[p].push_back(block);
	

	for (const auto& Cards : CardNames) 
		for (const auto& card : Cards)
			tokens[p].push_back(card);
		
	v_regex.resize(tokens.size());

	for (int i = 0; i < v_regex.size(); ++i) {
		v_regex[i].resize(tokens[i].size());
		for (int j = 0; j < v_regex[i].size(); ++j) {
			v_regex[i][j] = regex(tokens[i][j]);
		}
	}


	string input = "GEOMETRY";

	smatch match;

	//for (const auto& i : input) {
		for (const auto& j : v_regex) {
			for (const auto& k : j) {
				if (std::regex_match(input, match, k)) {
					for (auto i = 0; i < match.size(); ++i) {
						std::cout << "Match : " << match[i].str() << endl;
					}
				}
			}
		}
	//}

}

}