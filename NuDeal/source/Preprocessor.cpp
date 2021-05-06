#include "Preprocessor.h"
#include "IOUtil.h"

namespace IO
{

const string Preprocessor_t::macro_str = 
	R"([\b]*)"									// whitespace
	R"((#[A-z]{2,}))"						// <#define>
	R"(\s+)"										// whitespace
	R"((\w+))"									// <Identifier>
	R"((\(.*\))?)"							// <Argument>
	R"(\s+)"										// whitespace
	R"(((?:.*\\\s*\r?\n)*.*))";	// <Contents>

const regex Preprocessor_t::macro_re = regex(macro_str);

void Preprocessor_t::DeleteComment(string& contents)
{
	static constexpr char open[] = "/*", close[] = "*/";
	static constexpr char comment[] = "//";

	try {
		CheckBalance(contents, { string(open) }, { string(close) });
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
			[](char c) { return c != SC::LF; }, SC::Blank);
		pos = end + strlen(close) + 1;
	}
}

void Preprocessor_t::ApplyMacro(string& contents)
{
	auto beg = sregex_token_iterator(contents.begin(), contents.end(), macro_re);
	sregex_token_iterator end;
	vector<string> macro;

	// Pull Out

	for (auto iter = beg; iter != end; ++iter) {
		auto& s = iter->str();
		auto pos = contents.find(s), size = s.size();
		std::replace_if(s.begin(), s.end(),
			[](char c) { return c == SC::BackSlash || c == SC::LF; }, SC::Blank);
		macro.push_back(s);
		std::replace_if(contents.begin() + pos, contents.begin() + pos + size,
			[](char c) { return c != SC::LF; }, SC::Blank);

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

		arguments = Util::EraseSpace(arguments, {SC::Blank, SC::LF, SC::LeftParen, SC::RightParen} );
		auto v_args = Util::SplitFields(arguments, {SC::Comma});
		int num_args = v_args.size();

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
				regex re(R"(\b)" + arg + R"(\b)");
				replace = regex_replace(replace, re, val);
			}
			contents.replace(beg->first, beg->second, replace);
		}
	}
}

void Preprocessor_t::CheckBalance(const string& contents,
	const vector<string>& open,
	const vector<string>& close)
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
}

}