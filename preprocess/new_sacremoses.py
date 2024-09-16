#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

from six import text_type

from sacremoses.corpus import Perluniprops
from sacremoses.corpus import NonbreakingPrefixes
from sacremoses.util import is_cjk
from sacremoses.indic import VIRAMAS, NUKTAS

perluniprops = Perluniprops()
nonbreaking_prefixes = NonbreakingPrefixes()


class MosesDetokenizer(object):
    """
    This is a Python port of the Moses Detokenizer from
    https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/detokenizer.perl
    """

    # Currency Symbols.
    IsAlnum = text_type("".join(perluniprops.chars("IsAlnum")))
    IsAlpha = text_type("".join(perluniprops.chars("IsAlpha")))
    IsSc = text_type("".join(perluniprops.chars("IsSc")))

    AGGRESSIVE_HYPHEN_SPLIT = r" \@\-\@ ", r"-"

    # Merge multiple spaces.
    ONE_SPACE = re.compile(r" {2,}"), " "

    # Unescape special characters.
    UNESCAPE_FACTOR_SEPARATOR = r"&#124;", r"|"
    UNESCAPE_LEFT_ANGLE_BRACKET = r"&lt;", r"<"
    UNESCAPE_RIGHT_ANGLE_BRACKET = r"&gt;", r">"
    UNESCAPE_DOUBLE_QUOTE = r"&quot;", r'"'
    UNESCAPE_SINGLE_QUOTE = r"&apos;", r"'"
    UNESCAPE_SYNTAX_NONTERMINAL_LEFT = r"&#91;", r"["
    UNESCAPE_SYNTAX_NONTERMINAL_RIGHT = r"&#93;", r"]"
    UNESCAPE_AMPERSAND = r"&amp;", r"&"
    # The legacy regexes are used to support outputs from older Moses versions.
    UNESCAPE_FACTOR_SEPARATOR_LEGACY = r"&bar;", r"|"
    UNESCAPE_SYNTAX_NONTERMINAL_LEFT_LEGACY = r"&bra;", r"["
    UNESCAPE_SYNTAX_NONTERMINAL_RIGHT_LEGACY = r"&ket;", r"]"

    MOSES_UNESCAPE_XML_REGEXES = [
        UNESCAPE_FACTOR_SEPARATOR_LEGACY,
        UNESCAPE_FACTOR_SEPARATOR,
        UNESCAPE_LEFT_ANGLE_BRACKET,
        UNESCAPE_RIGHT_ANGLE_BRACKET,
        UNESCAPE_SYNTAX_NONTERMINAL_LEFT_LEGACY,
        UNESCAPE_SYNTAX_NONTERMINAL_RIGHT_LEGACY,
        UNESCAPE_DOUBLE_QUOTE,
        UNESCAPE_SINGLE_QUOTE,
        UNESCAPE_SYNTAX_NONTERMINAL_LEFT,
        UNESCAPE_SYNTAX_NONTERMINAL_RIGHT,
        UNESCAPE_AMPERSAND,
    ]

    FINNISH_MORPHSET_1 = [
        "N",
        "n",
        "A",
        "a",
        "\xc4",
        "\xe4",
        "ssa",
        "Ssa",
        "ss\xe4",
        "Ss\xe4",
        "sta",
        "st\xe4",
        "Sta",
        "St\xe4",
        "hun",
        "Hun",
        "hyn",
        "Hyn",
        "han",
        "Han",
        "h\xe4n",
        "H\xe4n",
        "h\xf6n",
        "H\xf6n",
        "un",
        "Un",
        "yn",
        "Yn",
        "an",
        "An",
        "\xe4n",
        "\xc4n",
        "\xf6n",
        "\xd6n",
        "seen",
        "Seen",
        "lla",
        "Lla",
        "ll\xe4",
        "Ll\xe4",
        "lta",
        "Lta",
        "lt\xe4",
        "Lt\xe4",
        "lle",
        "Lle",
        "ksi",
        "Ksi",
        "kse",
        "Kse",
        "tta",
        "Tta",
        "ine",
        "Ine",
    ]

    FINNISH_MORPHSET_2 = ["ni", "si", "mme", "nne", "nsa"]

    FINNISH_MORPHSET_3 = [
        "ko",
        "k\xf6",
        "han",
        "h\xe4n",
        "pa",
        "p\xe4",
        "kaan",
        "k\xe4\xe4n",
        "kin",
    ]

    FINNISH_REGEX = r"^({})({})?({})$".format(
        text_type("|".join(FINNISH_MORPHSET_1)),
        text_type("|".join(FINNISH_MORPHSET_2)),
        text_type("|".join(FINNISH_MORPHSET_3)),
    )

    def __init__(self, lang="en"):
        super(MosesDetokenizer, self).__init__()
        self.lang = lang

    def unescape_xml(self, text):
        for regexp, substitution in self.MOSES_UNESCAPE_XML_REGEXES:
            text = re.sub(regexp, substitution, text)
        return text

    def tokenize(self, tokens, return_str=True, unescape=True):
        """
        Python port of the Moses detokenizer.
        :param tokens: A list of strings, i.e. tokenized text.
        :type tokens: list(str)
        :return: str
        """
        # Convert the list of tokens into a string and pad it with spaces.
        text = r" {} ".format(" ".join(tokens))
        # Converts input string into unicode.
        text = text_type(text)
        # Detokenize the agressive hyphen split.
        regexp, substitution = self.AGGRESSIVE_HYPHEN_SPLIT
        text = re.sub(regexp, substitution, text)
        if unescape:
            # Unescape the XML symbols.
            text = self.unescape_xml(text)
        # Keep track of no. of quotation marks.
        quote_counts = {"'": 0, '"': 0, "``": 0, "`": 0, "''": 0}

        # The *prepend_space* variable is used to control the "effects" of
        # detokenization as the function loops through the list of tokens and
        # changes the *prepend_space* accordingly as it sequentially checks
        # through the language specific and language independent conditions.
        prepend_space = " "
        detokenized_text = ""
        tokens = text.split()
        # Iterate through every token and apply language specific detokenization rule(s).
        for i, token in enumerate(iter(tokens)):
            # Check if the first char is CJK.
            if is_cjk(token[0]) and self.lang != "ko":
                # Perform left shift if this is a second consecutive CJK word.
                if i > 0 and is_cjk(tokens[i - 1][-1]):
                    detokenized_text += token
                # But do nothing special if this is a CJK word that doesn't follow a CJK word
                else:
                    detokenized_text += prepend_space + token
                prepend_space = " "
            # If it's a currency symbol.
            elif re.search(r"^[" + self.IsSc + r"\(\[\{\¿\¡]+$", token):
                # Perform right shift on currency and other random punctuation items
                detokenized_text += prepend_space + token
                prepend_space = ""

            elif re.search(r"^[\,\.\?\!\:\;\\\%\}\]\)]+$", token):
                # In French, these punctuations are prefixed with a non-breakable space.
                if self.lang == "fr" and re.search(r"^[\?\!\:\;\\\%]$", token):
                    detokenized_text += " "
                # Perform left shift on punctuation items.
                detokenized_text += token
                prepend_space = " "

            elif (
                self.lang == "en"
                and i > 0
                and re.search(r"^['][{}]".format(self.IsAlpha), token)
            ):
                # and re.search(u'[{}]$'.format(self.IsAlnum), tokens[i-1])):
                # For English, left-shift the contraction.
                detokenized_text += token
                prepend_space = " "

            elif (
                self.lang == "cs"
                and i > 1
                and re.search(
                    r"^[0-9]+$", tokens[-2]
                )  # If the previous previous token is a number.
                and re.search(r"^[.,]$", tokens[-1])  # If previous token is a dot.
                and re.search(r"^[0-9]+$", token)
            ):  # If the current token is a number.
                # In Czech, left-shift floats that are decimal numbers.
                detokenized_text += token
                prepend_space = " "

            elif (
                self.lang in ["fr", "it", "ga"]
                and i <= len(tokens) - 2
                and re.search(r"[{}][']$".format(self.IsAlpha), token)
                and re.search(r"^[{}]".format(self.IsAlpha), tokens[i + 1])
            ):  # If the next token is alpha.
                # For French and Italian, right-shift the contraction.
                detokenized_text += prepend_space + token
                prepend_space = ""

            elif (
                self.lang == "cs"
                and i <= len(tokens) - 3
                and re.search(r"[{}][']$".format(self.IsAlpha), token)
                and re.search(r"^[-–]$", tokens[i + 1])
                and re.search(r"^li$|^mail.*", tokens[i + 2], re.IGNORECASE)
            ):  # In Perl, ($words[$i+2] =~ /^li$|^mail.*/i)
                # In Czech, right-shift "-li" and a few Czech dashed words (e.g. e-mail)
                detokenized_text += prepend_space + token + tokens[i + 1]
                next(tokens, None)  # Advance over the dash
                prepend_space = ""

            # Combine punctuation smartly.
            elif re.search(r"""^[\'\"„“`]+$""", token):
                normalized_quo = token
                if re.search(r"^[„“”]+$", token):
                    normalized_quo = '"'
                quote_counts[normalized_quo] = quote_counts.get(normalized_quo, 0)

                if self.lang == "cs" and token == "„":
                    quote_counts[normalized_quo] = 0
                if self.lang == "cs" and token == "“":
                    quote_counts[normalized_quo] = 1

                # change from here
                if (
                    self.lang == "en"
                    and token == "'"
                    and i > 0
                    and tokens[i - 1].endswith("s")
                    and tokens[i - 1][0].isupper()
                ):
                    # Left shift on single quote for possessives ending
                    # in "s", e.g. "The Jones' house"
                    detokenized_text += token
                    prepend_space = " "
                elif (
                    self.lang == "en"
                    and token == "'"
                    and i > 0
                    and i + 1 < len(tokens)
                    and re.search(r"[{}]".format(self.IsAlpha), tokens[i - 1])
                    and tokens[i + 1] == "s"
                ):
                    # Left shift on single quote for possessives
                    # e.g. "The Hagrid's house"
                    detokenized_text += token
                    prepend_space = ""

                elif quote_counts[normalized_quo] % 2 == 0:
                    if (
                        self.lang == "en"
                        and token == "'"
                        and i > 0
                        and tokens[i - 1].endswith("s")
                    ):
                        # Left shift on single quote for possessives ending
                        # in "s", e.g. "The Jones' house"
                        detokenized_text += token
                        prepend_space = " "
                    else:
                        # Right shift.
                        detokenized_text += prepend_space + token
                        prepend_space = ""
                        quote_counts[normalized_quo] += 1
                else:
                    # Left shift.
                    detokenized_text += token
                    prepend_space = " "
                    quote_counts[normalized_quo] += 1

            elif (
                self.lang == "fi"
                and re.search(r":$", tokens[i - 1])
                and re.search(self.FINNISH_REGEX, token)
            ):
                # Finnish : without intervening space if followed by case suffix
                # EU:N EU:n EU:ssa EU:sta EU:hun EU:iin ...
                detokenized_text += prepend_space + token
                prepend_space = " "

            else:
                detokenized_text += prepend_space + token
                prepend_space = " "

        # Merge multiple spaces.
        regexp, substitution = self.ONE_SPACE
        detokenized_text = re.sub(regexp, substitution, detokenized_text)
        # Removes heading and trailing spaces.
        detokenized_text = detokenized_text.strip()

        return detokenized_text if return_str else detokenized_text.split()

    def detokenize(self, tokens, return_str=True, unescape=True):
        """Duck-typing the abstract *tokenize()*."""
        return self.tokenize(tokens, return_str, unescape)
