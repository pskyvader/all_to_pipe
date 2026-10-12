"""
Unit tests for template placeholder substitution and emphasis validation.

Covers the two guards that stop malformed prompts reaching the encoder:
missing/empty placeholders, and emphasis syntax ComfyUI cannot honour.
"""

import sys
import logging
import pytest
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alltopipe_types.template import TemplateParser
from alltopipe_types.prompts import PositivePrompt, NegativePrompt

NL: str = chr(10)
BACKSLASH: str = chr(92)


def make_positive(**fields: str) -> PositivePrompt:
    prompt: PositivePrompt = PositivePrompt()
    for key, value in fields.items():
        setattr(prompt, key, value)
    return prompt


class TestFindMissingPlaceholders:
    def test_all_present_returns_empty(self) -> None:
        prompt = make_positive(characters="a red cat", hair="red hair")
        missing = TemplateParser.find_missing_placeholders(
            "a <characters> with <hair>", prompt
        )
        assert missing == []

    def test_unset_attribute_is_missing(self) -> None:
        prompt = make_positive(characters="a red cat")
        missing = TemplateParser.find_missing_placeholders(
            "a <characters> with <hair>", prompt
        )
        assert missing == ["hair"]

    def test_empty_string_is_missing(self) -> None:
        """An empty field corrupts emphasis exactly like an unset one."""
        prompt = make_positive(characters="a red cat", hair="")
        missing = TemplateParser.find_missing_placeholders(
            "a <characters> with <hair>", prompt
        )
        assert missing == ["hair"]

    def test_whitespace_only_is_missing(self) -> None:
        prompt = make_positive(characters="a cat", hair="   ")
        missing = TemplateParser.find_missing_placeholders("a <hair> cat", prompt)
        assert missing == ["hair"]

    def test_reports_every_missing_in_template_order(self) -> None:
        prompt = make_positive(characters="a cat")
        missing = TemplateParser.find_missing_placeholders(
            "<style> <characters> <lighting> <hair>", prompt
        )
        assert missing == ["style", "lighting", "hair"]

    def test_no_placeholders_returns_empty(self) -> None:
        prompt = make_positive()
        assert TemplateParser.find_missing_placeholders("a cat", prompt) == []

    def test_works_for_negative_prompts(self) -> None:
        prompt = NegativePrompt()
        setattr(prompt, "permanent", "")
        missing = TemplateParser.find_missing_placeholders(
            "<permanent> and <style>", prompt
        )
        assert missing == ["permanent", "style"]


class TestParenDepth:
    def test_balanced(self) -> None:
        assert TemplateParser.paren_depth("(a:1.1) and (b:0.5)") == 0

    def test_unclosed_open(self) -> None:
        assert TemplateParser.paren_depth("a cat (blurry:0.5") == 1

    def test_stray_close_is_negative(self) -> None:
        assert TemplateParser.paren_depth("a cat :1.2)") == -1

    def test_wrong_order_is_negative(self) -> None:
        """Equal counts but malformed, which simple counting would miss."""
        assert TemplateParser.paren_depth("a cat )(") == -1

    def test_no_parens(self) -> None:
        assert TemplateParser.paren_depth("a red cat") == 0

    def test_escaped_parens_are_literal(self) -> None:
        assert (
            TemplateParser.paren_depth(f"literal {BACKSLASH}(x{BACKSLASH}) here") == 0
        )

    def test_nested_emphasis(self) -> None:
        assert TemplateParser.paren_depth("((a:1.1))") == 0


class TestLocate:
    def test_single_line(self) -> None:
        assert TemplateParser.locate("abcdef", 3) == (1, 4)

    def test_second_line(self) -> None:
        assert TemplateParser.locate("ab" + NL + "cd", 3) == (2, 1)

    def test_second_char_of_second_line(self) -> None:
        assert TemplateParser.locate("ab" + NL + "cd", 4) == (2, 2)

    def test_start_of_line(self) -> None:
        assert TemplateParser.locate("ab" + NL + "cd", 3) == (2, 1)


class TestExcerpt:
    def test_brackets_the_span(self) -> None:
        assert TemplateParser.excerpt("a cat, (:1.1) dog", 7, 13) == "a cat, [(:1.1)] dog"

    def test_flattens_newlines(self) -> None:
        """A span containing a newline renders on one line inside the brackets."""
        out = TemplateParser.excerpt("a cat" + NL + "(:1.1) dog", 5, 13)
        assert NL not in out
        assert out == "a cat[ (:1.1) ]dog"

    def test_flattens_newlines_before_the_span(self) -> None:
        """Newlines in the leading context are flattened too."""
        text: str = "a cat" + NL + "on a table (:1.1)"
        start: int = text.index("(")
        end: int = text.index(")") + 1
        out = TemplateParser.excerpt(text, start, end)
        assert NL not in out
        assert out == "a cat on a table [(:1.1)]"

    def test_adds_ellipsis_when_truncated(self) -> None:
        text: str = "x" * 200 + "(:1.1)"
        out = TemplateParser.excerpt(text, 200, 206)
        assert out.startswith("...")

    def test_no_ellipsis_when_short(self) -> None:
        assert "..." not in TemplateParser.excerpt("a (:1.1)", 2, 8)


class TestValidatePrompt:
    @pytest.mark.parametrize(
        "text",
        [
            "a cat (:1.1)",
            f"a cat ({NL}:1.1)",
            f"a cat ({NL}{NL}:1.1)",
            "a cat (\t:1.1)",
            "a cat (\r\n:1.1)",
            "a cat ( : 1.1 )",
        ],
    )
    def test_empty_modifier_raises(self, text: str) -> None:
        with pytest.raises(ValueError, match="no text inside the parentheses"):
            TemplateParser.validate_prompt(text)

    def test_nested_empty_modifier_raises(self) -> None:
        with pytest.raises(ValueError, match="Malformed prompt emphasis"):
            TemplateParser.validate_prompt("((:1.1))")

    def test_empty_parens_raises(self) -> None:
        with pytest.raises(ValueError, match="empty and carries no text"):
            TemplateParser.validate_prompt("a cat ()")

    @pytest.mark.parametrize(
        "text",
        ["a cat (:1.2", "a cat :1.2)", "a cat )(", "a cat (blurry:0.5"],
    )
    def test_unbalanced_raises(self, text: str) -> None:
        with pytest.raises(ValueError, match="unbalanced parentheses"):
            TemplateParser.validate_prompt(text)

    def test_error_reports_line_and_column(self) -> None:
        text: str = NL.join(
            ["masterpiece, best quality,", "a woman with red hair,", "(:1.3) indoors"]
        )
        with pytest.raises(ValueError, match=r"line 3, column 1"):
            TemplateParser.validate_prompt(text)

    def test_error_includes_surrounding_context(self) -> None:
        text: str = "portrait of a woman, (blurry:0.5), standing, (:1.2) indoors"
        with pytest.raises(ValueError) as excinfo:
            TemplateParser.validate_prompt(text)
        message: str = str(excinfo.value)
        assert "[(:1.2)]" in message
        assert "standing" in message
        assert "indoors" in message

    def test_unbalanced_error_explains_escape(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            TemplateParser.validate_prompt("a cat (:1.2")
        message: str = str(excinfo.value)
        assert f"{BACKSLASH}(" in message
        assert "literal characters" in message

    @pytest.mark.parametrize(
        "text",
        [
            "a red cat, sitting, 50mm f/1.8",
            "(blurry:0.5) and (jpeg:0.8)",
            "(a:1.2) standing",
            f"literal {BACKSLASH}(x{BACKSLASH}) here",
            "a cat",
            "",
            f"masterpiece, ({NL}hair:1.1)",
        ],
    )
    def test_valid_text_passes(self, text: str) -> None:
        TemplateParser.validate_prompt(text)


class TestStripMalformedPrompt:
    def test_removes_empty_modifier(self) -> None:
        cleaned, removed = TemplateParser.strip_malformed_prompt("a cat, (:1.1)")
        assert ":1.1" not in cleaned
        assert len(removed) == 1
        assert "no text" in removed[0]

    def test_removes_empty_parens(self) -> None:
        cleaned, removed = TemplateParser.strip_malformed_prompt("a cat ()")
        assert "()" not in cleaned
        assert len(removed) == 1

    def test_removes_unmatched_open(self) -> None:
        cleaned, removed = TemplateParser.strip_malformed_prompt("a cat (:1.2")
        assert "(" not in cleaned
        assert "unmatched '('" in removed[0]

    def test_removes_unmatched_close(self) -> None:
        cleaned, removed = TemplateParser.strip_malformed_prompt("a cat :1.2)")
        assert ")" not in cleaned
        assert "unmatched ')'" in removed[0]

    def test_reports_line_and_column(self) -> None:
        text: str = NL.join(["masterpiece,", "a woman with red hair,", "(:1.3) indoors"])
        _cleaned, removed = TemplateParser.strip_malformed_prompt(text)
        assert "line 3, column 1" in removed[0]

    def test_keeps_valid_emphasis_untouched(self) -> None:
        cleaned, removed = TemplateParser.strip_malformed_prompt(
            "a cat (blurry:0.5) and (jpeg:0.8)"
        )
        assert cleaned == "a cat (blurry:0.5) and (jpeg:0.8)"
        assert removed == []

    def test_keeps_escaped_parens(self) -> None:
        text: str = f"literal {BACKSLASH}(x{BACKSLASH}) here"
        cleaned, removed = TemplateParser.strip_malformed_prompt(text)
        assert cleaned == text
        assert removed == []

    def test_valid_text_produces_no_removals(self) -> None:
        cleaned, removed = TemplateParser.strip_malformed_prompt("a red cat, 50mm f/1.8")
        assert cleaned == "a red cat, 50mm f/1.8"
        assert removed == []

    def test_nested_empty_modifier_is_fully_removed(self) -> None:
        cleaned, _removed = TemplateParser.strip_malformed_prompt("a cat ((:1.1))")
        assert ":1.1" not in cleaned
        assert "()" not in cleaned


class TestParseTemplate:
    def test_substitutes_present_values(self) -> None:
        prompt = make_positive(characters="a red cat", hair="red hair")
        result = TemplateParser.parse_template(
            "a <characters> with <hair>", prompt, allow_parse_errors=True
        )
        assert result == "a a red cat with red hair"

    def test_allow_parse_errors_substitutes_empty(self) -> None:
        prompt = make_positive(characters="a red cat")
        result = TemplateParser.parse_template(
            "a <characters> with <hair>", prompt, allow_parse_errors=True
        )
        assert result == "a a red cat with"

    def test_missing_placeholder_raises_when_not_allowed(self) -> None:
        """The <MISSING x> literal must never reach the encoder."""
        prompt = make_positive(characters="a red cat")
        with pytest.raises(ValueError, match="<hair>"):
            TemplateParser.parse_template(
                "a <characters> with <hair>", prompt, allow_parse_errors=False
            )

    def test_empty_field_raises_when_not_allowed(self) -> None:
        prompt = make_positive(characters="a red cat", hair="")
        with pytest.raises(ValueError, match="<hair>"):
            TemplateParser.parse_template(
                "a <characters> with <hair>", prompt, allow_parse_errors=False
            )

    def test_error_lists_every_missing_placeholder(self) -> None:
        prompt = make_positive(characters="a cat")
        with pytest.raises(ValueError) as excinfo:
            TemplateParser.parse_template(
                "<style> and <lighting>", prompt, allow_parse_errors=False
            )
        message: str = str(excinfo.value)
        assert "<style>" in message
        assert "<lighting>" in message

    def test_empty_field_raises_missing_placeholder_first(self) -> None:
        """With the toggle off, an empty field is reported as a missing value.

        This is checked before emphasis validation, so the user is told about
        the unset field rather than the symptom it produced.
        """
        prompt = make_positive(characters="a red cat", hair="")
        with pytest.raises(ValueError, match="<hair> but no usable value"):
            TemplateParser.parse_template(
                "a <characters>, (<hair>:1.2)", prompt, allow_parse_errors=False
            )

    def test_malformed_modifier_raises_when_not_allowed(self) -> None:
        """A hand-written bad modifier raises when errors are not tolerated."""
        prompt = make_positive(characters="a red cat")
        with pytest.raises(ValueError, match="no text inside the parentheses"):
            TemplateParser.parse_template(
                "a <characters>, (:1.2) sitting", prompt, allow_parse_errors=False
            )

    def test_unbalanced_raises_when_not_allowed(self) -> None:
        prompt = make_positive(characters="a red cat")
        with pytest.raises(ValueError, match="unbalanced parentheses"):
            TemplateParser.parse_template(
                "a <characters> (:1.2", prompt, allow_parse_errors=False
            )

    def test_malformed_modifier_is_repaired_when_allowed(self, caplog: pytest.LogCaptureFixture) -> None:
        """Allowing errors removes the bad fragment and warns instead of raising."""
        prompt = make_positive(characters="a red cat")
        with caplog.at_level(logging.WARNING):
            result = TemplateParser.parse_template(
                "a <characters>, (:1.2) sitting", prompt, allow_parse_errors=True
            )
        # The author's own comma survives; only the broken modifier is removed.
        assert result == "a a red cat, sitting"
        assert "(:1.2)" not in result
        assert "Malformed prompt emphasis was removed" in caplog.text
        assert "(:1.2)" in caplog.text

    def test_unbalanced_is_repaired_when_allowed(self) -> None:
        prompt = make_positive(characters="a red cat")
        result = TemplateParser.parse_template(
            "a <characters> (:1.2", prompt, allow_parse_errors=True
        )
        assert "(" not in result

    def test_missing_placeholder_warns_when_allowed(self, caplog: pytest.LogCaptureFixture) -> None:
        prompt = make_positive(characters="a red cat")
        with caplog.at_level(logging.WARNING):
            result = TemplateParser.parse_template(
                "a <characters> with <hair>", prompt, allow_parse_errors=True
            )
        assert result == "a a red cat with"
        assert "<hair>" in caplog.text

    def test_valid_prompt_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        prompt = make_positive(characters="a red cat", hair="red hair")
        with caplog.at_level(logging.WARNING):
            result = TemplateParser.parse_template(
                "a <characters> with <hair>", prompt, allow_parse_errors=True
            )
        assert result == "a a red cat with red hair"
        assert caplog.text == ""

    def test_whitespace_is_flattened(self) -> None:
        prompt = make_positive(characters="a red cat")
        result = TemplateParser.parse_template(
            f"a <characters>,{NL}\tsitting   on a  table", prompt, allow_parse_errors=True
        )
        assert result == "a a red cat, sitting on a table"

    def test_crlf_is_flattened(self) -> None:
        prompt = make_positive(characters="a red cat")
        result = TemplateParser.parse_template(
            "a <characters>\r\n(blurry:0.5)", prompt, allow_parse_errors=True
        )
        assert result == "a a red cat (blurry:0.5)"

    def test_commas_are_normalised(self) -> None:
        prompt = make_positive(characters="a red cat")
        result = TemplateParser.parse_template(
            "a <characters> ,,  sitting", prompt, allow_parse_errors=True
        )
        assert ",," not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
