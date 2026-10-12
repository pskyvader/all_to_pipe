"""
All-to-Pipe prompt template parser.

System for parsing text templates with variable substitution from prompt objects.
"""

import re
import logging
from .prompts import PositivePrompt, NegativePrompt


class Template:
    def __init__(
        self,
        template_type: str,
        placeholders: list[str],
        text: str,
        allow_parse_errors: bool,
        decay: float,
        decay_floor: float,
    ) -> None:

        if template_type not in ["positive", "negative"]:
            raise ValueError("template_type must be 'positive' or 'negative'")

        # If template is empty, skip
        if not text.strip():
            raise ValueError("template_text cannot be empty")

        self.type: str = template_type
        self.placeholders: list[str] = placeholders
        self.text: str = text
        self.allow_parse_errors: bool = allow_parse_errors
        self.parsed_template: str | None = None
        self.decay: float = decay
        self.decay_floor: float = decay_floor


class TemplateParser:
    """
    Parser for prompt templates with variable substitution.

    Supports templates with <variable> placeholders that are substituted
    from PositivePrompt and NegativePrompt attributes.

    Uses angle brackets <> to avoid conflicts with comfyui-dynamic-prompts module
    which uses {}, __term__, $, (), | and other symbols.
    """

    # Regex pattern for finding placeholders with <variable> syntax
    PLACEHOLDER_PATTERN = re.compile(r"<([^>]+)>")

    # Emphasis modifiers ComfyUI cannot honour: a weight with no content inside
    # the parentheses. \s* tolerates the newlines/tabs that multiline templates
    # contain, so these are caught before or after whitespace flattening alike.
    EMPTY_MODIFIER = re.compile(r"\(\s*:\s*[\d.]*\s*\)")
    EMPTY_PARENS = re.compile(r"\(\s*\)")

    CONTEXT_CHARS: int = 24

    @staticmethod
    def find_placeholders(template: str) -> list[str]:
        """
        Find all placeholders in a template string.

        Args:
            template: Template string with <variable> placeholders

        Returns:
            List of variable names found in the template

        Raises:
            ValueError: If template is None or not a string

        Example:
            >>> TemplateParser.find_placeholders("A <age> person")
            ['age']
        """
        matches = TemplateParser.PLACEHOLDER_PATTERN.findall(template)
        return matches

    @staticmethod
    def find_missing_placeholders(
        template: str, prompt_map: PositivePrompt | NegativePrompt
    ) -> list[str]:
        """
        Find placeholders the prompt object cannot supply a value for.

        Catches both attributes that were never set and attributes set to
        empty/whitespace text, since either one silently produces malformed
        emphasis syntax downstream.

        Args:
            template: Template string with <variable> placeholders
            prompt_map: PositivePrompt or NegativePrompt to read values from

        Returns:
            List of placeholder names with no usable value, in template order

        Example:
            >>> TemplateParser.find_missing_placeholders("A <age> <hair> cat", pos)
            ['hair']
        """
        missing: list[str] = []
        for placeholder in TemplateParser.find_placeholders(template):
            value: str | None = getattr(prompt_map, placeholder, None)
            if value is None or not str(value).strip():
                missing.append(placeholder)
        return missing

    @staticmethod
    def paren_depth(text: str) -> int:
        """
        Return the final parenthesis depth of a string.

        Returns 0 when balanced, a positive count of unclosed '(' characters,
        or -1 if a ')' appears with no matching '(' before it.

        Walks the string rather than using a regex so backslash escapes are
        honoured: \\( and \\) are literal characters, not emphasis markers.

        Args:
            text: Template text to scan

        Returns:
            Final depth, or -1 when depth went negative

        Example:
            >>> TemplateParser.paren_depth("(a:1.1) and (b:0.5)")
            0
        """
        depth: int = 0
        index: int = 0
        while index < len(text):
            char: str = text[index]
            if char == "\\" and index + 1 < len(text):
                index += 2
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    return -1
            index += 1
        return depth

    @staticmethod
    def locate(text: str, index: int) -> tuple[int, int]:
        """
        Convert a character offset into a 1-based line and column.

        Args:
            text: Template text the offset refers to
            index: Character offset within text

        Returns:
            Tuple of (line, column), both 1-based

        Example:
            >>> TemplateParser.locate("ab\\ncd", 3)
            (2, 2)
        """
        line: int = text.count("\n", 0, index) + 1
        column: int = index - (text.rfind("\n", 0, index) + 1) + 1
        return line, column

    @classmethod
    def excerpt(cls, text: str, start: int, end: int) -> str:
        """
        Render a span of text in context, bracketed, for error messages.

        Newlines inside the excerpt are flattened to spaces so the message stays
        readable when the template spans multiple lines.

        Args:
            text: Template text to excerpt
            start: Character offset where the offending span begins
            end: Character offset where the offending span ends

        Returns:
            Context window with the offending span wrapped in square brackets

        Example:
            >>> TemplateParser.excerpt("a cat, (:1.1) dog", 7, 13)
            'a cat, [(:1.1)] dog'
        """
        left: int = max(0, start - cls.CONTEXT_CHARS)
        right: int = min(len(text), end + cls.CONTEXT_CHARS)
        prefix: str = "..." if left > 0 else ""
        suffix: str = "..." if right < len(text) else ""
        return (
            prefix + text[left:start].replace("\n", " ")
            + "[" + text[start:end].replace("\n", " ") + "]"
            + text[end:right].replace("\n", " ")
            + suffix
        )

    @classmethod
    def validate_prompt(cls, text: str) -> None:
        """
        Raise ValueError on emphasis syntax that cannot be honoured.

        Catches a modifier carrying a weight but no text (which makes ComfyUI
        tokenize the ':1.2' literally), empty parentheses, and unbalanced
        parentheses (which silently degrade the surrounding text to weight 1.0).

        Args:
            text: Parsed template text to validate

        Raises:
            ValueError: If the text contains malformed emphasis syntax

        Example:
            >>> TemplateParser.validate_prompt("a cat (:1.1)")
            ValueError: Malformed prompt emphasis: ...
        """
        problems: list[str] = []

        depth: int = cls.paren_depth(text)
        if depth != 0:
            reason: str = (
                "a ) appears with no matching ( before it"
                if depth < 0
                else f"{depth} ( left unclosed"
            )
            problems.append(
                f"unbalanced parentheses, {reason}. Write \\( and \\) for "
                f"literal characters. In: "
                f"{cls.excerpt(text, 0, len(text))[:120]}"
            )

        for pattern, why in (
            (cls.EMPTY_MODIFIER, "has a weight but no text inside the parentheses"),
            (cls.EMPTY_PARENS, "is empty and carries no text"),
        ):
            for match in pattern.finditer(text):
                line, column = cls.locate(text, match.start())
                problems.append(
                    f"line {line}, column {column}: {match.group()!r} {why} "
                    f"-- in: {cls.excerpt(text, match.start(), match.end())}"
                )

        if problems:
            raise ValueError("Malformed prompt emphasis: " + "; ".join(problems))

    @staticmethod
    def strip_malformed_prompt(text: str) -> tuple[str, list[str]]:
        """
        Remove malformed emphasis syntax, returning the cleaned text and what was removed.

        The counterpart to validate_prompt: used when parsing errors are tolerated,
        so a broken prompt still generates instead of blocking the graph.

        Empty modifiers and empty parentheses are deleted outright. Unbalanced
        parentheses have their unmatched character removed, since the alternative
        is a weight applied to text the author did not intend.

        Args:
            text: Parsed template text to repair

        Returns:
            Tuple of (cleaned text, list of human-readable removals)

        Example:
            >>> TemplateParser.strip_malformed_prompt("a cat, (:1.1)")
            ('a cat', ["line 1, column 8: '(:1.1)' has a weight but no text"])
        """
        removed: list[str] = []
        cleaned: str = text

        for pattern, why in (
            (TemplateParser.EMPTY_MODIFIER, "has a weight but no text"),
            (TemplateParser.EMPTY_PARENS, "is empty and carries no text"),
        ):
            def _record(match: "re.Match[str]", reason: str = why) -> str:
                line, column = TemplateParser.locate(match.string, match.start())
                removed.append(
                    f"line {line}, column {column}: {match.group()!r} {reason} "
                    f"-- in: {TemplateParser.excerpt(match.string, match.start(), match.end())}"
                )
                return ""

            cleaned = pattern.sub(_record, cleaned)

        cleaned, unbalanced = TemplateParser.strip_unbalanced(cleaned)
        if unbalanced:
            removed.extend(unbalanced)

        return cleaned, removed

    @staticmethod
    def strip_unbalanced(text: str) -> tuple[str, list[str]]:
        """
        Remove parentheses that have no partner, returning the text and descriptions.

        Walks the string honouring backslash escapes, so \\( and \\) are literal
        characters and are never treated as delimiters.

        Args:
            text: Text to repair

        Returns:
            Tuple of (text without unmatched parentheses, list of descriptions)

        Example:
            >>> TemplateParser.strip_unbalanced("a cat (:1.2")
            ('a cat :1.2', ["line 1, column 7: unmatched '(' removed"])
        """
        keep: list[str] = []
        open_at: list[int] = []
        removals: dict[int, str] = {}
        index: int = 0
        offset: int = 0

        while index < len(text):
            char: str = text[index]
            if char == "\\" and index + 1 < len(text):
                keep.append(text[index : index + 2])
                index += 2
                offset += 2
                continue
            if char == "(":
                open_at.append(offset)
            elif char == ")":
                if open_at:
                    open_at.pop()
                else:
                    removals[offset] = "unmatched ')'"
            keep.append(char)
            index += 1
            offset += 1

        if open_at:
            for position in open_at:
                removals[position] = "unmatched '('"

        if not removals:
            return text, []

        cleaned: list[str] = []
        offset = 0
        descriptions: list[str] = []
        for char in text:
            if offset in removals:
                line, column = TemplateParser.locate(text, offset)
                descriptions.append(
                    f"line {line}, column {column}: {removals[offset]} removed "
                    f"-- in: {TemplateParser.excerpt(text, offset, offset + 1)}"
                )
            else:
                cleaned.append(char)
            offset += 1

        return "".join(cleaned), descriptions

    @staticmethod
    def parse_template(
        template: str,
        prompt_map: PositivePrompt | NegativePrompt,
        allow_parse_errors: bool = False,
    ) -> str:
        """
        Parse a template string with variable substitution from prompt objects.

        Replaces <variable> placeholders with values from prompt attributes.

        Args:
            template: Template string with <variable> placeholders
            prompt_map: PositivePrompt or NegativePrompt instance with substitution values
            allow_parse_errors: If True, replace missing values with empty string. If False, use placeholder.

        Returns:
            Parsed template string with all variables substituted

        Example:
            >>> pos = PositivePrompt()
            >>> pos.age = "young"
            >>> TemplateParser.parse_template("A <age> person", pos)
            'A young person'
        """

        result: str = template
        placeholders = TemplateParser.find_placeholders(template)

        missing: list[str] = TemplateParser.find_missing_placeholders(
            template, prompt_map
        )
        if missing:
            listed: str = ", ".join(f"<{name}>" for name in missing)
            if allow_parse_errors:
                logging.warning(
                    "Template references %s with no value set; substituting empty "
                    "text. Set the field or disable allow_parse_errors to treat "
                    "this as an error.",
                    listed,
                )
            else:
                raise ValueError(
                    f"Template references {listed} but no usable value is set. "
                    f"Set the field, give it text, or enable allow_parse_errors."
                )

        for placeholder in placeholders:
            # Try to get from positive prompt first
            value: str | None = None

            if hasattr(prompt_map, placeholder):
                value = getattr(prompt_map, placeholder)

            if value is None:
                # Use default value for missing variables
                value = ""

            # Replace placeholder with value (using <> syntax)
            result = result.replace(f"<{placeholder}>", str(value))

        if allow_parse_errors:
            result, repairs = TemplateParser.strip_malformed_prompt(result)
            if repairs:
                logging.warning(
                    "Malformed prompt emphasis was removed: %s",
                    "; ".join(repairs),
                )
        else:
            TemplateParser.validate_prompt(result)

        result = " ".join(result.split())
        result = ",".join(result.split(", ,"))
        result = ",".join(result.split(","))
        result = re.sub(r",+", ",", result)  # ",," -> ","
        result = re.sub(r"\s*,\s*", ", ", result)  # "word ,word" -> "word, word"
        result = result.strip().strip(",")

        return result

    # @staticmethod
    # def get_required_variables(template: str) -> List[str]:
    #     """
    #     Get list of required variables from a template.

    #     Args:
    #         template: Template string with placeholders

    #     Returns:
    #         List of required variable names (no duplicates)

    #     Raises:
    #         ValueError: If template is not a string
    #     """

    #     placeholders = TemplateParser.find_placeholders(template)
    #     # Remove duplicates while preserving order
    #     seen: Set[str] = set()
    #     # unique = []
    #     for p in placeholders:
    #         if p not in seen:
    #             seen.add(p)
    #             # unique.append(p)
    #     return list(seen)
