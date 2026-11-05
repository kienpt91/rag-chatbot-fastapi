"""robots.txt parsing and matching: group choice, wildcard semantics, the caps and the status rules."""

from __future__ import annotations

import pytest

from rag_chatbot.domain.robots import (
    ALLOW_ALL,
    DISALLOW_ALL,
    MAX_RULES,
    RobotsRules,
    parse_robots,
    rules_for_status,
)

AGENT = "RagChatbot/1.0 (+https://example.com/bot)"


def _blocked(pattern: str, path: str) -> bool:
    return not RobotsRules(allow=(), disallow=(pattern,), crawl_delay_s=None).allows(path)


def test_the_most_specific_group_wins_and_the_star_group_is_ignored_then() -> None:
    text = "User-agent: *\nDisallow: /private\n\nUser-agent: RagChatbot\nDisallow: /secret\nAllow: /secret/public\n"
    rules = parse_robots(text, AGENT)
    assert rules.disallow == ("/secret",)
    assert rules.allow == ("/secret/public",)
    assert rules.allows("/private")
    assert not rules.allows("/secret/a")
    assert rules.allows("/secret/public/x")


def test_the_star_group_applies_when_no_group_names_us() -> None:
    rules = parse_robots("User-agent: googlebot\nDisallow: /g\n\nUser-agent: *\nDisallow: /s\n", AGENT)
    assert rules.disallow == ("/s",)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "User-agent: googlebot\nDisallow: /\n",
        "User-agent: ragchatbotx\nDisallow: /\n",  # a longer token is a different agent
        "Disallow: /\n",  # rules before any user-agent line belong to nobody
        "just some words\n",
    ],
)
def test_a_file_with_no_group_for_us_allows_everything(text: str) -> None:
    assert parse_robots(text, AGENT) == ALLOW_ALL


def test_a_named_group_without_rules_allows_everything_even_if_the_star_group_forbids_all() -> None:
    rules = parse_robots("User-agent: *\nDisallow: /\n\nUser-agent: ragchatbot\n", AGENT)
    assert rules.allows("/anything")


def test_agent_names_match_case_insensitively_by_prefix_and_the_longest_wins() -> None:
    text = "User-agent: RAG\nDisallow: /short\n\nUser-agent: ragchat\nDisallow: /long\n"
    assert parse_robots(text, AGENT).disallow == ("/long",)
    assert parse_robots("User-agent: RAG\nDisallow: /short\n", AGENT).disallow == ("/short",)


def test_consecutive_user_agent_lines_share_one_group() -> None:
    rules = parse_robots("User-agent: other\nUser-agent: ragchatbot\nDisallow: /x\n", AGENT)
    assert rules.disallow == ("/x",)


def test_a_user_agent_line_after_rules_starts_a_new_group() -> None:
    rules = parse_robots("User-agent: ragchatbot\nDisallow: /x\nUser-agent: other\nDisallow: /y\n", AGENT)
    assert rules.disallow == ("/x",)


def test_groups_of_the_same_agent_are_merged() -> None:
    text = (
        "User-agent: *\nDisallow: /a\n\nUser-agent: other\nDisallow: /o\n\nUser-agent: *\nDisallow: /b\nAllow: /b/ok\n"
    )
    rules = parse_robots(text, AGENT)
    assert rules.disallow == ("/a", "/b")
    assert rules.allow == ("/b/ok",)


def test_rules_before_the_first_user_agent_line_are_ignored() -> None:
    assert parse_robots("Disallow: /x\nUser-agent: *\nDisallow: /y\n", AGENT).disallow == ("/y",)


def test_comments_unknown_fields_and_malformed_lines_are_ignored() -> None:
    text = (
        "User-agent: *  # everyone\n"
        "Sitemap: https://example.com/sitemap.xml\n"
        "Host: example.com\n"
        "Disallow /no-colon\n"
        "nonsense\n"
        ": /no-field\n"
        "Disallow: /ok # trailing comment\n"
        "#Disallow: /commented-out\n"
        "Disallow: private\n"  # a rule must start at the path root or with a wildcard
        "Disallow: *.pdf\n"
    )
    assert parse_robots(text, AGENT).disallow == ("/ok", "*.pdf")


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_every_line_ending_and_a_byte_order_mark_are_understood(newline: str) -> None:
    text = f"\ufeffUSER-AGENT: *{newline}DISALLOW: /a{newline}ALLOW: /a/b{newline}Crawl-Delay: 2{newline}"
    rules = parse_robots(text, AGENT)
    assert (rules.disallow, rules.allow, rules.crawl_delay_s) == (("/a",), ("/a/b",), 2.0)


def test_an_empty_disallow_allows_everything() -> None:
    rules = parse_robots("User-agent: *\nDisallow:\n", AGENT)
    assert rules.disallow == ()
    assert rules.allows("/anything")
    assert RobotsRules(allow=("",), disallow=("",), crawl_delay_s=None).allows("/x")


@pytest.mark.parametrize(
    ("pattern", "path", "blocked"),
    [
        ("/fish", "/fish", True),
        ("/fish", "/fish.html", True),
        ("/fish", "/fish/salmon.html", True),
        ("/fish", "/Fish.asp", False),
        ("/fish", "/catfish", False),
        ("/fish*", "/fish", True),
        ("/*.php", "/index.php", True),
        ("/*.php", "/folder/filename.php?parameters", True),
        ("/*.php$", "/filename.php", True),
        ("/*.php$", "/filename.php?parameters", False),
        ("/fish*.php", "/fish.php", True),
        ("/fish*.php", "/Fish.PHP", False),
        ("/*/private", "/a/b/private/x", True),
        ("/*/private", "/private", False),
        ("/a$", "/a", True),
        ("/a$", "/ab", False),
        ("/a$", "/a/", False),
        ("/*?id=", "/p?id=1", True),
        ("/*?", "/a?b", True),
        ("/*?", "/a", False),
        ("/a*b*c$", "/aXXbYYc", True),
        ("/a*b*c$", "/aXXbYYcZ", False),
        ("/*$", "/anything", True),
        ("/a*$", "/abc", True),
        ("/a*a$", "/a", False),
        ("/a*a$", "/aa", True),
        ("/a*ab", "/ab", False),
        ("/a**b", "/aXb", True),
        ("/a$b", "/a$b", True),  # a dollar sign that is not the last character is a literal
        ("/", "/", True),
        ("/", "/x/y", True),
    ],
)
def test_wildcards_and_the_end_anchor_follow_rfc_9309(pattern: str, path: str, blocked: bool) -> None:
    assert _blocked(pattern, path) is blocked


def test_the_longest_matching_rule_wins() -> None:
    rules = RobotsRules(allow=("/folder/page",), disallow=("/folder/",), crawl_delay_s=None)
    assert rules.allows("/folder/page.html")
    assert not rules.allows("/folder/other")
    longer_disallow = RobotsRules(allow=("/p",), disallow=("/page",), crawl_delay_s=None)
    assert not longer_disallow.allows("/page")
    assert longer_disallow.allows("/p2")


def test_allow_wins_a_tie_in_either_order() -> None:
    assert RobotsRules(allow=("/p",), disallow=("/p",), crawl_delay_s=None).allows("/p")
    assert parse_robots("User-agent: *\nDisallow: /p\nAllow: /p\n", AGENT).allows("/p")
    assert parse_robots("User-agent: *\nAllow: /p\nDisallow: /p\n", AGENT).allows("/p")


def test_a_path_no_rule_matches_is_allowed() -> None:
    assert parse_robots("User-agent: *\nDisallow: /private\n", AGENT).allows("/public")


@pytest.mark.parametrize(
    ("rule", "path", "blocked"),
    [
        ("/a/b", "/a%2Fb", False),  # an encoded slash is not a separator
        ("/a/b", "/a%2fb", False),
        ("/a%2Fb", "/a%2fb", True),  # hex digits compare case-insensitively
        ("/a%2Fb", "/a/b", False),
        ("/a/b", "/%61/b", True),  # an escaped unreserved character is the character
        ("/%61/b", "/a/b", True),
        ("/a/b", "/a/%62", True),
        ("/%7Euser", "/~user", True),
        ("/\u00e9", "/%C3%A9", True),
        ("/%c3%a9", "/\u00e9", True),
        ("/%25", "/%", True),
        ("/a b", "/a%20b", True),
    ],
)
def test_matching_is_done_on_percent_normalised_paths(rule: str, path: str, blocked: bool) -> None:
    assert _blocked(rule, path) is blocked


def test_a_fragment_is_not_part_of_the_path_and_a_missing_leading_slash_is_supplied() -> None:
    rules = RobotsRules(allow=(), disallow=("/a",), crawl_delay_s=None)
    assert not rules.allows("/a#frag")
    assert not rules.allows("a")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10", 10.0),
        ("1.5", 1.5),
        (".5", 0.5),
        ("0", 0.0),
        ("9999", 60.0),
        ("60", 60.0),
        ("abc", None),
        ("-1", None),
        ("+1", None),
        ("nan", None),
        ("inf", None),
        ("1e3", None),
        ("1_0", None),
        ("1.2.3", None),
        (".", None),
        ("", None),
        ("\u0663", None),  # a non-ASCII digit
    ],
)
def test_crawl_delay_is_a_plain_decimal_capped_at_sixty_seconds(value: str, expected: float | None) -> None:
    assert parse_robots(f"User-agent: *\nCrawl-delay: {value}\n", AGENT).crawl_delay_s == expected


def test_the_larger_crawl_delay_of_merged_groups_is_kept() -> None:
    text = "User-agent: *\nCrawl-delay: 3\n\nUser-agent: x\n\nUser-agent: *\nCrawl-delay: 7\n"
    assert parse_robots(text, AGENT).crawl_delay_s == 7.0


def test_at_most_one_thousand_rules_are_kept() -> None:
    text = "User-agent: *\n" + "".join(f"Disallow: /p{i}\nAllow: /q{i}\n" for i in range(1_500))
    rules = parse_robots(text, AGENT)
    assert len(rules.allow) + len(rules.disallow) == MAX_RULES


def test_a_rule_longer_than_any_url_is_ignored() -> None:
    rules = parse_robots("User-agent: *\nDisallow: /" + "a" * 5_000 + "\nDisallow: /ok\n", AGENT)
    assert rules.disallow == ("/ok",)


def test_everything_after_the_first_half_megabyte_is_ignored() -> None:
    filler = "# filler line\n" * 40_000  # 560,000 bytes
    rules = parse_robots("User-agent: *\nDisallow: /early\n" + filler + "Disallow: /late\n", AGENT)
    assert rules.disallow == ("/early",)


def test_a_rule_cut_in_half_by_the_size_cap_is_dropped_not_half_applied() -> None:
    cap = 500 * 1024
    head = "User-agent: *\nDisallow: /early\n"
    padding = "#" * (cap - 13 - len(head) - 1) + "\n"
    text = head + padding + "Disallow: /abcdef\n"  # the cap falls after "Disallow: /ab"
    assert parse_robots(text, AGENT).disallow == ("/early",)


def test_a_huge_multibyte_file_is_cut_by_bytes() -> None:
    text = "User-agent: *\nDisallow: /early\n" + "# \u20ac\u20ac\u20ac\n" * 100_000 + "Disallow: /late\n"
    assert parse_robots(text, AGENT).disallow == ("/early",)


def test_a_pattern_built_to_make_a_backtracking_matcher_explode_is_answered_at_once() -> None:
    pattern = "/" + "*a" * 500 + "*b"
    target = "/" + "a" * 2_000
    assert not _blocked(pattern, target)
    assert _blocked(pattern, target + "b")
    assert not _blocked("/" + "*a" * 500 + "b$", target + "bc")


@pytest.mark.parametrize("status", [404, 401, 403, 410, 451, 499])
def test_a_client_error_means_there_is_no_robots_file(status: int) -> None:
    assert rules_for_status(status, "User-agent: *\nDisallow: /\n", AGENT) == ALLOW_ALL


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 301, 302, 304, 100, 0, 199, 600])
def test_an_unreachable_site_is_treated_as_fully_disallowed(status: int) -> None:
    rules = rules_for_status(status, "", AGENT)
    assert rules == DISALLOW_ALL
    assert not rules.allows("/")
    assert not rules.allows("/any/path")


@pytest.mark.parametrize("status", [200, 204, 299])
def test_a_success_status_is_parsed(status: int) -> None:
    assert rules_for_status(status, "User-agent: *\nDisallow: /x\n", AGENT).disallow == ("/x",)


def test_the_constants_behave_as_named() -> None:
    assert ALLOW_ALL.allows("/") and ALLOW_ALL.allows("/x?y=z")
    assert not DISALLOW_ALL.allows("/") and not DISALLOW_ALL.allows("/x")
