"""Content a browser hides but a naive reader shows (and the reverse): the extractor must not index invisible text."""

from __future__ import annotations

import pytest

from rag_chatbot.domain.extraction import extract_html, extract_markdown


def html_text(body: str) -> str:
    return extract_html(f"<html><body><main><p>Open daily.</p>{body}</main></body></html>").text


def test_a_script_that_hides_an_inner_script_end_tag_stays_hidden_until_the_real_end() -> None:
    # script data double-escaped state: the first </script> only closes the inner script
    text = html_text("<script><!--<script>x</script> PAYLOAD </script><p>After the script.</p>")
    assert "PAYLOAD" not in text and "After the script." in text and "Open daily." in text


@pytest.mark.parametrize(
    "script",
    [
        "<script>a()</script> SHOWN",
        "<script><!-- a() --></script> SHOWN",
        "<script><!--<script>x</script>--></script> SHOWN",
    ],
)
def test_ordinary_scripts_still_end_where_a_browser_ends_them(script: str) -> None:
    assert "SHOWN" in html_text(script) and "a()" not in html_text(script)


def test_an_unterminated_double_escaped_script_hides_the_rest_like_a_browser() -> None:
    text = html_text("<script><!--<script>x</script> PAYLOAD <p>Rest</p>")
    assert "PAYLOAD" not in text and "Rest" not in text


def test_alt_text_of_images_is_never_indexed() -> None:
    text = html_text('<img width="1" height="1" alt="HIDDEN INSTRUCTION"><p>Visible.</p>')
    assert "HIDDEN" not in text and "Visible." in text


@pytest.mark.parametrize(
    "style",
    [
        "position:fixed;left:-9999px",
        "position:fixed;top:-5000px",
        "position:relative;left:-9999px",
        "transform:scale(0)",
        "clip-path:inset(100%)",
        "width:0;overflow:hidden",
        "font-size:1px",
        "font-size:0.5px",
        "margin-left:-9999px",
        "color:#fff;background-color:#fff",
        "color:white;background:white",
    ],
)
def test_more_ways_of_hiding_text_are_recognised(style: str) -> None:
    assert "SECRET" not in html_text(f'<div style="{style}">SECRET</div>')


@pytest.mark.parametrize(
    "style",
    [
        "font-size:14px",
        "color:#000;background-color:#fff",
        "position:fixed;left:10px",
        "transform:rotate(3deg)",
        "width:100px;overflow:hidden",
        "margin-left:-5px",
    ],
)
def test_ordinary_styles_do_not_hide_text(style: str) -> None:
    assert "SHOWN" in html_text(f'<div style="{style}">SHOWN</div>')


@pytest.mark.parametrize(
    "fragment",
    [
        "<span hidden>SECRET</span>",
        '<div style="display:none">SECRET</div>',
        "<template>SECRET</template>",
        '<p aria-hidden="true">SECRET</p>',
        '<div style="font-size:0">SECRET</div>',
        "<section><div hidden><span>SECRET</span></div></section>",
    ],
)
def test_inline_html_in_markdown_gets_the_same_hidden_text_rules(fragment: str) -> None:
    text = extract_markdown(f"# Title\n\nVisible start {fragment} visible end.\n\nNext paragraph.").text
    assert "SECRET" not in text and "Visible start" in text and "Next paragraph." in text


def test_markdown_keeps_visible_inline_html_and_void_elements_do_not_swallow_the_page() -> None:
    text = extract_markdown('Before <span class="x">kept</span> and <img src=x hidden> after.\n\nLast.').text
    assert "kept" in text and "after." in text and "Last." in text


def test_an_unclosed_hidden_element_in_markdown_hides_the_rest_but_work_stays_bounded() -> None:
    assert "TAIL" not in extract_markdown("Start <div hidden> TAIL").text
    extract_markdown("<div hidden>" * 20_000 + "x")
    extract_markdown("<span " * 20_000)
