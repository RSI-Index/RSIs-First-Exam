import pytest

from proposal_review_marker import (
    ReviewMarker,
    canonical_proposal_hash,
    parse_review_marker,
    render_review_marker,
)


def test_hash_is_stable_and_body_sensitive():
    assert canonical_proposal_hash("T", "B") == canonical_proposal_hash("T", "B")
    assert canonical_proposal_hash("T", "B") != canonical_proposal_hash("T", "B ")


def test_accept_marker_round_trips_unicode():
    marker = render_review_marker(
        decision="Accept",
        proposal_sha256="a" * 64,
        discussion_node_id="D_kw中文",
    )

    assert parse_review_marker(marker) == ReviewMarker(
        schema=1,
        decision="Accept",
        proposal_sha256="a" * 64,
        discussion_node_id="D_kw中文",
    )


@pytest.mark.parametrize(
    "decision", ["Reject", "Strong Reject", "require human review"]
)
def test_non_accept_decisions_are_not_eligible(decision):
    assert ReviewMarker(1, decision, "a" * 64, "D_1").eligible is False


def test_strong_accept_decision_is_eligible():
    assert ReviewMarker(1, "Strong Accept", "a" * 64, "D_1").eligible is True


@pytest.mark.parametrize(
    "body",
    [
        "<!-- rsi-proposal-review:{not-json} -->",
        "<!-- rsi-proposal-review:{\"decision\":\"Accept\"} -->",
        "<!-- rsi-proposal-review:{\"decision\":\"Maybe\",\"discussion_node_id\":\"D_1\",\"proposal_sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"schema\":1} -->",
        "<!-- rsi-proposal-review:{\"decision\":\"Accept\",\"discussion_node_id\":\"D_1\",\"proposal_sha256\":\"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\",\"schema\":1} -->",
        "<!-- rsi-proposal-review:{\"decision\":\"Accept\",\"discussion_node_id\":\"D_1\",\"proposal_sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"schema\":2} -->",
    ],
)
def test_parse_rejects_malformed_or_noncanonical_marker(body):
    assert parse_review_marker(body) is None


def test_parse_rejects_duplicate_markers():
    marker = render_review_marker("Accept", "a" * 64, "D_1")
    assert parse_review_marker(f"{marker}\n{marker}") is None


def test_parse_rejects_expected_discussion_node_id_mismatch():
    marker = render_review_marker("Accept", "a" * 64, "D_1")
    assert parse_review_marker(marker, expected_discussion_node_id="D_2") is None


def test_parse_allows_expected_discussion_node_id_match():
    marker = render_review_marker("Accept", "a" * 64, "D_1")
    assert parse_review_marker(marker, expected_discussion_node_id="D_1") == ReviewMarker(
        1, "Accept", "a" * 64, "D_1"
    )
