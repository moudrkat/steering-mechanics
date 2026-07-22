"""Unit tests for steermech pure scoring logic (no brainscope needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from steermech.eval import (score_efficacy, score_damage, combined_objective,
                            load_benign, load_intent)


def test_score_efficacy_perfect_suppression():
    diffs = [{"suppressed_positional": [{"word": "reminder"}, {"word": "task"}]},
             {"suppressed_positional": [{"word": "checklist"}]}]
    r = score_efficacy(diffs, avoid=["reminder", "checklist", "task"], target=[])
    assert r["miss"] == 0.0 and r["avoid_suppressed"] == 2


def test_score_efficacy_no_suppression_is_full_miss():
    diffs = [{"suppressed_positional": [{"word": "banana"}]}]
    r = score_efficacy(diffs, avoid=["reminder"], target=[])
    assert r["miss"] == 1.0


def test_score_efficacy_substring_match():
    diffs = [{"suppressed_positional": [{"word": "reminders"}]}]
    r = score_efficacy(diffs, avoid=["reminder"], target=[])
    assert r["avoid_suppressed"] == 1


def test_score_efficacy_empty_intent():
    assert score_efficacy([{}], avoid=[], target=[])["miss"] == 1.0


def test_score_damage():
    assert score_damage([0.1, 0.3, None, 0.2])["kl"] == 0.2
    assert score_damage([])["kl"] == 0.0


def test_combined_objective():
    assert combined_objective(0.3, 2.0, 0.1) == 0.5


def test_benign_and_intent_load():
    assert len(load_benign()) >= 10
    assert len(load_benign(5)) == 5
    intent = load_intent("discuss-no-tasks")
    assert "reminder" in intent["avoid"] and intent["prompts"]


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-q"])


def test_message_text_gathers_content_and_toolcalls():
    from steermech.eval import _message_text
    msg = {"content": "hello", "tool_calls": [
        {"function": {"name": "SuggestMessages", "arguments": '{"Message": "make a task"}'}}]}
    t = _message_text(msg)
    assert "hello" in t and "SuggestMessages" in t and "make a task" in t


def test_message_text_dict_arguments():
    from steermech.eval import _message_text
    msg = {"content": None, "tool_calls": [
        {"function": {"name": "X", "arguments": {"a": "reminder"}}}]}
    assert "reminder" in _message_text(msg)
