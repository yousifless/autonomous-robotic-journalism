from journalist_robot.core.json_utils import extract_balanced_json, parse_controller_output, parse_json_object


def test_extracts_first_object_ignoring_braces_in_strings():
    text = 'Sure! {"reasoning": "a {curly} wall", "linear_x": 0.2} trailing {"x": 1}'
    assert extract_balanced_json(text) == '{"reasoning": "a {curly} wall", "linear_x": 0.2}'


def test_returns_none_for_unbalanced_object():
    assert extract_balanced_json('{"linear_x": 0.2') is None


def test_parses_well_formed_answer():
    out = parse_controller_output(
        '{"linear_x": 0.3, "angular_z": -0.1, "confidence": 0.9, '
        '"newsworthiness_score": 0.8, "reasoning": "person at desk"}')
    assert out["linear_x"] == 0.3
    assert out["angular_z"] == -0.1
    assert out["confidence"] == 0.9
    assert out["newsworthiness_score"] == 0.8
    assert out["parsed_with_fallback"] is False


def test_fallback_on_truncated_answer_is_conservative():
    out = parse_controller_output('{"linear_x": 0.35, "angular_z": 0.2, "reasoning": "cut')
    assert out["parsed_with_fallback"] is True
    assert out["linear_x"] == 0.35
    assert out["confidence"] == 0.0          # missing confidence never yields a confident command


def test_garbage_gives_zero_command():
    out = parse_controller_output("I cannot see anything.")
    assert (out["linear_x"], out["angular_z"], out["confidence"]) == (0.0, 0.0, 0.0)


def test_parse_json_object():
    assert parse_json_object('x {"a": true} y') == {"a": True}
    assert parse_json_object("no json") is None
