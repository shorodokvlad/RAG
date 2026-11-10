from app.indexing.specs import SpecItem, parse_description, parse_specifications
from tests.factories import SPEC_JSON


def test_json_specs_drop_empty_items_and_sections():
    sections = parse_specifications(SPEC_JSON)

    assert [s.title for s in sections] == ["GENERAL CHARACTERISTICS", "DISPLAY"]
    assert sections[0].items == (SpecItem("Operating System", "iOS"), SpecItem("Connectivity", "5G"))


def test_legacy_text_with_headers_tabs_and_colons():
    text = "Specificații\nGeneral\nBrand: Apple\nModel\tiPhone 16\n\nDisplay\nScreen Size: 6.1 inch"

    sections = parse_specifications(text)

    assert [s.title for s in sections] == ["GENERAL", "DISPLAY"]
    assert sections[0].items == (SpecItem("Brand", "Apple"), SpecItem("Model", "iPhone 16"))
    assert sections[1].items == (SpecItem("Screen Size", "6.1 inch"),)


def test_pairs_before_any_header_go_to_general_section():
    sections = parse_specifications("Brand: Apple")
    assert sections[0].title == "GENERAL CHARACTERISTICS"


def test_plain_prose_has_no_specs():
    assert parse_specifications("A great phone with an amazing camera.") is None
    assert parse_specifications("") is None
    assert parse_specifications(None) is None


def test_invalid_json_falls_back_to_text_parsing():
    assert parse_specifications("[not json\nBrand: Apple")[0].items == (SpecItem("Brand", "Apple"),)


def test_parse_description_keeps_prose_for_text_but_not_for_json():
    structured = parse_description(SPEC_JSON)
    assert structured.is_structured and structured.prose is None

    mixed = parse_description("Amazing camera.\nBrand: Apple")
    assert not mixed.is_structured
    assert mixed.prose == "Amazing camera.\nBrand: Apple"
    assert mixed.sections[0].items == (SpecItem("Brand", "Apple"),)
