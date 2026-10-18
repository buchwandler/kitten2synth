from pathlib import Path

from kitten2synth.catalog import bundle_record, catalog_item, load_catalog


CATALOG = Path(__file__).resolve().parent / "fixtures" / "bundles.json"


def test_catalog_converts_to_onnxvoice_item():
    data = load_catalog(CATALOG)
    record = bundle_record("v2", data)
    item = catalog_item(record)
    assert item.system == "kitten2"
    assert item.id == "kitten-tts-v2"
    assert item.sample_rate == 24000
    assert item.default_voice == "Bruno"
    assert any(a.filename == "lm/lm.onnx" for a in item.artifacts)
    assert any(a.filename == "lm/lm.ternary-exact-int8.onnx.data" for a in item.artifacts)
