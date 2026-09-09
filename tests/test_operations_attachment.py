import importlib
import os
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="module")
def api_module():
    """Charge l'API avec un petit substitut du core absent du dépôt API."""
    package = types.ModuleType("rfp_parser")
    package.__path__ = []
    exports = types.ModuleType("rfp_parser.exports")
    prompting = types.ModuleType("rfp_parser.prompting")
    exports.export_outputs = lambda *args, **kwargs: {}

    def build_chat_payload(text, model, attachments=None):
        content = text
        for index, attachment in enumerate(attachments or [], start=1):
            content += f"\n=== ATTACHMENT #{index} ===\n{attachment}"
        return {"model": model, "messages": [{"role": "user", "content": content}]}

    prompting.build_chat_payload = build_chat_payload
    sys.modules.update(
        {
            "rfp_parser": package,
            "rfp_parser.exports": exports,
            "rfp_parser.prompting": prompting,
        }
    )
    sys.modules.pop("rfp_api_app", None)
    return importlib.import_module("rfp_api_app")


@pytest.fixture
def workbook(tmp_path, monkeypatch, api_module):
    path = tmp_path / "Operations.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(
            {"operation": ["Fill", "Stir"], "flow_l_h": [250, 500]}
        ).to_excel(writer, sheet_name="Standards", index=False)
    monkeypatch.setenv("RFP_OPERATIONS_PATH", str(path))
    monkeypatch.setenv("RFP_OPERATIONS_ATTACHMENT_ENABLE", "true")
    api_module._OPERATIONS_CACHE.clear()
    return path


def _config(api_module):
    return api_module.LLMRuntimeConfig(
        provider="deepinfra",
        model="test-model",
        api_key="test-key",
        base_or_url="https://example.invalid/v1",
        chat_url="https://example.invalid/v1/chat/completions",
        max_tokens=100,
        temperature=0.1,
        source="test",
    )


def test_build_payload_injects_operations_workbook(api_module, workbook):
    payload = api_module.build_payload("test", _config(api_module))

    content = payload["messages"][-1]["content"]
    assert "=== ATTACHMENT #1 ===" in content
    assert "=== OPERATIONS STANDARDS ===" in content
    assert "flow_l_h" in content
    assert payload["response_format"] == {"type": "json_object"}


def test_missing_workbook_does_not_block_payload(
    api_module, tmp_path, monkeypatch
):
    monkeypatch.setenv("RFP_OPERATIONS_PATH", str(tmp_path / "missing.xlsx"))
    api_module._OPERATIONS_CACHE.clear()

    payload = api_module.build_payload("test", _config(api_module))

    assert payload["messages"][-1]["content"] == "test"


def test_unchanged_mtime_uses_cache(api_module, workbook, monkeypatch):
    calls = 0
    original = api_module._build_operations_attachment

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(api_module, "_build_operations_attachment", counted)
    first = api_module._get_operations_attachment()
    second = api_module._get_operations_attachment()

    assert first is second
    assert calls == 1


def test_changed_mtime_rebuilds_cache(api_module, workbook, monkeypatch):
    calls = 0
    original = api_module._build_operations_attachment

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(api_module, "_build_operations_attachment", counted)
    api_module._get_operations_attachment()
    stat = workbook.stat()
    os.utime(workbook, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    api_module._get_operations_attachment()

    assert calls == 2


def test_rows_and_chars_are_limited(api_module, tmp_path, monkeypatch):
    path = tmp_path / "large.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(
            {"operation": [f"Operation {i}" for i in range(100)], "value": range(100)}
        ).to_excel(writer, sheet_name="Large", index=False)

    monkeypatch.setenv("RFP_OPERATIONS_PATH", str(path))
    monkeypatch.setenv("RFP_OPERATIONS_MAX_ROWS", "5")
    monkeypatch.setenv("RFP_OPERATIONS_MAX_CHARS", "180")
    api_module._OPERATIONS_CACHE.clear()

    attachment = api_module._get_operations_attachment()

    assert attachment is not None
    assert len(attachment) <= 180
    assert "Operation 4" in attachment
    assert "Operation 5" not in attachment
    assert "=== END OPERATIONS STANDARDS ===" in attachment
