import base64
import json
from pathlib import Path

import httpx
import pytest

from src.config import Settings
from src.errors import AppError, ErrorCode
from src.providers.mock_provider import MockVLMProvider
from src.providers.ollama_provider import OllamaProvider
from src.schemas import (
    ApocalypseScenario,
    AppraisalPayload,
    ImageIdentityVerdict,
    ScanRequest,
)


def request() -> ScanRequest:
    return ScanRequest(
        scenario=ApocalypseScenario.CITY_BLACKOUT,
        apocalypse_years=3,
        description="测试充电宝",
    )


def valid_content() -> str:
    appraisal = MockVLMProvider().analyze(None, request())
    payload = {
        field: getattr(appraisal, field)
        for field in AppraisalPayload.model_fields
    }
    return AppraisalPayload(**payload).model_dump_json()


def identity_content() -> str:
    return ImageIdentityVerdict(
        same_physical_object=True,
        category_preserved=True,
        silhouette_and_proportions_preserved=True,
        camera_and_composition_preserved=True,
        functional_features_preserved=True,
        only_surface_condition_changed=True,
        post_apocalyptic_damage_clearly_visible=True,
        before_functional_features=["front port", "side button"],
        after_functional_features=["front port", "side button"],
        added_features=[],
        missing_features=[],
        moved_or_duplicated_features=[],
        issues=[],
        confidence=0.96,
    ).model_dump_json()


def settings() -> Settings:
    return Settings(
        ollama_base_url="http://ollama.test",
        ollama_model="qwen3-vl:8b-instruct-q8_0",
        ollama_keep_alive="0",
        ollama_num_ctx=8192,
    )


def test_analyze_uses_exact_model_vision_and_grammar_safe_schema(
    tmp_path: Path,
) -> None:
    captured: dict = {}

    def handler(http_request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(http_request.content))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": valid_content()}},
        )

    client = httpx.Client(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(handler),
    )
    provider = OllamaProvider(settings(), client=client)
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"example-image-bytes")

    result = provider.analyze(image_path, request())

    assert result.original_item == "测试充电宝"
    assert captured["model"] == "qwen3-vl:8b-instruct-q8_0"
    assert captured["stream"] is False
    assert captured["think"] is False
    assert captured["keep_alive"] == "0"
    assert captured["options"] == {"temperature": 0, "num_ctx": 8192}
    assert captured["messages"][1]["images"]
    assert captured["format"]["additionalProperties"] is False
    assert set(captured["format"]["required"]) == set(AppraisalPayload.model_fields)
    serialized_format = json.dumps(captured["format"])
    assert "minLength" not in serialized_format
    assert "maxLength" not in serialized_format
    assert captured["format"]["properties"]["observed_evidence"]["maxItems"] == 12
    assert '"maxLength":4000' in captured["messages"][1]["content"]


def test_invalid_schema_is_retried_once() -> None:
    attempts = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        content = "{}" if attempts == 1 else valid_content()
        return httpx.Response(200, json={"message": {"content": content}})

    client = httpx.Client(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(handler),
    )
    result = OllamaProvider(settings(), client=client).analyze(None, request())

    assert attempts == 2
    assert result.apocalypse_name


def test_text_only_analyze_uses_description_evidence_prompt() -> None:
    captured: dict = {}

    def handler(http_request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(http_request.content))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": valid_content()}},
        )

    client = httpx.Client(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(handler),
    )
    result = OllamaProvider(settings(), client=client).analyze(None, request())

    assert result.original_item == "测试充电宝"
    assert "images" not in captured["messages"][1]
    assert "当前输入只有用户的物品文字描述" in captured["messages"][0]["content"]
    assert "照片" not in captured["messages"][0]["content"]
    assert "照片证据" not in captured["messages"][1]["content"]
    assert "observed_evidence 必须注明来源是用户描述" in captured["messages"][1][
        "content"
    ]


def test_scenario_label_is_rejected_as_an_apocalypse_translation() -> None:
    requests: list[dict] = []
    invalid = json.loads(valid_content())
    invalid["apocalypse_name"] = request().scenario.label

    def handler(http_request: httpx.Request) -> httpx.Response:
        body = json.loads(http_request.content)
        requests.append(body)
        content = json.dumps(invalid) if len(requests) == 1 else valid_content()
        return httpx.Response(200, json={"message": {"content": content}})

    client = httpx.Client(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(handler),
    )
    result = OllamaProvider(settings(), client=client).analyze(None, request())

    assert len(requests) == 2
    assert result.apocalypse_name == "便携式能源储备核心"
    assert "not the original name or scenario label" in requests[1]["messages"][1]["content"]


def test_verify_identity_sends_before_and_after_in_order(tmp_path: Path) -> None:
    captured: dict = {}

    def handler(http_request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(http_request.content))
        return httpx.Response(200, json={"message": {"content": identity_content()}})

    client = httpx.Client(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(handler),
    )
    source = tmp_path / "before.png"
    candidate = tmp_path / "after.png"
    source.write_bytes(b"before-image")
    candidate.write_bytes(b"after-image")
    appraisal = MockVLMProvider().analyze(None, request())

    verdict = OllamaProvider(settings(), client=client).verify_identity(
        source,
        candidate,
        appraisal,
        request(),
    )

    images = captured["messages"][1]["images"]
    assert [base64.b64decode(value) for value in images] == [
        b"before-image",
        b"after-image",
    ]
    assert captured["model"] == "qwen3-vl:8b-instruct-q8_0"
    assert captured["think"] is False
    assert captured["options"] == {"temperature": 0, "num_ctx": 8192}
    assert captured["format"]["additionalProperties"] is False
    assert "minLength" not in json.dumps(captured["format"])
    assert "post_apocalyptic_damage_clearly_visible" in captured["format"]["required"]
    identity_prompt = captured["messages"][1]["content"]
    assert "extreme, near-ruin, visually dominant" in identity_prompt
    assert "merely moderate weathering is false" in identity_prompt
    assert "Judge the target object rather than incidental background" in identity_prompt
    assert "deep cracking, peeling layers" in identity_prompt
    assert "geometry or parts actually changed" in identity_prompt
    assert verdict.is_acceptable(0.8)


def test_verify_identity_retries_invalid_schema_once(tmp_path: Path) -> None:
    attempts = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        content = "{}" if attempts == 1 else identity_content()
        return httpx.Response(200, json={"message": {"content": content}})

    client = httpx.Client(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(handler),
    )
    source = tmp_path / "before.png"
    candidate = tmp_path / "after.png"
    source.write_bytes(b"before")
    candidate.write_bytes(b"after")

    verdict = OllamaProvider(settings(), client=client).verify_identity(
        source,
        candidate,
        MockVLMProvider().analyze(None, request()),
        request(),
    )

    assert attempts == 2
    assert verdict.is_acceptable(0.8)


def test_connection_error_is_user_safe() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=http_request)

    client = httpx.Client(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(handler),
    )
    provider = OllamaProvider(settings(), client=client)

    with pytest.raises(AppError) as error:
        provider.analyze(None, request())

    assert error.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert "offline" not in error.value.user_message


def test_unload_uses_generate_keep_alive_zero() -> None:
    captured: dict = {}

    def handler(http_request: httpx.Request) -> httpx.Response:
        if http_request.method == "GET":
            assert http_request.url.path == "/api/ps"
            return httpx.Response(200, json={"models": []})
        captured.update(json.loads(http_request.content))
        captured["read_timeout"] = http_request.extensions["timeout"]["read"]
        return httpx.Response(200, json={"done": True})

    client = httpx.Client(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(handler),
    )
    OllamaProvider(settings(), client=client).unload()

    assert captured["model"] == "qwen3-vl:8b-instruct-q8_0"
    assert captured["keep_alive"] == 0
    assert captured["read_timeout"] == 15


def test_unload_waits_until_exact_model_is_absent() -> None:
    status_calls = 0
    now = [0.0]

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal status_calls
        if http_request.method == "POST":
            return httpx.Response(200, json={"done": True})
        status_calls += 1
        models = (
            [{"name": "qwen3-vl:8b-instruct-q8_0"}]
            if status_calls == 1
            else []
        )
        return httpx.Response(200, json={"models": models})

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    client = httpx.Client(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(handler),
    )
    provider = OllamaProvider(
        settings(), client=client, monotonic=clock, sleep=sleep
    )

    provider.unload()

    assert status_calls == 2
    assert now[0] == 0.2
