import json
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from PIL import Image

from src.config import Settings
from src.errors import AppError, ErrorCode
from src.providers.comfyui_provider import SCENARIO_SURFACE_EFFECTS, ComfyUIProvider
from src.providers.mock_provider import MockVLMProvider
from src.schemas import ApocalypseScenario, ScanRequest


def request() -> ScanRequest:
    return ScanRequest(
        scenario=ApocalypseScenario.CITY_BLACKOUT,
        apocalypse_years=3,
        description="测试充电宝",
    )


def appraisal():
    return MockVLMProvider().analyze(None, request())


def image_bytes(color: str) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 24), color).save(buffer, format="PNG")
    return buffer.getvalue()


def workflow() -> dict:
    return {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": "placeholder.png"},
            "_meta": {"title": "DSB_INPUT"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "placeholder", "clip": ["8", 0]},
            "_meta": {"title": "DSB_PROMPT"},
        },
        "3": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": 1},
            "_meta": {"title": "DSB_SEED"},
        },
        "4": {
            "class_type": "SaveImage",
            "inputs": {"images": ["9", 0], "filename_prefix": "ComfyUI"},
            "_meta": {"title": "DSB_OUTPUT"},
        },
        "8": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen.sft"}},
        "9": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}},
    }


def text_workflow() -> dict:
    return {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "placeholder", "clip": ["8", 0]},
            "_meta": {"title": "DSB_PROMPT"},
        },
        "2": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": 1},
            "_meta": {"title": "DSB_SEED"},
        },
        "3": {
            "class_type": "SaveImage",
            "inputs": {"images": ["9", 0], "filename_prefix": "ComfyUI"},
            "_meta": {"title": "DSB_OUTPUT"},
        },
        "8": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen.sft"}},
        "9": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}},
    }


def settings(
    tmp_path: Path,
    graph: dict | None = None,
    timeout: float = 10,
    text_graph: dict | None = None,
) -> Settings:
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(graph or workflow()), encoding="utf-8")
    text_path = tmp_path / "text-workflow.json"
    text_path.write_text(json.dumps(text_graph or text_workflow()), encoding="utf-8")
    return Settings(
        comfyui_base_url="http://comfy.test",
        comfyui_workflow_path=path,
        comfyui_text_workflow_path=text_path,
        comfyui_timeout_seconds=timeout,
        comfyui_poll_interval_seconds=1,
        output_dir=tmp_path / "outputs",
    )


def input_image(tmp_path: Path) -> Path:
    path = tmp_path / "input.png"
    path.write_bytes(image_bytes("black"))
    return path


def test_unload_accepts_comfyui_empty_success_response(tmp_path: Path) -> None:
    captured: dict = {}

    def handler(http_request: httpx.Request) -> httpx.Response:
        if http_request.url.path == "/free":
            captured.update(json.loads(http_request.content))
            return httpx.Response(200)
        assert http_request.url.path == "/system_stats"
        return httpx.Response(
            200,
            json={
                "devices": [
                    {
                        "type": "cuda",
                        "torch_vram_total": 32 * 1024 * 1024,
                    }
                ]
            },
        )

    client = httpx.Client(
        base_url="http://comfy.test",
        transport=httpx.MockTransport(handler),
    )

    ComfyUIProvider(
        settings(tmp_path), client=client, global_free_vram_mb=lambda: 22000
    ).unload()

    assert captured == {"unload_models": True, "free_memory": True}


def test_unload_waits_for_worker_memory_release(tmp_path: Path) -> None:
    status_calls = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal status_calls
        if http_request.url.path == "/free":
            return httpx.Response(200)
        assert http_request.url.path == "/system_stats"
        status_calls += 1
        allocation = 8 * 1024**3 if status_calls == 1 else 32 * 1024**2
        return httpx.Response(
            200,
            json={
                "devices": [
                    {
                        "type": "cuda",
                        "torch_vram_total": allocation,
                    }
                ]
            },
        )

    now = [0.0]

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    client = httpx.Client(
        base_url="http://comfy.test",
        transport=httpx.MockTransport(handler),
    )
    provider = ComfyUIProvider(
        settings(tmp_path),
        client=client,
        monotonic=clock,
        sleep=sleep,
        global_free_vram_mb=lambda: 22000,
    )

    provider.unload()

    assert status_calls == 2
    assert now[0] == 1


def test_unload_blocks_model_switch_while_global_vram_is_occupied(
    tmp_path: Path,
) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        if http_request.url.path == "/free":
            return httpx.Response(200)
        assert http_request.url.path == "/system_stats"
        return httpx.Response(
            200,
            json={
                "devices": [
                    {"type": "cuda", "torch_vram_total": 32 * 1024**2}
                ]
            },
        )

    now = [0.0]

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    client = httpx.Client(
        base_url="http://comfy.test",
        transport=httpx.MockTransport(handler),
    )
    provider = ComfyUIProvider(
        settings(tmp_path, timeout=2),
        client=client,
        monotonic=clock,
        sleep=sleep,
        global_free_vram_mb=lambda: 2000,
    )

    with pytest.raises(AppError) as error:
        provider.unload()

    assert error.value.code is ErrorCode.TIMEOUT
    assert "其他本地模型" in error.value.user_message
    assert now[0] == 2


def test_prompt_compiles_a_short_positive_identity_safe_description() -> None:
    prompt = ComfyUIProvider._prompt(appraisal(), request())

    assert "extreme, near-ruin, visually dominant surface degradation" in prompt
    assert "thick embedded grime" in prompt
    assert "same individual object" in prompt
    assert "every structural detail in the original location" in prompt
    assert "underlying color pattern remains recognizable" in prompt
    assert "functional geometry remains unchanged" in prompt
    assert "shallow" not in prompt
    assert "colors unchanged" not in prompt
    assert "screens" not in prompt
    assert "logos" not in prompt


@pytest.mark.parametrize(
    ("years", "expected"),
    [
        (0.25, "strongly pronounced"),
        (1, "severe"),
        (3, "extreme, near-ruin"),
        (10, "devastating, long-neglected"),
        (30, "ruin-grade, barely salvageable"),
    ],
)
def test_weathering_intensity_is_never_subtle(years: float, expected: str) -> None:
    assert ComfyUIProvider._weathering_intensity(years) == expected


def test_all_scenarios_use_strong_surface_damage_language() -> None:
    combined = " ".join(SCENARIO_SURFACE_EFFECTS.values()).lower()

    assert "faint" not in combined
    assert "sparse" not in combined
    assert "thin " not in combined
    assert "shallow" not in combined
    for marker in ("thick", "heavy", "broad", "extensive", "established"):
        assert marker in combined


def test_happy_path_injects_only_marked_nodes_and_saves_image(tmp_path: Path) -> None:
    original = workflow()
    submitted: dict = {}
    history_calls = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal history_calls
        if http_request.url.path == "/upload/image":
            assert b"dsb_" in http_request.content
            return httpx.Response(
                200, json={"name": "uploaded.png", "subfolder": "", "type": "input"}
            )
        if http_request.url.path == "/prompt":
            submitted.update(json.loads(http_request.content))
            return httpx.Response(200, json={"prompt_id": "job-1", "node_errors": {}})
        if http_request.url.path == "/history/job-1":
            history_calls += 1
            if history_calls == 1:
                return httpx.Response(200, json={})
            return httpx.Response(
                200,
                json={
                    "job-1": {
                        "status": {"status_str": "success", "completed": True},
                        "outputs": {
                            "4": {
                                "images": [
                                    {
                                        "filename": "result.png",
                                        "subfolder": "DSB",
                                        "type": "output",
                                    }
                                ]
                            }
                        },
                    }
                },
            )
        if http_request.url.path == "/view":
            return httpx.Response(
                200, content=image_bytes("white"), headers={"content-type": "image/png"}
            )
        raise AssertionError(http_request.url)

    now = [0.0]

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    client = httpx.Client(
        base_url="http://comfy.test", transport=httpx.MockTransport(handler)
    )
    provider = ComfyUIProvider(
        settings(tmp_path, original), client=client, monotonic=clock, sleep=sleep
    )

    result = provider.edit(input_image(tmp_path), appraisal(), request())

    assert result is not None and result.is_file()
    with Image.open(result) as generated:
        assert generated.size == (32, 24)
    graph = submitted["prompt"]
    assert graph["1"]["inputs"]["image"] == "uploaded.png"
    assert "extreme, near-ruin, visually dominant" in graph["2"]["inputs"]["text"]
    assert "same individual object" in graph["2"]["inputs"]["text"]
    assert graph["3"]["inputs"]["noise_seed"] != 1
    assert graph["4"]["inputs"]["filename_prefix"].startswith("DSB/")
    assert original["1"]["inputs"]["image"] == "placeholder.png"


def test_missing_marker_fails_before_http(tmp_path: Path) -> None:
    graph = workflow()
    graph["1"].pop("_meta")
    calls = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    client = httpx.Client(
        base_url="http://comfy.test", transport=httpx.MockTransport(handler)
    )
    provider = ComfyUIProvider(settings(tmp_path, graph), client=client)

    with pytest.raises(AppError) as error:
        provider.edit(input_image(tmp_path), appraisal(), request())

    assert error.value.code is ErrorCode.IMAGE_EDIT_FAILED
    assert error.value.retriable is False
    assert calls == 0


def test_node_errors_are_non_retriable(tmp_path: Path) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        if http_request.url.path == "/upload/image":
            return httpx.Response(
                200, json={"name": "upload.png", "subfolder": "", "type": "input"}
            )
        return httpx.Response(200, json={"prompt_id": "x", "node_errors": {"1": {}}})

    client = httpx.Client(
        base_url="http://comfy.test", transport=httpx.MockTransport(handler)
    )
    provider = ComfyUIProvider(settings(tmp_path), client=client)

    with pytest.raises(AppError) as error:
        provider.edit(input_image(tmp_path), appraisal(), request())

    assert error.value.code is ErrorCode.IMAGE_EDIT_FAILED
    assert error.value.retriable is False


def test_history_oom_has_specific_error(tmp_path: Path) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        if http_request.url.path == "/upload/image":
            return httpx.Response(
                200, json={"name": "upload.png", "subfolder": "", "type": "input"}
            )
        if http_request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "oom"})
        return httpx.Response(
            200,
            json={
                "oom": {
                    "status": {
                        "status_str": "error",
                        "completed": True,
                        "messages": [
                            [
                                "execution_error",
                                {"exception_message": "CUDA out of memory"},
                            ]
                        ],
                    }
                }
            },
        )

    client = httpx.Client(
        base_url="http://comfy.test", transport=httpx.MockTransport(handler)
    )
    provider = ComfyUIProvider(settings(tmp_path), client=client)

    with pytest.raises(AppError) as error:
        provider.edit(input_image(tmp_path), appraisal(), request())

    assert error.value.code is ErrorCode.GPU_OOM
    assert error.value.retriable is True


def test_total_deadline_times_out_and_removes_queued_job(tmp_path: Path) -> None:
    queue_delete: dict = {}
    targeted_interrupt: dict = {}
    cancellation_order: list[str] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        if http_request.url.path == "/upload/image":
            return httpx.Response(
                200, json={"name": "upload.png", "subfolder": "", "type": "input"}
            )
        if http_request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "slow"})
        if http_request.url.path == "/history/slow":
            return httpx.Response(200, json={})
        if http_request.url.path == "/interrupt":
            cancellation_order.append("interrupt")
            targeted_interrupt.update(json.loads(http_request.content))
            return httpx.Response(200, json={})
        if http_request.url.path == "/queue":
            cancellation_order.append("queue")
            queue_delete.update(json.loads(http_request.content))
            return httpx.Response(200, json={})
        raise AssertionError(http_request.url)

    now = [0.0]

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    client = httpx.Client(
        base_url="http://comfy.test", transport=httpx.MockTransport(handler)
    )
    provider = ComfyUIProvider(
        settings(tmp_path, timeout=2), client=client, monotonic=clock, sleep=sleep
    )

    with pytest.raises(AppError) as error:
        provider.edit(input_image(tmp_path), appraisal(), request())

    assert error.value.code is ErrorCode.TIMEOUT
    assert now[0] == 2
    assert targeted_interrupt == {"prompt_id": "slow"}
    assert queue_delete == {"delete": ["slow"]}
    assert cancellation_order == ["queue", "interrupt"]


def test_output_path_traversal_is_rejected(tmp_path: Path) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        if http_request.url.path == "/upload/image":
            return httpx.Response(
                200, json={"name": "upload.png", "subfolder": "", "type": "input"}
            )
        if http_request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "job"})
        return httpx.Response(
            200,
            json={
                "job": {
                    "status": {"completed": True},
                    "outputs": {
                        "4": {
                            "images": [
                                {
                                    "filename": "result.png",
                                    "subfolder": "../secret",
                                    "type": "output",
                                }
                            ]
                        }
                    },
                }
            },
        )

    client = httpx.Client(
        base_url="http://comfy.test", transport=httpx.MockTransport(handler)
    )
    provider = ComfyUIProvider(settings(tmp_path), client=client)

    with pytest.raises(AppError) as error:
        provider.edit(input_image(tmp_path), appraisal(), request())

    assert error.value.code is ErrorCode.IMAGE_EDIT_FAILED


def test_text_only_edit_uses_text_workflow_without_upload(tmp_path: Path) -> None:
    original = text_workflow()
    submitted: dict = {}
    calls: list[str] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        calls.append(http_request.url.path)
        assert http_request.url.path != "/upload/image"
        if http_request.url.path == "/prompt":
            submitted.update(json.loads(http_request.content))
            return httpx.Response(200, json={"prompt_id": "text-job", "node_errors": {}})
        if http_request.url.path == "/history/text-job":
            return httpx.Response(
                200,
                json={
                    "text-job": {
                        "status": {"status_str": "success", "completed": True},
                        "outputs": {
                            "3": {
                                "images": [
                                    {
                                        "filename": "text-result.png",
                                        "subfolder": "DSB",
                                        "type": "output",
                                    }
                                ]
                            }
                        },
                    }
                },
            )
        if http_request.url.path == "/view":
            return httpx.Response(
                200,
                content=image_bytes("gray"),
                headers={"content-type": "image/png"},
            )
        raise AssertionError(http_request.url)

    client = httpx.Client(
        base_url="http://comfy.test", transport=httpx.MockTransport(handler)
    )
    provider = ComfyUIProvider(settings(tmp_path, text_graph=original), client=client)

    result = provider.edit(None, appraisal(), request())

    assert result is not None and result.is_file()
    assert calls == ["/prompt", "/history/text-job", "/view"]
    graph = submitted["prompt"]
    assert "测试充电宝" in graph["1"]["inputs"]["text"]
    assert "city blackout" in graph["1"]["inputs"]["text"]
    assert graph["2"]["inputs"]["noise_seed"] != 1
    assert graph["3"]["inputs"]["filename_prefix"].startswith("DSB/")
    assert original["1"]["inputs"]["text"] == "placeholder"
    assert original["2"]["inputs"]["noise_seed"] == 1
