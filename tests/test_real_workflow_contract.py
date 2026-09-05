import json

from src.config import PROJECT_ROOT, Settings

WORKFLOW_PATH = PROJECT_ROOT / "workflows" / "flux2_klein_4b_api.json"
TEXT_WORKFLOW_PATH = (
    PROJECT_ROOT / "workflows" / "flux2_klein_4b_text_to_image_api.json"
)


def test_default_settings_select_the_4b_workflow(monkeypatch) -> None:
    monkeypatch.delenv("COMFYUI_WORKFLOW_PATH", raising=False)
    monkeypatch.delenv("COMFYUI_TEXT_WORKFLOW_PATH", raising=False)
    settings = Settings(_env_file=None)

    assert settings.comfyui_workflow_path == WORKFLOW_PATH
    assert settings.comfyui_text_workflow_path == TEXT_WORKFLOW_PATH


def test_real_workflow_is_flat_and_has_unique_provider_markers() -> None:
    graph = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))

    assert graph
    assert "nodes" not in graph
    assert all(isinstance(node.get("inputs"), dict) for node in graph.values())
    assert all(isinstance(node.get("class_type"), str) for node in graph.values())

    expected = {
        "DSB_INPUT": ("LoadImage", "image"),
        "DSB_PROMPT": ("CLIPTextEncode", "text"),
        "DSB_SEED": ("RandomNoise", "noise_seed"),
        "DSB_OUTPUT": ("SaveImage", "filename_prefix"),
    }
    for marker, (class_type, input_name) in expected.items():
        matches = [
            node
            for node in graph.values()
            if node.get("_meta", {}).get("title") == marker
        ]
        assert len(matches) == 1
        assert matches[0]["class_type"] == class_type
        assert input_name in matches[0]["inputs"]


def test_real_workflow_references_only_the_selected_model_stack() -> None:
    raw = WORKFLOW_PATH.read_text(encoding="utf-8").lower()

    assert "flux-2-klein-4b-fp8.safetensors" in raw
    assert "qwen_3_4b.safetensors" in raw
    assert "flux2-vae.safetensors" in raw
    assert "9b" not in raw


def test_real_workflow_uses_identity_preserving_img2img_sampling() -> None:
    graph = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))

    split_nodes = [
        node
        for node in graph.values()
        if node.get("class_type") == "SplitSigmasDenoise"
    ]
    assert len(split_nodes) == 1
    assert split_nodes[0]["inputs"] == {"sigmas": ["16", 0], "denoise": 0.9}
    assert graph["17"]["inputs"]["latent_image"] == ["9", 0]
    assert graph["17"]["inputs"]["sigmas"] == ["20", 1]
    assert graph["2"]["inputs"]["resolution_steps"] == 16
    assert not any(
        node.get("class_type") == "EmptyFlux2LatentImage" for node in graph.values()
    )


def test_text_workflow_is_flat_and_has_only_text_provider_markers() -> None:
    graph = json.loads(TEXT_WORKFLOW_PATH.read_text(encoding="utf-8"))

    assert graph
    assert "nodes" not in graph
    assert all(isinstance(node.get("inputs"), dict) for node in graph.values())
    assert all(isinstance(node.get("class_type"), str) for node in graph.values())
    expected = {
        "DSB_PROMPT": ("CLIPTextEncode", "text"),
        "DSB_SEED": ("RandomNoise", "noise_seed"),
        "DSB_OUTPUT": ("SaveImage", "filename_prefix"),
    }
    for marker, (class_type, input_name) in expected.items():
        matches = [
            node
            for node in graph.values()
            if node.get("_meta", {}).get("title") == marker
        ]
        assert len(matches) == 1
        assert matches[0]["class_type"] == class_type
        assert input_name in matches[0]["inputs"]
    assert not any(
        node.get("_meta", {}).get("title") == "DSB_INPUT"
        for node in graph.values()
    )


def test_text_workflow_uses_the_distilled_four_step_text_to_image_graph() -> None:
    graph = json.loads(TEXT_WORKFLOW_PATH.read_text(encoding="utf-8"))
    raw = TEXT_WORKFLOW_PATH.read_text(encoding="utf-8").lower()

    assert "flux-2-klein-4b-fp8.safetensors" in raw
    assert "qwen_3_4b.safetensors" in raw
    assert "flux2-vae.safetensors" in raw
    assert "9b" not in raw
    assert graph["6"]["class_type"] == "EmptyFlux2LatentImage"
    assert graph["6"]["inputs"] == {
        "width": 1024,
        "height": 1024,
        "batch_size": 1,
    }
    assert graph["10"]["class_type"] == "Flux2Scheduler"
    assert graph["10"]["inputs"] == {"steps": 4, "width": 1024, "height": 1024}
    assert graph["9"]["inputs"]["sampler_name"] == "euler"
    assert graph["8"]["inputs"]["cfg"] == 1.0
    assert graph["11"]["inputs"]["sigmas"] == ["10", 0]
    assert graph["11"]["inputs"]["latent_image"] == ["6", 0]
    assert graph["12"]["inputs"]["samples"] == ["11", 0]
    assert graph["13"]["inputs"]["images"] == ["12", 0]
    forbidden = {"LoadImage", "VAEEncode", "ReferenceLatent", "SplitSigmasDenoise"}
    assert not any(node["class_type"] in forbidden for node in graph.values())
