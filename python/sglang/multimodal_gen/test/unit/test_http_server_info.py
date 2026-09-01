from types import SimpleNamespace

from sglang.multimodal_gen.runtime.entrypoints.http_server import (
    SERVER_INSTANCE_ID,
    _runtime_config_for_server_info,
)


def test_runtime_config_exposes_effective_generation_identity(monkeypatch):
    assert SERVER_INSTANCE_ID
    monkeypatch.setenv("MINIMAX_H3_ADALN_PRECOMPUTE", "1")
    monkeypatch.setenv("MINIMAX_H3_ADALN_TABLE_PATH", "/models/steps20.safetensors")
    monkeypatch.setenv("MINIMAX_H3_LOAD_ADALN_WEIGHTS", "0")
    monkeypatch.setenv("MINIMAX_H3_FORCE_VAE_RESIDENT", "1")
    monkeypatch.setenv("MINIMAX_H3_CONVROT", "1")
    monkeypatch.setenv("MINIMAX_H3_DUMP_LATENTS_PATH", "/runs/latents/{request_id}.pt")
    monkeypatch.setenv("MINIMAX_H3_DEBUG_REUSE_TEXT_EMBEDDINGS", "0")
    server_args = SimpleNamespace(
        backend=SimpleNamespace(value="sglang"),
        model_variant="fl2va",
        transformer_weights_path="/models/dit-v5-convrot",
        component_paths={"text_encoder": "/models/encoder-int8"},
        attention_backend="fa",
        component_attention_backends={"transformer": "fa"},
        performance_mode="memory",
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        layerwise_offload_components=["text_encoder"],
        layerwise_offload_prefetch_size=1.0,
        cache_dit_config=None,
        lora_path=None,
        strict_ports=True,
        port=30500,
        broker_port=30501,
        master_port=30502,
        scheduler_port=30503,
        scheduler_ports=[30503],
    )

    assert _runtime_config_for_server_info(server_args) == {
        "backend": "sglang",
        "model_variant": "fl2va",
        "transformer_weights_path": "/models/dit-v5-convrot",
        "component_paths": {"text_encoder": "/models/encoder-int8"},
        "attention_backend": "fa",
        "component_attention_backends": {"transformer": "fa"},
        "performance_mode": "memory",
        "dit_cpu_offload": False,
        "text_encoder_cpu_offload": False,
        "vae_cpu_offload": False,
        "layerwise_offload_components": ["text_encoder"],
        "layerwise_offload_prefetch_size": 1.0,
        "cache_dit_config": None,
        "lora_path": None,
        "strict_ports": True,
        "minimax_h3": {
            "adaln_precompute": "1",
            "adaln_table_path": "/models/steps20.safetensors",
            "load_adaln_weights": "0",
            "force_vae_resident": "1",
            "convrot_assertion": "1",
            "latent_dump_path_template": "/runs/latents/{request_id}.pt",
            "reuse_text_embeddings": "0",
        },
        "ports": {
            "http": 30500,
            "broker": 30501,
            "master": 30502,
            "schedulers": [30503],
        },
    }


def test_runtime_config_falls_back_to_defaults_and_single_scheduler_port(monkeypatch):
    for name in (
        "MINIMAX_H3_ADALN_PRECOMPUTE",
        "MINIMAX_H3_ADALN_TABLE_PATH",
        "MINIMAX_H3_LOAD_ADALN_WEIGHTS",
        "MINIMAX_H3_FORCE_VAE_RESIDENT",
        "MINIMAX_H3_CONVROT",
        "MINIMAX_H3_DUMP_LATENTS_PATH",
        "MINIMAX_H3_DEBUG_REUSE_TEXT_EMBEDDINGS",
    ):
        monkeypatch.delenv(name, raising=False)
    server_args = SimpleNamespace(
        backend="sglang",
        model_variant=None,
        transformer_weights_path=None,
        component_paths=None,
        attention_backend=None,
        component_attention_backends=None,
        performance_mode="auto",
        dit_cpu_offload=None,
        text_encoder_cpu_offload=None,
        vae_cpu_offload=False,
        layerwise_offload_components=None,
        layerwise_offload_prefetch_size=0.0,
        cache_dit_config=None,
        lora_path=None,
        strict_ports=False,
        port=30000,
        broker_port=30001,
        master_port=30005,
        scheduler_port=5555,
        scheduler_ports=None,
    )

    config = _runtime_config_for_server_info(server_args)

    assert config["component_paths"] == {}
    assert config["component_attention_backends"] == {}
    assert config["layerwise_offload_components"] == []
    assert config["minimax_h3"] == {
        "adaln_precompute": "1",
        "adaln_table_path": (
            "/srv/models/MiniMax-H3-adaln-table-hardened/steps20.safetensors"
        ),
        "load_adaln_weights": "0",
        "force_vae_resident": "1",
        "convrot_assertion": None,
        "latent_dump_path_template": None,
        "reuse_text_embeddings": "0",
    }
    assert config["ports"]["schedulers"] == [5555]
