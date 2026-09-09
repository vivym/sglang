import argparse


def media_encoder(_args: argparse.Namespace, extra_argv: list[str]) -> None:
    from sglang.multimodal_gen.runtime.media_encoder.server import main

    main(extra_argv)
