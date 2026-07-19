def test_model_aliases_include_baselines():
    from medpmc_clip_eval.model import MODEL_ALIASES, PUBLIC_BASELINE_MODELS, model_tag_for_key

    assert MODEL_ALIASES["medpmc"] == "MedPMC-CLIP"
    assert MODEL_ALIASES["bmc"] == "BMC"
    assert MODEL_ALIASES["medsiglip"] == "MedSigLIP"
    assert "bmc" in PUBLIC_BASELINE_MODELS
    assert model_tag_for_key("medpmc-clip") == "MedPMC-CLIP"


def test_cli_model_parser_all():
    from medpmc_clip_eval.cli import build_parser, parse_models

    parser = build_parser()
    args = parser.parse_args(["--models", "all"])
    models = parse_models(args)
    assert models[0] == "medpmc"
    assert "bmc" in models
    assert "medsiglip" in models
