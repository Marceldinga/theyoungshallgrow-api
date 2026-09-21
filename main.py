
# =============================================================================
# Faithi v4.3.0 - model-routing update
#
# IMPORTANT:
# This file contains the drop-in replacement sections for the user's existing
# Faithi v4.2.0 main.py. The rest of the original backend remains unchanged.
# =============================================================================

# Replace APP_VERSION with:
APP_VERSION = "4.3.0"


# =============================================================================
# REPLACE THE EXISTING DEFAULT_MODELS / HF_MODEL_PRIMARY / HF_MODELS BLOCK
# WITH THIS BLOCK
# =============================================================================

DEFAULT_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]

HF_MODEL_PRIMARY = (
    os.getenv(
        "HF_MODEL_PRIMARY",
        "meta-llama/Llama-3.1-8B-Instruct",
    ).strip()
    or "meta-llama/Llama-3.1-8B-Instruct"
)

_env_fallbacks = [
    model.strip()
    for model in os.getenv("HF_MODEL_FALLBACKS", "").split(",")
    if model.strip()
]

HF_MODELS: List[str] = []

for model in [
    HF_MODEL_PRIMARY,
    *_env_fallbacks,
    *DEFAULT_MODELS,
]:
    if model and model not in HF_MODELS:
        HF_MODELS.append(model)


# =============================================================================
# REPLACE _model_order WITH THIS VERSION
# =============================================================================

def _model_order(
    preferred: Optional[str] = None,
    include_disabled: bool = False,
) -> List[str]:
    configured = _unique(
        ([preferred] if preferred else [])
        + [HF_MODEL_PRIMARY]
        + HF_MODELS
    )

    healthy = [
        model
        for model in configured
        if MODEL_HEALTH.get(model, {}).get("ok") is True
    ]

    untested = [
        model
        for model in configured
        if model not in MODEL_HEALTH
    ]

    disabled = [
        model
        for model in configured
        if MODEL_HEALTH.get(model, {}).get("ok") is False
    ]

    ordered: List[str] = []

    if preferred and preferred in configured:
        state = MODEL_HEALTH.get(preferred)
        if state is None or state.get("ok") is True:
            ordered.append(preferred)

    for model in healthy:
        if model not in ordered:
            ordered.append(model)

    for model in untested:
        if model not in ordered:
            ordered.append(model)

    if include_disabled:
        for model in disabled:
            if model not in ordered:
                ordered.append(model)

    return ordered


# =============================================================================
# REPLACE _smart_model_order WITH THIS VERSION
# =============================================================================

def _smart_model_order(
    preferred: Optional[str] = None,
) -> List[str]:
    candidates = _model_order(preferred)

    if not candidates:
        return []

    if preferred and preferred in candidates:
        rest = [
            model
            for model in candidates
            if model != preferred
        ]
        rest.sort(
            key=_model_quality_score,
            reverse=True,
        )
        return [preferred] + rest

    return sorted(
        candidates,
        key=_model_quality_score,
        reverse=True,
    )


# =============================================================================
# REPLACE /models/test WITH THIS VERSION
# =============================================================================

@app.get("/models/test")
def test_models():
    if not HF_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="HF_TOKEN is not configured.",
        )

    results = []

    # Test every configured model, including models that previously failed.
    for model in HF_MODELS:
        results.append(_probe_model(model))

    healthy = [
        result["model"]
        for result in results
        if result.get("ok")
    ]

    disabled = [
        result["model"]
        for result in results
        if not result.get("ok")
    ]

    return {
        "ok": bool(healthy),
        "assistant": ASSISTANT_NAME,
        "primary_model": HF_MODEL_PRIMARY,
        "configured_models": HF_MODELS,
        "healthy_models": healthy,
        "disabled_models": disabled,
        "results": results,
        "next_routing_order": _model_order(),
        "routing_note": (
            "All configured models were tested. "
            "Faithi keeps every model registered but normal customer "
            "requests use healthy models first and skip models currently "
            "known to be unavailable."
        ),
    }


# =============================================================================
# REPLACE /models/healthy WITH THIS VERSION
# =============================================================================

@app.get("/models/healthy")
def healthy_models():
    healthy = [
        model
        for model in HF_MODELS
        if MODEL_HEALTH.get(model, {}).get("ok") is True
    ]

    disabled = [
        model
        for model in HF_MODELS
        if MODEL_HEALTH.get(model, {}).get("ok") is False
    ]

    untested = [
        model
        for model in HF_MODELS
        if model not in MODEL_HEALTH
    ]

    return {
        "assistant": ASSISTANT_NAME,
        "primary_model": HF_MODEL_PRIMARY,
        "configured": HF_MODELS,
        "healthy": healthy,
        "disabled": disabled,
        "untested": untested,
        "next_routing_order": _model_order(),
        "status": MODEL_HEALTH,
    }
