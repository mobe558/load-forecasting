"""Model registry - one file per model family (course layout 4.1).
Imports are lazy so the baselines run without PyTorch installed."""


def build_model(name: str, n_past: int, n_fut: int, horizon: int, revin: bool = False, **params):
    if name == "lstm":
        from .lstm import LSTMForecaster as M
    elif name == "transformer":
        from .transformer import TransformerForecaster as M
    elif name == "tft":
        from .tft import TFTForecaster as M
    else:
        raise ValueError(f"unknown model {name}")
    model = M(n_past=n_past, n_fut=n_fut, horizon=horizon, **params)
    if revin:
        from .revin import RevIN
        model = RevIN(model)
    return model
