import yaml

from samplers import (
    QAMSampler,
    PSKSampler,
    RayleighBlockFadingChannelSampler,
    DopplerSpreadChannelSampler,
    CustomBlockFadingChannelSampler,
)

MODULATION_SAMPLERS = {
    "QAM": QAMSampler,
    "PSK": PSKSampler,
}

CHANNEL_SAMPLERS = {
    "RayleighBlockFading": RayleighBlockFadingChannelSampler,
    "DopplerSpread": DopplerSpreadChannelSampler,
    "CustomBlockFading": CustomBlockFadingChannelSampler,
}


def load_config(path):
    """Load a YAML config file and return it as a plain dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_samplers(cfg):
    """Instantiate modulation and channel samplers from a config dict.

    Returns:
        (modulation_sampler, channel_sampler)
    """
    # Modulation sampler
    mod_cfg = cfg["modulation"]
    mod_type = mod_cfg["type"]
    if mod_type not in MODULATION_SAMPLERS:
        raise ValueError(f"Unknown modulation type: {mod_type!r}. Choose from {list(MODULATION_SAMPLERS)}")

    mod_kwargs = {k: v for k, v in mod_cfg.items() if k != "type"}
    modulation_sampler = MODULATION_SAMPLERS[mod_type](**mod_kwargs)

    # Channel sampler
    ch_cfg = cfg["channel"]
    ch_type = ch_cfg["type"]
    if ch_type not in CHANNEL_SAMPLERS:
        raise ValueError(f"Unknown channel type: {ch_type!r}. Choose from {list(CHANNEL_SAMPLERS)}")

    ch_kwargs = {k: v for k, v in ch_cfg.items() if k != "type"}
    # Channel samplers also need n_tx_antennas from the modulation config
    ch_kwargs["n_tx_antennas"] = mod_cfg["n_tx_antennas"]
    channel_sampler = CHANNEL_SAMPLERS[ch_type](**ch_kwargs)

    return modulation_sampler, channel_sampler
