"""Read both existing AIA metadata schemas without changing pixel/channel order."""

import numpy as np

WAVELENGTHS = [94, 131, 171, 193, 211, 335]
CHANNELS = [f"aia{value}" for value in WAVELENGTHS]


def load_aia_frame(path, expected_shape=(512, 512, 6)):
    with np.load(path, allow_pickle=False) as frame:
        if "x" not in frame.files:
            raise ValueError("AIA object lacks pixel tensor x")
        named = frame["channels"].tolist() if "channels" in frame.files else None
        numeric = frame["wavelengths"].tolist() if "wavelengths" in frame.files else None
        if named is None and numeric is None:
            raise ValueError("AIA object lacks channels/wavelengths metadata; do not guess the order")
        if named is not None and named != CHANNELS:
            raise ValueError("AIA channel order mismatch")
        if numeric is not None and numeric != WAVELENGTHS:
            raise ValueError("AIA wavelength/channel order mismatch")
        x = frame["x"]
        if x.shape != expected_shape:
            raise ValueError("AIA shape mismatch")
        if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
            raise ValueError("AIA tensor has unsupported or nonfinite values")
        return x, {"shape": list(x.shape), "dtype": str(x.dtype), "channels": CHANNELS, "finite": True,
                   "metadata_schema": "channels_and_wavelengths" if named is not None and numeric is not None
                       else "channel_names" if named is not None else "numeric_wavelengths"}
