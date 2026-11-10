"""msgpack + numpy serialisation of the env protocol (needs only `msgpack` and `numpy`: no openpi dependency).

Arrays travel as {"__ndarray__": True, "data": <raw bytes>, "dtype": <str>, "shape": [...]} and numpy scalars as
{"__npscalar__": True, ...}. Unpacked arrays are READ-ONLY views on the received bytes: copy before writing to them
(torch.from_numpy on a read-only array warns; use np.array(x) first).
"""
from __future__ import annotations

import msgpack
import numpy as np


def _encode(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype.hasobject or obj.dtype.kind in "VOUS":
            raise TypeError(f"cannot serialise an array of dtype {obj.dtype}")
        return {"__ndarray__": True, "data": np.ascontiguousarray(obj).tobytes(), "dtype": obj.dtype.str,
                "shape": list(obj.shape)}
    if isinstance(obj, np.generic):
        return {"__npscalar__": True, "data": obj.item(), "dtype": obj.dtype.str}
    raise TypeError(f"cannot serialise {type(obj)}")


def _decode(obj):
    if "__ndarray__" in obj:
        return np.ndarray(buffer=obj["data"], dtype=np.dtype(obj["dtype"]), shape=tuple(obj["shape"]))
    if "__npscalar__" in obj:
        return np.dtype(obj["dtype"]).type(obj["data"])
    return obj


def pack(message: dict) -> bytes:
    return msgpack.packb(message, default=_encode, use_bin_type=True)


def unpack(raw: bytes) -> dict:
    return msgpack.unpackb(raw, object_hook=_decode, raw=False)
