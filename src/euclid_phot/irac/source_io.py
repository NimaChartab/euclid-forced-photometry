"""Portable fitted-source checkpoints without pickling Tractor/image graphs.

These sources are intended for frozen-morphology measurement. Shape coordinate
systems and freeze masks are preserved; optimizer objects/caches are not saved.
"""
from __future__ import annotations

import json
from pathlib import Path

from tractor import DevGalaxy, ExpGalaxy, GalaxyShape, NanoMaggies, PointSource, RaDecPos
from tractor.ellipses import EllipseE, EllipseESoft
from tractor.galaxy import FixedCompositeGalaxy, FracDev, SoftenedFracDev
from tractor.sersic import SersicGalaxy, SersicIndex

SCHEMA_VERSION = 1
_PARAMETERS = {cls.__name__: cls for cls in (
    GalaxyShape, EllipseE, EllipseESoft, FracDev, SoftenedFracDev, SersicIndex,
)}


def _state(obj):
    state = {}
    if hasattr(obj, "liquid"):
        state["liquid"] = [bool(v) for v in obj.liquid]
    if hasattr(obj, "subs"):
        state["subs"] = [_state(sub) for sub in obj.subs]
    return state


def _restore_state(obj, state):
    if "liquid" in state:
        if len(state["liquid"]) != len(obj.liquid):
            raise ValueError("source freeze-mask layout changed")
        obj.liquid[:] = state["liquid"]
    if "subs" in state:
        if len(state["subs"]) != len(obj.subs):
            raise ValueError("source parameter layout changed")
        for sub, substate in zip(obj.subs, state["subs"]):
            _restore_state(sub, substate)


def _parameter(obj):
    name = type(obj).__name__
    if name not in _PARAMETERS or type(obj) is not _PARAMETERS[name]:
        raise ValueError(f"unsupported shape/parameter type: {type(obj)}")
    return {"type": name, "values": [float(v) for v in obj.getAllParams()]}


def _decode_parameter(record):
    return _PARAMETERS[record["type"]](*record["values"])


def encode_sources(sources):
    records = []
    for source in sources:
        name = type(source).__name__
        if name not in {"PointSource", "SimpleGalaxy", "ExpGalaxy", "DevGalaxy",
                        "FixedCompositeGalaxy", "SersicGalaxy"}:
            raise ValueError(f"unsupported source class: {type(source)}")
        brightness = source.getBrightness()
        if type(brightness) is not NanoMaggies:
            raise ValueError("source checkpoint requires NanoMaggies brightness")
        record = {
            "type": name,
            "position": [float(source.pos.ra), float(source.pos.dec)],
            "bands": list(brightness.order),
            "fluxes": [float(brightness.getFlux(b)) for b in brightness.order],
            "state": _state(source),
        }
        if name == "FixedCompositeGalaxy":
            for key in ("shapeExp", "shapeDev", "fracDev"):
                record[key] = _parameter(getattr(source, key))
        elif name != "PointSource":
            record["shape"] = _parameter(source.shape)
        if name == "SersicGalaxy":
            record["sersicindex"] = _parameter(source.sersicindex)
        if hasattr(source, "halfsize"):
            record["halfsize"] = None if source.halfsize is None else int(source.halfsize)
        records.append(record)
    return {"schema": SCHEMA_VERSION, "sources": records}


def decode_sources(document):
    if document["schema"] != SCHEMA_VERSION:
        raise ValueError("unsupported source checkpoint schema")
    sources = []
    for record in document["sources"]:
        pos = RaDecPos(*record["position"])
        bright = NanoMaggies(order=record["bands"],
                            **dict(zip(record["bands"], record["fluxes"])))
        name = record["type"]
        if name == "PointSource":
            source = PointSource(pos, bright)
        elif name == "FixedCompositeGalaxy":
            source = FixedCompositeGalaxy(pos, bright,
                *[_decode_parameter(record[key])
                  for key in ("fracDev", "shapeExp", "shapeDev")])
        elif name == "SimpleGalaxy":
            # Imported only when explicitly restoring the reference's custom
            # source class; importing euclid_phot.irac itself remains independent.
            import importlib
            SimpleGalaxy = importlib.import_module(
                "euclid_phot.selection"
            ).SimpleGalaxy
            source = SimpleGalaxy(pos, bright)
            source.shape = _decode_parameter(record["shape"])
        elif name == "SersicGalaxy":
            source = SersicGalaxy(pos, bright, _decode_parameter(record["shape"]),
                                 _decode_parameter(record["sersicindex"]))
        elif name in ("ExpGalaxy", "DevGalaxy"):
            cls = ExpGalaxy if name == "ExpGalaxy" else DevGalaxy
            source = cls(pos, bright, _decode_parameter(record["shape"]))
        else:
            raise ValueError(f"unsupported source class: {name}")
        if "halfsize" in record:
            source.halfsize = record["halfsize"]
        _restore_state(source, record["state"])
        sources.append(source)
    return sources


def write_sources(path, sources):
    Path(path).write_text(json.dumps(encode_sources(sources), allow_nan=False))


def read_sources(path):
    return decode_sources(json.loads(Path(path).read_text()))
