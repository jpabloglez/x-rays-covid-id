"""Dataset adapters, and the registry the CLI resolves names against."""

from __future__ import annotations

from cxr.sources.base import Source, SourceError, SourceInfo
from cxr.sources.bimcv import BimcvCovid19
from cxr.sources.covid_radiography import CovidRadiography
from cxr.sources.nih import ChestXray14
from cxr.sources.rsna import RsnaPneumonia

REGISTRY: dict[str, type[Source]] = {
    BimcvCovid19.info.name: BimcvCovid19,
    CovidRadiography.info.name: CovidRadiography,
    ChestXray14.info.name: ChestXray14,
    RsnaPneumonia.info.name: RsnaPneumonia,
}


def get(name: str) -> Source:
    """Instantiate an adapter by name."""
    try:
        return REGISTRY[name]()
    except KeyError as error:
        raise SourceError(
            f"unknown source {name!r}; known sources are {sorted(REGISTRY)}"
        ) from error


def covid_capable() -> list[str]:
    """Sources that can actually supply a COVID label.

    Worth calling before assembling a three-class corpus: if this returns one
    name, every COVID image comes from one place and G4 will say so.
    """
    return sorted(name for name, source in REGISTRY.items() if source.info.has_covid_label)


__all__ = [
    "REGISTRY",
    "ChestXray14",
    "CovidRadiography",
    "RsnaPneumonia",
    "Source",
    "SourceError",
    "SourceInfo",
    "covid_capable",
    "get",
]
