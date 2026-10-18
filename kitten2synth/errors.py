class Kitten2SynthError(Exception):
    """Base exception for kitten2synth."""


class CatalogError(Kitten2SynthError):
    pass


class EngineUnavailableError(Kitten2SynthError):
    pass


class InvalidBundleError(Kitten2SynthError):
    pass
