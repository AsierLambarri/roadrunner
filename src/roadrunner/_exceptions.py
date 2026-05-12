class RoadrunnerError(Exception):
    pass


class ConfigurationError(RoadrunnerError):
    pass


class SnapshotLoadError(RoadrunnerError):
    pass


class NoBoundParticlesError(RoadrunnerError):
    pass


class ConvergenceError(RoadrunnerError):
    pass


class RestartError(RoadrunnerError):
    pass
