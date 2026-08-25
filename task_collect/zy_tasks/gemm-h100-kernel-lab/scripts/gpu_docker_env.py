from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.docker.docker import DockerEnvironment


class GpuEnabledDockerEnvironment(DockerEnvironment):
    @property
    def capabilities(self) -> EnvironmentCapabilities:
        inherited = super().capabilities
        return inherited.model_copy(update={"gpus": True})
