import asyncio
import logging
import random
from typing import AsyncIterable, Sequence, AsyncIterator, Iterable

from lightkube import AsyncClient
from lightkube.types import CascadeType
from lightkube import operators as op
from lightkube.models.batch_v1 import JobSpec
from lightkube.models.core_v1 import (
    Capabilities,
    Container,
    PodSpec,
    PodTemplateSpec,
    ResourceRequirements,
    SeccompProfile,
    SecurityContext,
)
from lightkube.models.meta_v1 import ObjectMeta
from lightkube.resources.batch_v1 import Job
from lightkube.resources.core_v1 import Node, Pod
from rich.progress import Progress, TextColumn
import dataclasses


@dataclasses.dataclass(frozen=True)
class PodUpdate:
    name: str
    node: str
    phase: str


async def amain():
    client = AsyncClient()
    await delete_previous_jobs(client)
    nodes = await get_gpu_nodes(client)
    jobs = await asyncio.gather(*(create_gpu_job(client, node) for node in nodes))
    await progress_tui(nodes, aiter(watch_pods(client)))
    await delete_jobs(client, jobs)
    await client.close()


async def progress_tui(nodes: Sequence[str], updates: AsyncIterator[PodUpdate]):
    progress_bar = Progress(
        TextColumn("{task.description}"), TextColumn("{task.fields[phase]}")
    )
    with progress_bar as progress:
        tasks = {node: progress.add_task(node, phase=None, total=1) for node in nodes}

        while not progress.finished:
            update = await anext(updates)
            task = tasks[update.node]
            progress.update(
                task,
                phase=update.phase,
                completed=(1 if update.phase == "Succeeded" else 0),
            )


async def get_gpu_nodes(client: AsyncClient) -> Sequence[str]:
    nodes = client.list(
        Node,
        labels={
            "nvidia.com/gpu.present": op.equal("true"),
            "nvidia.com/gpu.deploy.container-toolkit": op.equal("true"),
            "nvidia.com/gpu.deploy.operands": op.not_equal("false"),
        },
    )
    return [
        node.metadata.name
        async for node in nodes
        if node.metadata and node.metadata.name
    ]


async def delete_previous_jobs(client: AsyncClient):
    jobs = client.list(Job, labels={"app.kubernetes.io/name": "gpu_check"})
    deletions = [__delete_job(client, job) async for job in jobs]
    await asyncio.gather(*deletions)

    pods = [
        pod
        async for pod in client.list(
            Pod, labels={"app.kubernetes.io/name": "gpu_check"}
        )
    ]
    if len(pods) > 0:
        names = [
            pod.metadata.name if pod.metadata and pod.metadata.name else None
            for pod in pods
        ]
        raise RuntimeError(f"Orphan pods (failed to clean up previous run): {names}")


async def __delete_job(client: AsyncClient, job: Job):
    if not (job.metadata and job.metadata.name):
        raise ValueError(f"Job does not have a name: {job.to_dict()}")
    await client.delete(Job, job.metadata.name, cascade=CascadeType.FOREGROUND)
    logging.info(f"Deleted job/{job.metadata.name}")


async def delete_jobs(client: AsyncClient, jobs: Iterable[Job]):
    await asyncio.gather(*(__delete_job(client, job) for job in jobs))


async def create_gpu_job(client: AsyncClient, node: str) -> Job:
    job = Job(
        metadata=ObjectMeta(
            name=f"gpu-check-{node}",
            labels={
                "app.kubernetes.io/name": "gpu_check",
            },
        ),
        spec=JobSpec(
            completions=1,
            backoffLimit=0,
            activeDeadlineSeconds=300,
            template=PodTemplateSpec(
                metadata=ObjectMeta(
                    labels={
                        "app.kubernetes.io/name": "gpu_check",
                    },
                    annotations={
                        "kubectl.kubernetes.io/default-container": "nvidia-smi"
                    },
                ),
                spec=PodSpec(
                    containers=[
                        Container(
                            name="nvidia-smi",
                            command=["nvidia-smi"],
                            image="docker.io/library/ubuntu:24.04",
                            resources=ResourceRequirements(
                                requests={
                                    "memory": "128Mi",
                                    "cpu": "100m",
                                    "nvidia.com/gpu": "1",
                                },
                                limits={
                                    "memory": "128Mi",
                                    "cpu": "100m",
                                    "nvidia.com/gpu": "1",
                                },
                            ),
                            securityContext=SecurityContext(
                                allowPrivilegeEscalation=False,
                                capabilities=Capabilities(drop=["ALL"]),
                                runAsUser=random.randint(10000, 99999),
                                runAsNonRoot=True,
                                seccompProfile=SeccompProfile(type="RuntimeDefault"),
                            ),
                        )
                    ],
                    nodeSelector={"kubernetes.io/hostname": node},
                    restartPolicy="Never",
                ),
            ),
            ttlSecondsAfterFinished=3600,
        ),
    )
    return await client.create(job)


async def watch_pods(client: AsyncClient) -> AsyncIterable[PodUpdate]:
    watch = client.watch(Pod, labels={"app.kubernetes.io/name": "gpu_check"})
    try:
        async for _, obj in watch:
            if not (
                obj.metadata
                and obj.metadata.name
                and obj.spec
                and obj.spec.nodeSelector
                and obj.status
                and obj.status.phase
            ):
                raise RuntimeError(f"Pod data is incomplete: {obj}")
            yield PodUpdate(
                name=obj.metadata.name,
                node=obj.spec.nodeSelector["kubernetes.io/hostname"],
                phase=obj.status.phase,
            )
    finally:
        await watch.aclose()  # ty:ignore[unresolved-attribute]


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
