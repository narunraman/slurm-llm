"""OpenAI-compatible local proxy that routes each request to the vLLM server for its model."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field

from aiohttp import ClientError, ClientSession, ClientTimeout, web
from rich.console import Console

from .config import ConfigLoader
from .health import detect_vllm_port
from .job import JobManager
from .ssh_manager import SSHConnectionManager
from .tunnels import TunnelHandle, TunnelRegistry

console = Console()


@dataclass
class ActiveModel:
    """A model reachable through a local SSH port forward."""

    model_name: str  # routing key: model id, plus ":N" for instance N > 0
    job_id: str
    cluster_name: str
    node: str
    remote_port: int  # vLLM port on the compute node
    local_port: int  # forwarded port on this machine
    tunnel: TunnelHandle | None = None
    last_used: float = field(default_factory=time.time)
    instance: int = 0
    status: str = "healthy"  # healthy, unhealthy or unknown
    served_name: str | None = None  # the name vLLM serves the model under


class PortAllocator:
    """Hands out local ports for tunnels from a fixed range."""

    def __init__(self, start: int = 30000, end: int = 30100):
        self.start = start
        self.end = end
        self.available = set(range(start, end))
        self.allocated: dict[int, str] = {}

    def allocate(self, model_name: str) -> int:
        if not self.available:
            raise RuntimeError(f"No free local ports in {self.start}-{self.end}")
        port = min(self.available)
        self.available.remove(port)
        self.allocated[port] = model_name
        return port

    def free(self, port: int) -> None:
        if self.allocated.pop(port, None):
            self.available.add(port)


def _split_instance(model_name: str) -> tuple[str, int]:
    """"qwen-7b:1" -> ("qwen-7b", 1); names without a numeric suffix are instance 0."""
    if ":" in model_name and model_name.rsplit(":", 1)[-1].isdigit():
        name, instance = model_name.rsplit(":", 1)
        return name, int(instance)
    return model_name, 0


def _error(message: str, status: int) -> web.Response:
    return web.json_response({"error": {"message": message}}, status=status)


class OpenAIAPIProxy:
    """Serves /v1/chat/completions and /v1/completions for every running model."""

    def __init__(self, port: int = 8000, auto_activate: bool = True):
        self.port = port
        self.auto_activate = auto_activate
        self.app = web.Application()
        self.job_manager = JobManager()
        self.config_loader = ConfigLoader()
        self.ssh_manager = SSHConnectionManager(self.config_loader)
        # forwards are recorded, so ones left behind by a proxy that was killed can be cancelled later
        self.tunnel_registry = TunnelRegistry()
        self.active_models: dict[str, ActiveModel] = {}
        self.port_allocator = PortAllocator()
        self.health_task: asyncio.Task | None = None
        self.stopped = asyncio.Event()
        self._activation_lock = asyncio.Lock()
        self.setup_routes()

    def setup_routes(self):
        r = self.app.router
        r.add_post("/v1/chat/completions", self.chat_completions)
        r.add_post("/v1/completions", self.completions)
        r.add_get("/v1/models", self.list_models)
        r.add_get("/health", self.health_check)
        r.add_post("/v1/models/{model_name}/activate", self.activate_model)
        r.add_delete("/v1/models/{model_name}/deactivate", self.deactivate_model)
        r.add_get("/v1/models/{model_name}/status", self.model_status)
        r.add_get("/jobs", self.list_jobs)

    def _routing_key(self, model_name: str) -> tuple[str, str, int] | None:
        """(routing key, name without instance, instance), or None if the model is unknown."""
        name, instance = _split_instance(model_name)
        model = self.config_loader.resolve_model(name)
        if not model:
            return None
        return (f"{model.id}:{instance}" if instance else model.id), name, instance

    async def chat_completions(self, request: web.Request) -> web.StreamResponse:
        return await self._proxy_request(request, "/v1/chat/completions")

    async def completions(self, request: web.Request) -> web.StreamResponse:
        return await self._proxy_request(request, "/v1/completions")

    async def _proxy_request(self, request: web.Request, path: str) -> web.StreamResponse:
        try:
            data = await request.json()
        except Exception:
            return _error("Request body must be JSON", 400)
        model_name = data.get("model")
        if not model_name:
            return _error("Missing 'model' field in request", 400)
        routed = self._routing_key(model_name)
        if not routed:
            return _error(f"Model '{model_name}' not found in the model registry", 404)
        key, name, instance = routed

        active = self.active_models.get(key)
        if not active:
            if not self.auto_activate:
                return _error(f"Model '{model_name}' is not active; POST /v1/models/{model_name}/activate first", 503)
            console.print(f"[dim]Activating '{model_name}'...[/dim]")
            if not await self._auto_activate_model(name, instance=instance):
                return _error(
                    f"Model '{model_name}' has no running job. Start one with `sllm prepare {model_name}`.", 503
                )
            active = self.active_models[key]

        active.last_used = time.time()
        if not active.served_name:
            active.served_name = await self._query_served_model_name(active.local_port)
        if active.served_name:
            data["model"] = active.served_name  # clients may use short names
        try:
            return await self._forward_to_backend(request, f"http://localhost:{active.local_port}{path}", data)
        except Exception as e:
            console.print(f"[red]Error forwarding to {model_name}: {e}[/red]")
            return _error(f"Internal server error: {e}", 500)

    def _model_entry(self, key: str, am: ActiveModel) -> dict:
        """A /v1/models entry: listed under the short registry name (what the README tells clients
        to use), with the other names requests may use as `aliases` (the Hugging Face repo id, the
        registry id and the name vLLM serves the model under), each with ":N" for instance N."""
        name, instance = _split_instance(key)
        suffix = f":{instance}" if instance else ""
        model = self.config_loader.resolve_model(name)
        names = [model.name, model.hf_repo, model.id, model.served_name] if model else [name]
        names = list(dict.fromkeys(n + suffix for n in names))
        return {
            "id": names[0],
            "object": "model",
            "created": 0,
            "owned_by": "slurm-llm",
            "hf_id": model.hf_repo if model else None,
            "aliases": names[1:],
            "instance": instance,
            "cluster": am.cluster_name,
            "node": am.node,
            "local_port": am.local_port,
            "remote_port": am.remote_port,
            "status": am.status,
            "last_used": am.last_used,
        }

    async def list_models(self, request: web.Request) -> web.Response:
        data = [self._model_entry(key, am) for key, am in self.active_models.items()]
        return web.json_response({"object": "list", "data": data})

    async def health_check(self, request: web.Request) -> web.Response:
        jobs = self.job_manager.list_jobs(refresh=False)
        return web.json_response({
            "status": "healthy",
            "proxy_port": self.port,
            "active_models": list(self.active_models),
            "active_model_count": len(self.active_models),
            "running_jobs": [
                {"job_id": job_id, "model": job.model_name, "node": job.node, "status": job.status}
                for job_id, job in jobs.items()
                if job.status == "RUNNING"
            ],
        })

    async def list_jobs(self, request: web.Request) -> web.Response:
        def refresh():
            clients = {}
            for name in self.config_loader.load_clusters():
                try:
                    clients[name] = self.ssh_manager.get_client(name)
                except Exception as e:
                    console.print(f"[yellow]Could not reach {name}: {e}[/yellow]")
            return self.job_manager.list_jobs(ssh_clients=clients, refresh=True)

        try:
            jobs = await asyncio.to_thread(refresh)
            return web.json_response({"jobs": [job.to_dict() for job in jobs.values()]})
        except Exception as e:
            return _error(f"Error listing jobs: {e}", 500)

    def _create_tunnel(
        self, cluster_name: str, node: str, local_port: int, remote_port: int, model: str = ""
    ) -> TunnelHandle:
        """Forward localhost:local_port to node:remote_port through the cluster's master connection."""
        ssh_client = self.ssh_manager.get_client(cluster_name)
        handle = TunnelHandle(
            local_port=local_port,
            forward_spec=f"{local_port}:{node}:{remote_port}",
            ssh_target=ssh_client.destination,
            control_socket=ssh_client.ssh_master_socket,
            ssh_options=(),
            owner_pid=os.getpid(),
        )
        console.print(f"[dim]Creating tunnel: localhost:{local_port} → {node}:{remote_port}[/dim]")
        result = subprocess.run(handle.command("forward"), capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            raise RuntimeError(f"Could not create the tunnel: {result.stderr.strip() or 'ssh failed'}")
        self.tunnel_registry.register_tunnel(handle, model=model, cluster=cluster_name)
        return handle

    def _close_tunnel(self, am: ActiveModel) -> None:
        """Cancel one forward, leaving the shared connection (and other forwards) open."""
        if am.tunnel:
            try:
                subprocess.run(am.tunnel.command("cancel"), capture_output=True, timeout=10)
            except (OSError, subprocess.TimeoutExpired) as e:
                console.print(f"[yellow]Could not close the tunnel on port {am.local_port}: {e}[/yellow]")
            self.tunnel_registry.unregister_tunnel(am.tunnel.local_port, am.tunnel.owner_pid)
        self.port_allocator.free(am.local_port)

    async def _forward_to_backend(self, request: web.Request, url: str, data: dict) -> web.StreamResponse:
        """POST to the tunnelled vLLM server, streaming the response back if requested."""
        try:
            async with ClientSession() as session:
                async with session.post(url, json=data, timeout=ClientTimeout(total=None, sock_connect=30)) as resp:
                    if data.get("stream"):
                        response = web.StreamResponse(status=resp.status)
                        response.headers["Content-Type"] = resp.headers.get("Content-Type", "text/event-stream")
                        response.headers["Cache-Control"] = "no-cache"
                        await response.prepare(request)
                        async for chunk in resp.content.iter_any():
                            await response.write(chunk)
                        await response.write_eof()
                        return response
                    body = await resp.read()
                    return web.Response(body=body, status=resp.status, content_type=resp.content_type)
        except ClientError as e:
            return _error(f"Backend connection failed: {e}", 502)

    async def _query_served_model_name(self, local_port: int) -> str | None:
        try:
            async with ClientSession() as session:
                async with session.get(f"http://localhost:{local_port}/v1/models", timeout=ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        models = (await resp.json()).get("data") or []
                        return models[0].get("id") if models else None
        except Exception as e:
            console.print(f"[yellow]Could not read the served model name: {e}[/yellow]")
        return None

    async def _auto_activate_model(self, model_name: str, instance: int = 0) -> bool:
        """Tunnel to a RUNNING job of the model, if there is one. Returns True on success."""
        async with self._activation_lock:
            routed = self._routing_key(f"{model_name}:{instance}" if instance else model_name)
            if not routed:
                return False
            key = routed[0]
            if key in self.active_models:
                return True
            model = self.config_loader.resolve_model(model_name)
            job = next(
                (
                    j for j in self.job_manager.list_jobs(refresh=False).values()
                    if j.status == "RUNNING" and j.instance == instance
                    and getattr(self.config_loader.resolve_model(j.model_name), "id", j.model_id) == model.id
                ),
                None,
            )
            if not job:
                return False
            try:
                am = await asyncio.to_thread(self._activate_job, job, key, instance)
            except Exception as e:
                console.print(f"[yellow]Could not activate '{model_name}': {e}[/yellow]")
                return False
            if not am:
                return False
            am.served_name = await self._query_served_model_name(am.local_port)
            self.active_models[key] = am
            display = f"{model_name}:{instance}" if instance else model_name
            console.print(f"[green]✓ Activated '{display}' on localhost:{am.local_port}[/green]")
            return True

    def _activate_job(self, job, key: str, instance: int) -> ActiveModel | None:
        """Blocking part of activation: check the job, find its port, open the tunnel."""
        ssh_client = self.ssh_manager.get_client(job.cluster_name)
        status, node, _ = ssh_client.get_job_status(job.job_id)
        if status != "RUNNING" or not node:
            return None
        remote_port = detect_vllm_port(job, ssh_client, timeout=30)
        local_port = self.port_allocator.allocate(key)
        try:
            tunnel = self._create_tunnel(job.cluster_name, node, local_port, remote_port, model=key)
        except Exception:
            self.port_allocator.free(local_port)
            raise
        return ActiveModel(
            model_name=key,
            job_id=job.job_id,
            cluster_name=job.cluster_name,
            node=node,
            remote_port=remote_port,
            local_port=local_port,
            tunnel=tunnel,
            instance=instance,
        )

    async def _auto_activate_all_running(self):
        jobs = [j for j in self.job_manager.list_jobs(refresh=False).values() if j.status == "RUNNING"]
        if not jobs:
            console.print("[dim]No RUNNING jobs yet; models are activated on their first request.[/dim]")
            return
        console.print(f"[dim]Activating {len(jobs)} RUNNING job(s)...[/dim]")
        activated = 0
        for job in jobs:
            activated += await self._auto_activate_model(job.model_name, instance=job.instance)
        console.print(f"[green]✓ Activated {activated}/{len(jobs)} model(s)[/green]")

    async def activate_model(self, request: web.Request) -> web.Response:
        model_name = request.match_info["model_name"]
        routed = self._routing_key(model_name)
        if not routed:
            return _error(f"Model '{model_name}' not found in the model registry", 404)
        key, name, instance = routed
        if not await self._auto_activate_model(name, instance=instance):
            return _error(f"No RUNNING job for '{model_name}'. Start one with `sllm prepare {model_name}`.", 404)
        am = self.active_models[key]
        return web.json_response({
            "message": f"Model '{model_name}' active",
            "model_name": model_name,
            "job_id": am.job_id,
            "cluster": am.cluster_name,
            "node": am.node,
            "local_port": am.local_port,
            "remote_port": am.remote_port,
        })

    async def deactivate_model(self, request: web.Request) -> web.Response:
        model_name = request.match_info["model_name"]
        routed = self._routing_key(model_name)
        am = self.active_models.pop(routed[0], None) if routed else None
        if not am:
            return _error(f"Model '{model_name}' is not active", 404)
        self._close_tunnel(am)
        console.print(f"[yellow]Model '{model_name}' deactivated[/yellow]")
        return web.json_response({"message": f"Model '{model_name}' deactivated"})

    async def model_status(self, request: web.Request) -> web.Response:
        model_name = request.match_info["model_name"]
        routed = self._routing_key(model_name)
        am = self.active_models.get(routed[0]) if routed else None
        if not am:
            return web.json_response({"model_name": model_name, "active": False})
        return web.json_response({
            "model_name": am.model_name,
            "active": True,
            "job_id": am.job_id,
            "cluster": am.cluster_name,
            "node": am.node,
            "local_port": am.local_port,
            "remote_port": am.remote_port,
            "status": am.status,
            "last_used": am.last_used,
        })

    async def _health_monitor(self):
        """Every minute, check each model's job and tunnel; stop when none is healthy any more."""
        while True:
            await asyncio.sleep(60)
            for key, am in list(self.active_models.items()):
                try:
                    ssh_client = self.ssh_manager.get_client(am.cluster_name)
                    status, node, _ = await asyncio.to_thread(ssh_client.get_job_status, am.job_id)
                    if status != "RUNNING" or node != am.node:
                        console.print(f"[yellow]Model '{key}': job is {status}; marking unhealthy[/yellow]")
                        am.status = "unhealthy"
                        continue
                    async with ClientSession() as session:
                        async with session.get(
                            f"http://localhost:{am.local_port}/health", timeout=ClientTimeout(total=5)
                        ) as resp:
                            am.status = "healthy" if resp.status == 200 else "unhealthy"
                except Exception as e:
                    console.print(f"[yellow]Could not check '{key}': {e}[/yellow]")
                    am.status = "unknown"
            if self.active_models and not any(am.status == "healthy" for am in self.active_models.values()):
                console.print("\n[red]✗ No healthy models left; stopping the proxy.[/red]")
                self.stopped.set()
                return

    async def start_server(self) -> web.AppRunner:
        runner = web.AppRunner(self.app)
        await runner.setup()
        await web.TCPSite(runner, "localhost", self.port).start()

        console.print(f"[green]OpenAI-compatible proxy on http://localhost:{self.port}/v1[/green]")
        console.print("[dim]  POST /v1/chat/completions, /v1/completions · GET /v1/models, /health, /jobs[/dim]")
        if self.auto_activate:
            await self._auto_activate_all_running()
        self.health_task = asyncio.create_task(self._health_monitor())
        return runner

    def cleanup(self):
        if self.health_task:
            self.health_task.cancel()
        for am in list(self.active_models.values()):
            self._close_tunnel(am)
        self.active_models.clear()


async def run_proxy_server(port: int = 8000):
    """Run the proxy until interrupted (or until no model is healthy any more)."""
    proxy = OpenAIAPIProxy(port)
    cleaned = await asyncio.to_thread(proxy.tunnel_registry.cleanup_stale_tunnels)
    if cleaned:
        console.print(f"[dim]Cancelled {cleaned} port forward(s) left by a proxy that was killed[/dim]")
    # `sllm kill` sends SIGTERM, and closing the terminal SIGHUP; asyncio.run handles neither, so
    # without this the forwards on the shared SSH connection would stay open. Stop as on Ctrl+C.
    loop = asyncio.get_running_loop()
    handled = []
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            loop.add_signal_handler(sig, proxy.stopped.set)
            handled.append(sig)
        except (NotImplementedError, RuntimeError, ValueError):  # e.g. not the main thread
            pass
    try:
        try:
            runner = await proxy.start_server()
        except OSError as e:
            from .config import SlurmLLMError

            proxy.cleanup()
            raise SlurmLLMError(f"Could not listen on localhost:{port}: {e.strerror or e}. Try --port.") from None
        try:
            await proxy.stopped.wait()
        finally:
            console.print("[yellow]Shutting down the proxy...[/yellow]")
            proxy.cleanup()
            await runner.cleanup()
    finally:
        for sig in handled:
            loop.remove_signal_handler(sig)
