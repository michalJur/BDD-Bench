"""Docker SDK wrapper with streaming output support."""

from __future__ import annotations

from datetime import datetime
import io
import logging
import os
import platform
from pathlib import Path
import re
import time

import docker
from docker.errors import APIError, ImageNotFound, NotFound
from docker.models.containers import Container
from docker.models.images import Image
from dotenv import load_dotenv

load_dotenv()

_client: docker.DockerClient | None = None
_REMOTE_LOOKUP_MAX_ATTEMPTS = 3
_REMOTE_LOOKUP_RETRY_DELAY_SECONDS = 2.0
_REMOTE_LOOKUP_RETRYABLE_STATUS_CODES = {408, 500, 502, 503, 504}
_REMOTE_LOOKUP_RETRYABLE_MESSAGE_SNIPPETS = (
    "timeout",
    "timed out",
    "i/o timeout",
    "proxyconnect",
    "connection refused",
    "connection reset",
    "temporary failure",
    "tls handshake timeout",
)
_VERBOSE_ENV_VAR = "BDD_BENCH_VERBOSE"
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSY_ENV_VALUES = {"0", "false", "no", "off"}


def _is_verbose_enabled(default: bool = True) -> bool:
    raw = os.environ.get(_VERBOSE_ENV_VAR)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUTHY_ENV_VALUES:
        return True
    if normalized in _FALSY_ENV_VALUES:
        return False
    return default


def _find_docker_socket() -> str | None:
    """Find a usable Docker socket, checking common locations."""
    # Check DOCKER_HOST environment variable first
    if os.environ.get("DOCKER_HOST"):
        return None  # Let docker.from_env() handle it

    # Common socket locations to try
    socket_paths = [
        Path("/var/run/docker.sock"),  # Standard Linux/Docker Desktop
        Path.home() / ".colima" / "default" / "docker.sock",  # Colima
        Path.home() / ".rd" / "docker.sock",  # Rancher Desktop
    ]

    for socket_path in socket_paths:
        if socket_path.exists():
            return f"unix://{socket_path}"

    return None


def get_client() -> docker.DockerClient:
    """Get or create the Docker client singleton."""
    global _client
    if _client is None:
        try:
            # Try to find socket if DOCKER_HOST isn't set
            socket_url = _find_docker_socket()
            # Use longer timeout for slow operations (commit, build, etc.)
            timeout = 300  # 5 minutes
            if socket_url:
                _client = docker.DockerClient(base_url=socket_url, timeout=timeout)
            else:
                _client = docker.from_env(timeout=timeout)
        except docker.errors.DockerException as error:
            raise SystemExit(
                "Cannot connect to Docker.\n"
                "  - If using Colima: run 'colima start'\n"
                "  - If using Rancher Desktop: ensure it's running\n"
                f"  Error: {error}"
            ) from error
    return _client


def _registry_auth_config() -> dict[str, str] | None:
    username = os.environ.get("DOCKERHUB_USERNAME")
    token = os.environ.get("DOCKERHUB_TOKEN")
    return {"username": username, "password": token} if username and token else None


def ensure_docker_available() -> None:
    """Verify Docker daemon is accessible."""
    try:
        client = get_client()
        client.ping()
    except docker.errors.DockerException as error:
        raise SystemExit(
            "Cannot connect to Docker daemon.\n"
            "  - If using Colima: run 'colima start'\n"
            "  - If using Docker Desktop: ensure it's running\n"
            f"  Error: {error}"
        ) from error


def image_exists(tag: str) -> bool:
    """Check if a Docker image exists."""
    return _resolve_local_image_reference(tag) is not None


def container_exists(name: str) -> bool:
    """Check if a Docker container exists."""
    try:
        get_client().containers.get(name)
        return True
    except NotFound:
        return False


def remove_image(tag: str, force: bool = True) -> bool:
    """Remove a Docker image if it exists."""
    resolved_tag = _resolve_local_image_reference(tag)
    if resolved_tag is None:
        return True
    logging.info(f"Removing existing image {resolved_tag}")
    try:
        get_client().images.remove(resolved_tag, force=force)
        return True
    except APIError as error:
        raise SystemExit(f"Failed to remove image {resolved_tag}: {error}") from error


def remove_container(name: str, force: bool = True) -> bool:
    """Remove a Docker container if it exists."""
    if not container_exists(name):
        return True
    logging.info(f"Removing existing container {name}")
    try:
        container = get_client().containers.get(name)
        container.remove(force=force)
        return True
    except NotFound:
        # Another worker may have removed it between existence check and remove.
        return True
    except APIError as error:
        if getattr(error, "status_code", None) == 409:
            response = getattr(error, "response", None)
            response_text = ""
            if response is not None:
                response_text = str(getattr(response, "text", "") or "")
            explanation = getattr(error, "explanation", None)
            if isinstance(explanation, str) and explanation.strip():
                message = explanation.lower()
            else:
                message = " ".join(str(part) for part in getattr(error, "args", ()) if part).lower()
            conflict_text = f"{response_text} {message}".lower()
            if "removal" in conflict_text and "in progress" in conflict_text:
                deadline = time.time() + 10.0
                while time.time() < deadline:
                    if not container_exists(name):
                        return True
                    time.sleep(0.2)
                raise SystemExit(
                    f"Timed out waiting for container {name} removal to complete after Docker 409 conflict."
                ) from error
        raise SystemExit(f"Failed to remove container {name}: {error!r}") from error


def _split_repository_and_tag(image_ref: str) -> tuple[str, str]:
    if "@" in image_ref:
        raise ValueError(f"Digest references are not supported here: {image_ref}")

    repository, separator, possible_tag = image_ref.rpartition(":")
    if not separator or "/" in possible_tag:
        return image_ref, "latest"
    return repository, possible_tag


def _render_repository_tag(repository: str, tag: str) -> str:
    return f"{repository}:{tag}"


def _resolve_local_image_reference(image_ref: str) -> str | None:
    candidate_refs = [image_ref]
    try:
        repository, tag = _split_repository_and_tag(image_ref)
    except ValueError:
        repository, tag = image_ref, ""
    if repository and tag:
        rendered = _render_repository_tag(repository, tag)
        if rendered != image_ref:
            candidate_refs.append(rendered)

    client = get_client()
    for candidate in candidate_refs:
        try:
            client.images.get(candidate)
            return candidate
        except ImageNotFound:
            continue
    return None


def list_repository_tags(repository: str) -> list[str]:
    client = get_client()
    prefix = f"{repository}:"
    matching_tags: set[str] = set()
    for image in client.images.list(name=repository):
        for tag in image.tags:
            if tag.startswith(prefix):
                matching_tags.add(tag)
    return sorted(matching_tags)


def _is_retryable_remote_lookup_error(error: APIError) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code in _REMOTE_LOOKUP_RETRYABLE_STATUS_CODES:
        return True

    message = str(error).lower()
    return any(snippet in message for snippet in _REMOTE_LOOKUP_RETRYABLE_MESSAGE_SNIPPETS)


def remote_image_exists(
    image_ref: str,
    *,
    max_attempts: int = _REMOTE_LOOKUP_MAX_ATTEMPTS,
    retry_delay_seconds: float = _REMOTE_LOOKUP_RETRY_DELAY_SECONDS,
    strict: bool = False,
) -> bool:
    client = get_client()
    attempt_limit = max(1, int(max_attempts))
    delay_seconds = max(0.0, float(retry_delay_seconds))

    for attempt in range(1, attempt_limit + 1):
        try:
            client.images.get_registry_data(
                image_ref,
                auth_config=_registry_auth_config(),
            )
            return True
        except NotFound:
            return False
        except APIError as error:
            status_code = getattr(error, "status_code", None)
            if status_code == 404:
                return False
            if status_code in {401, 403}:
                if strict:
                    raise SystemExit(
                        f"Remote image lookup denied for {image_ref} (status {status_code})."
                    ) from error
                logging.warning(
                    f"Remote image lookup denied for {image_ref} (status {status_code}); "
                    "treating as missing and continuing."
                )
                return False
            message = str(error).lower()
            if "not found" in message or "manifest unknown" in message:
                return False
            if "unauthorized" in message or "denied" in message:
                if strict:
                    raise SystemExit(f"Remote image lookup denied for {image_ref}.") from error
                logging.warning(
                    f"Remote image lookup denied for {image_ref}; "
                    "treating as missing and continuing."
                )
                return False
            if status_code == 429 or "toomanyrequests" in message or "rate limit" in message:
                raise SystemExit(
                    "Docker Hub rate limit reached while checking remote image "
                    f"{image_ref}: {error}\n"
                    "Manifest existence checks count against Docker Hub pull limits. "
                    "Reduce scope with --repo/--pr, clean stale stage-13 marker files "
                    "in output_dataset/17_push_markers/, "
                    "wait for limit reset, or switch to a registry with higher limits."
                ) from error

            retryable = _is_retryable_remote_lookup_error(error)
            if retryable and attempt < attempt_limit:
                logging.warning(
                    f"Transient remote image lookup failure for {image_ref} (attempt {attempt}/{attempt_limit}): "
                    f"{error}. Retrying in {delay_seconds:.1f}s."
                )
                time.sleep(delay_seconds)
                continue

            raise SystemExit(f"Failed to query remote image {image_ref}: {error}") from error

    raise SystemExit(f"Failed to query remote image {image_ref}: retry loop exited unexpectedly.")


def remote_image_digest(
    image_ref: str,
    *,
    max_attempts: int = _REMOTE_LOOKUP_MAX_ATTEMPTS,
    retry_delay_seconds: float = _REMOTE_LOOKUP_RETRY_DELAY_SECONDS,
) -> str:
    """Return the registry's immutable digest for an image reference."""
    client = get_client()
    attempt_limit = max(1, int(max_attempts))
    delay_seconds = max(0.0, float(retry_delay_seconds))
    for attempt in range(1, attempt_limit + 1):
        try:
            digest = client.images.get_registry_data(
                image_ref,
                auth_config=_registry_auth_config(),
            ).id
        except (NotFound, APIError) as error:
            if _is_retryable_remote_lookup_error(error) and attempt < attempt_limit:
                logging.warning(
                    "Transient remote image digest failure for %s (attempt %d/%d): "
                    "%s. Retrying in %.1fs.",
                    image_ref,
                    attempt,
                    attempt_limit,
                    error,
                    delay_seconds,
                )
                time.sleep(delay_seconds)
                continue
            raise SystemExit(
                f"Failed to resolve remote image digest for {image_ref}: {error}"
            ) from error
        if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise SystemExit(f"Registry returned an invalid digest for {image_ref}: {digest!r}")
        return digest

    raise SystemExit(
        f"Failed to resolve remote image digest for {image_ref}: retry loop exited unexpectedly."
    )


def tag_image(source_ref: str, target_ref: str) -> str:
    resolved_source = _resolve_local_image_reference(source_ref)
    if resolved_source is None:
        raise SystemExit(f"Source image not found for tagging: {source_ref}")

    target_repository, target_tag = _split_repository_and_tag(target_ref)
    rendered_target = _render_repository_tag(target_repository, target_tag)

    source_image = get_client().images.get(resolved_source)
    source_image.tag(repository=target_repository, tag=target_tag)
    logging.info(f"Tagged image {resolved_source} as {rendered_target}")
    return rendered_target


def push_image(
    source_tag: str,
    target_ref: str,
    *,
    stream_output: bool | None = None,
) -> str:
    if stream_output is None:
        stream_output = _is_verbose_enabled(default=True)
    client = get_client()
    resolved_source = _resolve_local_image_reference(source_tag)
    if resolved_source is None:
        raise SystemExit(f"Source image not found for push: {source_tag}")
    source_image = client.images.get(resolved_source)

    target_repository, target_tag = _split_repository_and_tag(target_ref)
    rendered_target = _render_repository_tag(target_repository, target_tag)

    try:
        source_image.tag(repository=target_repository, tag=target_tag)
        logging.info(f"Pushing image {rendered_target}...")
        push_stream = client.api.push(
            repository=target_repository,
            tag=target_tag,
            stream=True,
            decode=True,
            auth_config=_registry_auth_config(),
        )
        for chunk in push_stream:
            if "error" in chunk:
                raise SystemExit(f"Failed to push image {rendered_target}: {chunk['error']}")
            if not stream_output:
                continue
            status = chunk.get("status")
            detail = chunk.get("progress")
            if status and detail:
                print(f"{status} {detail}")
            elif status:
                print(status)
    except APIError as error:
        raise SystemExit(f"Failed to push image {rendered_target}: {error}") from error

    logging.info(f"Pushed image {rendered_target}")
    return rendered_target


def pull_image(
    source_ref: str,
    *,
    local_tag: str | None = None,
    stream_output: bool | None = None,
) -> Image:
    if stream_output is None:
        stream_output = _is_verbose_enabled(default=True)
    client = get_client()
    if "@" in source_ref:
        source_repository, source_tag = source_ref.rsplit("@", 1)
        rendered_source = source_ref
    else:
        source_repository, source_tag = _split_repository_and_tag(source_ref)
        rendered_source = _render_repository_tag(source_repository, source_tag)

    logging.info(f"Pulling image {rendered_source}...")
    try:
        pull_stream = client.api.pull(
            repository=source_repository,
            tag=source_tag,
            stream=True,
            decode=True,
            auth_config=_registry_auth_config(),
        )
        for chunk in pull_stream:
            if "error" in chunk:
                raise SystemExit(f"Failed to pull image {rendered_source}: {chunk['error']}")
            if not stream_output:
                continue
            status = chunk.get("status")
            detail = chunk.get("progress")
            if status and detail:
                print(f"{status} {detail}")
            elif status:
                print(status)
    except APIError as error:
        raise SystemExit(f"Failed to pull image {rendered_source}: {error}") from error

    try:
        image = client.images.get(rendered_source)
    except ImageNotFound as error:
        raise SystemExit(
            f"Pulled image {rendered_source} could not be resolved locally."
        ) from error

    if local_tag and local_tag != rendered_source:
        local_repository, local_tag_name = _split_repository_and_tag(local_tag)
        rendered_local = _render_repository_tag(local_repository, local_tag_name)
        image.tag(repository=local_repository, tag=local_tag_name)
        logging.info(f"Tagged pulled image as {rendered_local}")
        return client.images.get(rendered_local)

    logging.info(f"Pulled image {rendered_source}")
    return image


def build_image(
    tag: str,
    dockerfile_content: str,
    *,
    stream_output: bool | None = None,
) -> Image:
    """Build a Docker image from Dockerfile content with streaming output."""
    if stream_output is None:
        stream_output = _is_verbose_enabled(default=True)
    client = get_client()

    # Remove existing image if present
    if image_exists(tag):
        remove_image(tag)
        logging.info(f"Removed existing image {tag}")

    platform_arg = None
    if platform.system() == "Darwin":
        platform_arg = "linux/amd64"

    logging.info(f"Building image {tag}...")

    # Build with streaming
    dockerfile_obj = io.BytesIO(dockerfile_content.encode("utf-8"))

    try:
        if stream_output:
            # Use low-level API for streaming
            build_logs = client.api.build(
                fileobj=dockerfile_obj,
                tag=tag,
                rm=True,
                decode=True,
                platform=platform_arg,
            )
            for chunk in build_logs:
                if "stream" in chunk:
                    line = chunk["stream"].rstrip("\n")
                    if line:
                        print(line)
                elif "error" in chunk:
                    raise SystemExit(f"Docker build error: {chunk['error']}")
            # Get the built image
            image = client.images.get(tag)
        else:
            image, _ = client.images.build(
                fileobj=dockerfile_obj,
                tag=tag,
                rm=True,
                platform=platform_arg,
            )
    except APIError as error:
        raise SystemExit(f"Failed to build image {tag}: {error}") from error

    logging.info(f"Built image {tag}")
    return image


def create_container(
    image: str,
    name: str,
    command: str | list[str] = "sleep infinity",
) -> Container:
    """Create a new container from an image."""
    client = get_client()
    container = client.containers.create(
        image,
        command=command,
        name=name,
        detach=True,
    )
    logging.info(f"Created container {name}")
    return container


def start_container(name: str) -> Container:
    """Start an existing container."""
    container = get_client().containers.get(name)
    container.start()
    logging.info(f"Started container {name}")
    return container


def stop_container(name: str, timeout: int = 0) -> None:
    """Stop a running container."""
    try:
        container = get_client().containers.get(name)
        container.stop(timeout=timeout)
    except NotFound:
        pass
    except APIError:
        pass


def execute_script_in_container(
    container_name: str,
    script_content: str,
    log_path: Path,
    *,
    stream_to_console: bool | None = None,
) -> int:
    exit_code, _ = exec_streaming(
        container_name,
        ["bash", "-lc", script_content],
        log_path=log_path,
        stream_to_console=stream_to_console,
    )
    return exit_code


def commit_container(container_name: str, image_ref: str) -> Image:
    """Commit a container to a new image."""
    repository, tag = _split_repository_and_tag(image_ref)
    rendered_target = _render_repository_tag(repository, tag)
    container = get_client().containers.get(container_name)
    image = container.commit(repository=repository, tag=tag)
    logging.info(f"Committed container {container_name} to {rendered_target}")
    return image


def exec_streaming(
    container_name: str,
    command: str | list[str],
    *,
    workdir: str | None = None,
    environment: dict[str, str] | None = None,
    stream_to_console: bool | None = None,
    log_path: Path | None = None,
    max_log_bytes: int | None = None,
    tty: bool = True,
    privileged: bool = False,
) -> tuple[int, str]:
    if stream_to_console is None:
        stream_to_console = _is_verbose_enabled(default=True)
    ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    container = get_client().containers.get(container_name)

    # Create exec instance using low-level API for proper exit code retrieval
    exec_instance = get_client().api.exec_create(
        container.id,
        command,
        stdout=True,
        stderr=True,
        tty=tty,
        workdir=workdir,
        environment=environment,
        privileged=privileged,
    )

    # Start exec with streaming
    output_generator = get_client().api.exec_start(
        exec_instance["Id"],
        stream=True,
        demux=False,
    )

    captured_lines: list[str] = []
    log_file = None
    bytes_written = 0
    truncated_due_to_size_limit = False

    try:
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", encoding="utf-8")
            # Write command at the top of the log file
            cmd_str = " ".join(command) if isinstance(command, list) else command
            start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = (
                f"--- LOG STARTED AT {start_timestamp} ---\n"
                f"=== COMMAND ===\n{cmd_str}\n\n=== OUTPUT ===\n"
            )
            log_file.write(header)
            log_file.flush()
            bytes_written += len(header.encode("utf-8"))

        for chunk in output_generator:
            if chunk:
                # Properly decode with incremental decoder to handle partial characters
                text = chunk.decode("utf-8", errors="replace")
                captured_lines.append(text)

                if stream_to_console:
                    print(text, end="", flush=True)

                if log_file:
                    clean_line = ANSI_ESCAPE.sub("", text)
                    log_file.write(clean_line)
                    log_file.flush()

                    bytes_written += len(clean_line.encode("utf-8"))
                    if max_log_bytes is not None:
                        if bytes_written > max_log_bytes:
                            truncated_due_to_size_limit = True
                            log_file.write(
                                (
                                    "\n--- TEST RUN LOG TOO LONG - STOPPING: exceeded max log file size limit "
                                    f"({max_log_bytes} bytes) ---\n"
                                )
                            )
                            log_file.flush()
                            logging.error(
                                f"Stopping container {container_name}: log exceeded "
                                f"{max_log_bytes} bytes."
                            )
                            try:
                                container.stop(timeout=0)
                            except (NotFound, APIError):
                                pass
                            break

        # Get exit code
        try:
            exec_info = get_client().api.exec_inspect(exec_instance["Id"])
            exit_code = exec_info.get("ExitCode")
        except APIError:
            exit_code = None
        if exit_code is None:
            exit_code = 137 if truncated_due_to_size_limit else 0

        if log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"\n--- LOG COMPLETED AT {timestamp} (exit code: {exit_code}) ---\n")

    finally:
        if log_file:
            log_file.close()

    return exit_code, "".join(captured_lines)
