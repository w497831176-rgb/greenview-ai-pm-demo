"""Code-reviewed trusted capability catalog and Draft-only control helpers.

The catalog is deliberately code-owned.  Database rows are editable Draft
state; a row is never trusted merely because it exists, is enabled, is bound
to an Agent, or has a persuasive name/description.  Only stable identities
listed below may enter a newly compiled RuntimeRelease.

Adding an identity to this module is a supply-chain change and therefore
requires the normal code review, deterministic tests and deployment flow.
The runtime control plane may only enable/disable and bind/unbind these
already reviewed identities.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from db.property_db import (
    get_agent_by_agent_id,
    get_agent_knowledge_bindings,
    get_agent_skills,
    get_agent_tools,
    get_knowledge_doc,
    get_mcp_server,
    get_mcp_tool,
    get_skill,
    list_agents,
    list_knowledge_docs,
    list_mcp_servers,
    list_mcp_tools,
    list_skills,
    set_agent_knowledge_bindings,
    set_agent_skills,
    set_agent_tools,
    set_knowledge_doc_indexed_flag,
    set_skill_enabled,
    toggle_mcp_server_enabled,
    update_agent,
)


CATALOG_VERSION = "trusted-catalog-2026-08-08.1"
TRUSTED_SOURCE = "code_reviewed_manifest"
TRUSTED_STATUS = "trusted"

CAPABILITY_TYPES = {
    "agent",
    "skill",
    "knowledge",
    "mcp_server",
    "mcp_tool",
    "system_tool",
}
VERTICAL_DOMAINS = {"property", "isolated_general"}

# Reviewed implementation fingerprints captured from the controlled v27
# catalog at admission time. Enable flags and Agent bindings are deliberately
# excluded by the safe hash so the runtime control plane may hot-orchestrate
# them; implementation content, schemas and MCP launch definitions are pinned.
_REVIEWED_CONTENT_HASHES: Dict[str, str] = {
    "agent:ACPT_AGENT_宠物服务": "292b5147385f3d97c6c5603318f18970c03be9777e19c3bcc3089f942bcb0db5",
    "agent:billing": "8f4c83bc405d0e757519625ba344181e53640ba533e8a900192df1b12fca8cea",
    "agent:complaint": "6575f55ca5573c8770ee98939f4b27b1f33209e06bde11d9aefaa2def69b839f",
    "agent:customer_service": "8496a6967db047145c82f4c3d8166ae8411651f68420d44bde04f263914e6b4b",
    "agent:maintenance": "5f2f6fb4bc374c458fabcd078f2dd032720644d8634b0ef80b41f0f2c7260559",
    "agent:mars-greenhouse-agent": "e092f5151c6317716516a80ea6008904ea09f574fb0751f2d2912d34ef13f033",
    "agent:router": "abde1f30135434ffc6ff2adf071750e56b1971673ee73c22e2358c84821b3cbc",
    "agent:乱七八糟agent": "18ab182efe0fd8d62ba33ceb56881b709c06442bf49a45c8582f4cc59979e493",
    "agent:儿童教育Agent": "899e1ee14af69d0805362b4c230809389637db473c3654acde0f332d6adc3a6c",
    "agent:系外行星观测 Agent": "aac116fec80d70214abc5ada49e05fe6ed86d74dad803cb38e96abd951c72fef",
    "knowledge:1": "b93fc1d95ce1848dd88444a1a3c38806a7b9de77631142d703b0a3ad895ba978",
    "knowledge:100": "d63077fc87f8bd91986b08d2a9f881790a6f37d6ad51a9bd5f0a7008ccf5f20d",
    "knowledge:120": "9f4800725582422c89cc9cd7002a099be367255ba64a903bbd291379f1c1d6c8",
    "knowledge:123": "e8484459fec753f23a33d5343b0cafd46e0a637e01da9a07dfb805a37b7dfbd2",
    "knowledge:124": "0b5920f19d0b7456bbcb41643f2633f2ce4724508019ce43b3bc947d715a0ffe",
    "knowledge:2": "ba18d23e1ed556a7d088e68964f0d1dade50ea3eefed09a319506fdbda793a96",
    "knowledge:27": "4069ee49c91dacd4db2ef28c71770c18f6739f58fca081b3e88b3d14dde219a6",
    "knowledge:28": "cf794be9e5b65c5179ebc202c555f61ef5dd28def46f5a49da6b302d70e39b25",
    "knowledge:29": "2e0d84ffd0b30d67fa646cca067db2b7653b3338910562de2e257563a2568e62",
    "knowledge:3": "24a771b70a8887d95309c5c3b35a74c668b7645f596ff3f0148667baff4ffc62",
    "knowledge:30": "85d8cff40c2575372cbe3ee58f1d26f54c1a58cf03baa661da52e0b327a47e94",
    "knowledge:31": "33e3273ba94474b0c3996bddac2e19fd334a83e6728ae9405aea3bc8531f52ff",
    "knowledge:32": "43a1040967c7cee7d654bf9b7410a95ce5d6addd558834629ef86d7cc12c9742",
    "knowledge:33": "0fb771f16de1062461639a60d1aa0b8091c25d127391152641c93c24766e105d",
    "knowledge:34": "8e985a2b7b53a26ca0412cec7c3c8c750af357210d65e0de0310fe8aa9a093e9",
    "knowledge:35": "f72bb34303ac076b93940b98484949c67a1f1a0ca81cc1ffc745b7d88b9d281d",
    "knowledge:36": "ac6d6ab97103ef1a5be8b704ca87ac7ea0a5ab800c614f6261e36b8c1dbb7ebe",
    "knowledge:37": "4a3a6364eb3dda6fbb5c94be70530ab88ff7ad2fff2cda1d6aa1f3c8ae4d4cf4",
    "knowledge:38": "0fb5e4afb122a0e3fa4a42263f33625c5b78b9519310479579314911b433b2bc",
    "knowledge:39": "7af59f75e652126a1eb6ea03fb8da1d23c3e912bfbdced1af6fd6649c0281567",
    "knowledge:4": "7af9662f0ef91ac1af1a343ec9449c810f5b0de11ce09917742949d52849dda4",
    "knowledge:40": "c16fa6377c060264c91a911ff45b2b30ccc47489d9ecdc3b9c51f5f318fc41c7",
    "knowledge:41": "6b51b953919cf21f1945c0d04afaf7d629e716e0c07ec1d19836672c7544f09b",
    "knowledge:42": "4b8e7fa078a4acecdf021a6bc5d38cf5fa66f8196c26d4126a3067d1a1e5fdb8",
    "knowledge:43": "be7fa9b38265a01773502fc3f428b4dfe19a5eadd7019117899e13e2272144c1",
    "knowledge:44": "ad5e31c14ffd367abf1fbb027e2ef5ee664a0260ea311632a6c4d3e9f8ee7bf4",
    "knowledge:45": "33f2ca1b1a1854ab6b37ff8950c760ee7ccba844e01cc9afd7875439bb7dba81",
    "knowledge:46": "b9ad09901add3374a27e2a9d6fb5b5a84d71b0ce702b9db958859a0456b93707",
    "knowledge:5": "0e38f34f527556c1aac6dca91dd503c6747ba6aeaf9e24ee63157e7413a9da9c",
    "knowledge:75": "a29b74c3d7c79eca1a9703a64cd2cc9a1b40dc415576aa7f73bc69e05a475a2b",
    "knowledge:76": "42a15c2a8d2a120e8e04745e430ff905bd55297d2279f4709936b3d965e63227",
    "knowledge:91": "afaa913afdabfc54929f7111d6fc2357b5e8fe73bf691d331c5f7f875f42200d",
    "knowledge:93": "ee951db6fd29a78c82615f07245de10b425105a0142abd32b671f38741e23814",
    "knowledge:98": "25165a1f138a5eff053c6b3b87d5f418ff65a036ae2082084f40a25b5ae5d149",
    "mcp_server:500": "d64c1f6ad4ad64658df1c126c9036f9a42a625209e025d4081ff643a1f1cf1ef",
    "mcp_server:501": "2b1c8f31bf70f63c2b997f6f80232b9b9df955c95f3a47e6e901ffc5e8b0ee55",
    "mcp_server:55": "d6e078a6024e9855f6d0362a105ce578448c53e0abbce1652f35c54db80a8d6c",
    "mcp_server:56": "21e889198c6119136d991b4d520775f5a622303eed197ac64f86b81414fb4e2e",
    "mcp_server:8": "d84e477fd60960bd2580181077f52f1860fced62acd133bccde5ca46c43bd233",
    "mcp_tool:1": "fd9e3dc021672f7e4a1ddaa65179d58f0b3e54e64bc17d1641d203f203ed3fc4",
    "mcp_tool:107": "5818f7631c581ea81b1eb0debc84514b668c6f2035a43b80d6b7392cd0ec2d15",
    "mcp_tool:108": "792fbb5a9252154244598d9f9a8d1908861590b8238552df2f47d55b4f41c63b",
    "mcp_tool:109": "86194996eb841ee3316aeb7b31763b299a11e77badc7c8a9aaa125475302e370",
    "mcp_tool:110": "945099906801d0a5ea8bd290bedd421b13c2c07a1bf63e5035579f30c6704011",
    "mcp_tool:111": "72e0c8fed88f9428038c0b4de932dc691018eb8cb7d02df3bb89b4102b3a041e",
    "mcp_tool:2": "57e57f22229f4d30b19346ccb7547cb24325fe0ab81c2c36dd37e84953083e7b",
    "mcp_tool:2182": "8c3909debb8e6efd5d5caf173b64bfe0a76b6b57c818dff2d40c0fab6f827a1e",
    "mcp_tool:2183": "7df92a3af2bd8c6d5e7e7bf21ca83283c274a89c390c3d9363330c0cf8db187f",
    "mcp_tool:2184": "6fb68de447e3a10ce44172b834c5e46852c964874f9cd4f5e318a4feee1aeaa4",
    "mcp_tool:3423": "2fc4158a9ab4dc8ca9165154013b7b4be43f8b320c2f41e97cdd7df55e68e439",
    "mcp_tool:3727": "dbd6e0140f2bb0d1c02007828532d5c14918d6a790b7ca72482ba42f298b2a23",
    "skill:115": "077b224a3dd338472897392dbfc98e84de47361531f23d692eca2a421df11019",
    "skill:116": "b34141f7d129d6026b935a22e6b6ce8183a67e2755a3b6afe12e7c821c9ef30a",
    "skill:117": "95867a61e87370b45e7bc435784a071c1ba05d05d10649b4d462e938de13317c",
    "skill:13": "669362ae497ea2b8044c2a8bf6a33de3449390e20c06d1d1add42112a202bc19",
    "skill:17": "9e5c76331ccf7b0d610d223228bd9dddef5721f574c0e4a159299c1c45c6a3e7",
    "skill:2": "a128ce572660df4277c7c5e17fd8a68fdaa3ca8ac95bc99e8e652f05797c7464",
    "skill:75": "a11f15dddb7dffd9388c49d32cfbdf3866082b76f02f9761424b5a9631e75930",
    "skill:8": "875cdcc6fb0ae9ed5c465c9255391606345d214f6e1d5bfbcb658c7182f5ee85",
    "skill:87": "1e5785e81db239820a7ea6959bd28c7fed2ab741f1757d15c094e7c74e32d689",
    "skill:90": "23c12eeab6a5887b9f8ac2fc8ed8f964a06593b58bb75301de21680086e0214e",
    "skill:93": "fc40c86076ad7c6a9211a26d96db0286955a42ffccbe7c71814907b7548a1800",
}

# External assets that the runtime can actually consume are pinned separately
# from database fields. Values are replaced below with code-reviewed digests;
# the explicit identity list prevents a path from becoming trusted merely by
# existing in mutable storage.
_REVIEWED_ARTIFACT_HASHES: Dict[str, str] = {
    "skill:2": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    "skill:8": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    "skill:13": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    "skill:17": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    "skill:75": "b6f2b2b27f6f1fd714497a23348eaaf9c8aee83b8a0c82090c5856c68332befb",
    "skill:87": "e5f439be00ac480be77d15c255fc02505f000d8ddc034a7a6e5b54865b116e2c",
    "skill:90": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    "skill:93": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    "skill:115": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    "skill:116": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    "skill:117": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    "mcp_server:8": "98040d20959d1c7bdf6ef9183b5998fcb2a3196106826618ccd249e03dc43bac",
    "mcp_server:55": "79ac319aa65416650eac781d7a970ca47a568ddee7e2a8ee403b4f822a7b8013",
    "mcp_server:56": "e30f8d92d356b7f4c1a3f98ce82866086e669777dd1674601a29719588093ece",
    "mcp_server:500": "59ed0980ccd0334a7ce131133eaf4fa73767de560b3c18ec8c1302bae84c53d4",
    "mcp_server:501": "041c320f1b318e1d3a93841ba8af1d48bd5565467e1e836d13de0feebbe27ca2",
}


class CapabilityCatalogError(ValueError):
    """Raised when Draft control attempts to escape the trusted catalog."""


def _entry(
    capability_type: str,
    stable_id: Any,
    domain_scope: str,
    *,
    effect: Optional[str] = None,
    server_id: Optional[int] = None,
) -> Dict[str, Any]:
    reviewed_content_hash = _REVIEWED_CONTENT_HASHES.get(
        f"{capability_type}:{stable_id}"
    )
    if not reviewed_content_hash:
        raise RuntimeError(
            f"trusted manifest entry lacks reviewed content hash: "
            f"{capability_type}/{stable_id}"
        )
    reviewed_artifact_hash = _REVIEWED_ARTIFACT_HASHES.get(
        f"{capability_type}:{stable_id}"
    )
    if capability_type in {"skill", "mcp_server"} and not reviewed_artifact_hash:
        raise RuntimeError(
            f"trusted manifest entry lacks reviewed artifact hash: "
            f"{capability_type}/{stable_id}"
        )
    return {
        "capability_type": capability_type,
        "stable_id": stable_id,
        "version": CATALOG_VERSION,
        "domain_scope": domain_scope,
        "trust_status": TRUSTED_STATUS,
        "source": TRUSTED_SOURCE,
        "effect": effect,
        "server_id": server_id,
        "reviewed_content_hash": reviewed_content_hash,
        "reviewed_artifact_hash": reviewed_artifact_hash,
    }


# Stable technical identities from the already code-reviewed production
# inventory used by RuntimeRelease v27.  Display names and current bindings are
# intentionally absent: neither has authority to define trust or domain.
_MANIFEST_ENTRIES: Tuple[Dict[str, Any], ...] = tuple(
    [
        _entry("agent", "maintenance", "property"),
        _entry("agent", "billing", "property"),
        _entry("agent", "complaint", "property"),
        _entry("agent", "customer_service", "property"),
        _entry("agent", "router", "property"),
        _entry("agent", "儿童教育Agent", "isolated_general"),
        _entry("agent", "系外行星观测 Agent", "isolated_general"),
        _entry("agent", "乱七八糟agent", "isolated_general"),
        _entry("agent", "mars-greenhouse-agent", "isolated_general"),
        _entry("agent", "ACPT_AGENT_宠物服务", "property"),
    ]
    + [
        _entry("skill", skill_id, "property")
        for skill_id in (2, 8, 17, 90, 93)
    ]
    + [
        _entry("skill", skill_id, "isolated_general")
        for skill_id in (13, 87, 115, 116, 117)
    ]
    + [_entry("skill", 75, "control_plane")]
    + [
        _entry("knowledge", doc_id, "property")
        for doc_id in (
            *range(1, 6),
            *range(27, 47),
            75,
            76,
            91,
            93,
            98,
            100,
            120,
        )
    ]
    + [
        _entry("knowledge", doc_id, "isolated_general")
        for doc_id in (123, 124)
    ]
    + [
        _entry("mcp_server", server_id, domain)
        for server_id, domain in (
            (8, "property"),
            (55, "property"),
            (56, "property"),
            (500, "isolated_general"),
            (501, "isolated_general"),
        )
    ]
    + [
        _entry("mcp_tool", tool_id, domain, effect="read", server_id=server_id)
        for tool_id, server_id, domain in (
            (1, 8, "property"),
            (2, 8, "property"),
            (107, 55, "property"),
            (108, 55, "property"),
            (109, 55, "property"),
            (110, 55, "property"),
            (111, 56, "property"),
            (2182, 56, "property"),
            (2183, 56, "property"),
            (2184, 56, "property"),
            (3423, 500, "isolated_general"),
            (3727, 501, "isolated_general"),
        )
    ]
)

_MANIFEST: Dict[Tuple[str, str], Dict[str, Any]] = {
    (entry["capability_type"], str(entry["stable_id"])): entry
    for entry in _MANIFEST_ENTRIES
}


def _normalize_type(capability_type: str) -> str:
    value = str(capability_type or "").strip().lower()
    aliases = {
        "rag": "knowledge",
        "knowledge_doc": "knowledge",
        "knowledge_document": "knowledge",
        "mcp": "mcp_server",
        "tool": "system_tool",
    }
    value = aliases.get(value, value)
    if value not in CAPABILITY_TYPES:
        raise CapabilityCatalogError(f"unsupported capability type: {capability_type!r}")
    return value


def _manifest_key(capability_type: str, stable_id: Any) -> Tuple[str, str]:
    return _normalize_type(capability_type), str(stable_id).strip()


def get_trusted_metadata(
    capability_type: str,
    stable_id: Any,
) -> Optional[Dict[str, Any]]:
    """Return immutable manifest metadata without consulting names/bindings."""

    entry = _MANIFEST.get(_manifest_key(capability_type, stable_id))
    return deepcopy(entry) if entry else None


def trusted_domain_scope(capability_type: str, stable_id: Any) -> str:
    entry = get_trusted_metadata(capability_type, stable_id)
    if not entry:
        raise CapabilityCatalogError(
            f"capability is not in the code-reviewed catalog: "
            f"{_normalize_type(capability_type)}/{stable_id}"
        )
    return str(entry["domain_scope"])


def trusted_capability_ids(capability_type: str) -> set[Any]:
    normalized = _normalize_type(capability_type)
    return {
        entry["stable_id"]
        for entry in _MANIFEST_ENTRIES
        if entry["capability_type"] == normalized
    }


def _runtime_object(capability_type: str, stable_id: Any) -> Optional[Dict[str, Any]]:
    normalized = _normalize_type(capability_type)
    try:
        if normalized == "agent":
            return get_agent_by_agent_id(str(stable_id))
        if normalized == "skill":
            return get_skill(int(stable_id))
        if normalized == "knowledge":
            return get_knowledge_doc(int(stable_id))
        if normalized == "mcp_server":
            return get_mcp_server(int(stable_id))
        if normalized == "mcp_tool":
            return get_mcp_tool(int(stable_id))
    except (TypeError, ValueError):
        return None
    # No independent system Tool table exists in the current data model.  The
    # empty manifest is intentional and non-empty requests fail closed.
    return None


_ARTIFACT_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".yiai-venv",
    "__pycache__",
    "node_modules",
    ".uv-cache",
    "venv",
}
_MAX_SKILL_REFERENCE_BYTES = 256_000
_MAX_MCP_ARTIFACT_BYTES = 8_000_000


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _allowed_artifact_roots() -> List[Tuple[str, Path]]:
    roots = [
        ("data", Path(os.getenv("PROPERTY_DATA_DIR", "/app/data")).resolve()),
        ("app", Path(__file__).resolve().parents[2]),
    ]
    unique: List[Tuple[str, Path]] = []
    for label, root in roots:
        if all(root != existing for _, existing in unique):
            unique.append((label, root))
    return unique


def _assert_controlled_artifact_path(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not any(_is_within(resolved, root) for _, root in _allowed_artifact_roots()):
        raise CapabilityCatalogError(
            f"trusted artifact path escapes controlled roots: {path}"
        )
    return resolved


def _artifact_label(path: Path) -> str:
    for label, root in _allowed_artifact_roots():
        if _is_within(path, root):
            return f"{label}:{path.relative_to(root).as_posix()}"
    raise CapabilityCatalogError(f"trusted artifact path has no controlled label: {path}")


def _is_deployment_env_file(path: Path) -> bool:
    return path.name == ".env" or path.name.startswith(".env.")


def _walk_controlled_files(root: Path) -> List[Path]:
    if root.is_symlink():
        raise CapabilityCatalogError(f"trusted artifact may not be a symlink: {root}")
    if root.is_file():
        return [root]
    files: List[Path] = []
    for current, dir_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        kept_dirs: List[str] = []
        for name in sorted(dir_names):
            if name in _ARTIFACT_IGNORED_DIRS:
                continue
            candidate = current_path / name
            if candidate.is_symlink():
                raise CapabilityCatalogError(
                    f"trusted artifact directory may not be a symlink: {candidate}"
                )
            kept_dirs.append(name)
        dir_names[:] = kept_dirs
        for name in sorted(file_names):
            candidate = current_path / name
            if candidate.is_symlink():
                raise CapabilityCatalogError(
                    f"trusted artifact file may not be a symlink: {candidate}"
                )
            if not candidate.is_file():
                raise CapabilityCatalogError(
                    f"trusted artifact must be a regular file: {candidate}"
                )
            files.append(candidate)
    return sorted(files, key=lambda item: item.as_posix())


def _skill_source_root(skill: Dict[str, Any]) -> Optional[Path]:
    storage_path = str(skill.get("storage_path") or "").strip()
    if storage_path:
        candidate = Path(storage_path)
        if candidate.exists():
            return _assert_controlled_artifact_path(candidate)
    try:
        import skill_storage

        candidate = Path(skill_storage._skill_dir(int(skill["id"])))
        if candidate.exists():
            return _assert_controlled_artifact_path(candidate)
    except (CapabilityCatalogError, TypeError, ValueError):
        raise
    except Exception:
        return None
    return None


def trusted_skill_reference_snapshots(skill: Dict[str, Any]) -> List[Dict[str, str]]:
    """Read the exact immutable Skill references a Release may package.

    Deployment-owned ``.env`` files are never read. A Skill reference named
    like an environment file is invalid rather than silently omitted because
    the runtime must not inject secrets into model context.
    """

    source = _skill_source_root(skill)
    references_root = source / "references" if source else None
    if not references_root or not references_root.is_dir():
        return []
    references_root = _assert_controlled_artifact_path(references_root)
    snapshots: List[Dict[str, str]] = []
    consumed = 0
    for reference in _walk_controlled_files(references_root):
        relative = reference.relative_to(references_root).as_posix()
        if _is_deployment_env_file(reference):
            raise CapabilityCatalogError(
                f"Skill reference may not contain a deployment env file: {relative}"
            )
        raw = reference.read_bytes()
        consumed += len(raw)
        if consumed > _MAX_SKILL_REFERENCE_BYTES:
            raise CapabilityCatalogError(
                f"Skill references exceed reviewed size limit: {skill.get('id')}"
            )
        snapshots.append(
            {
                "path": relative,
                "content": raw.decode("utf-8", errors="replace"),
                "content_hash": hashlib.sha256(raw).hexdigest(),
            }
        )
    return snapshots


def _mcp_artifact_paths(server: Dict[str, Any]) -> List[Path]:
    raw_candidates: List[str] = []
    for value in (
        server.get("package_path"),
        server.get("command"),
        server.get("detected_entrypoint"),
        *(server.get("args") or []),
    ):
        text = str(value or "").strip()
        if text and (text.startswith("/") or text.startswith(".")):
            raw_candidates.append(text)
    resolved: List[Path] = []
    for raw in raw_candidates:
        candidate = Path(raw)
        if not candidate.exists():
            if raw == str(server.get("package_path") or "").strip():
                raise CapabilityCatalogError(
                    f"trusted MCP package path is missing: {server.get('id')}"
                )
            continue
        path = _assert_controlled_artifact_path(candidate)
        if path not in resolved:
            resolved.append(path)
    selected: List[Path] = []
    for path in sorted(resolved, key=lambda item: (len(item.parts), item.as_posix())):
        if any(parent.is_dir() and _is_within(path, parent) for parent in selected):
            continue
        selected.append(path)
    return selected


def _mcp_artifact_records(server: Dict[str, Any]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    consumed = 0
    for root in _mcp_artifact_paths(server):
        for artifact in _walk_controlled_files(root):
            label = _artifact_label(artifact)
            if _is_deployment_env_file(artifact):
                # Secret values are deployment-owned and must never be read or
                # embedded in a code-review hash. Presence is recorded only.
                records.append(
                    {"path": label, "content_hash": "deployment_env_excluded"}
                )
                continue
            raw = artifact.read_bytes()
            consumed += len(raw)
            if consumed > _MAX_MCP_ARTIFACT_BYTES:
                raise CapabilityCatalogError(
                    f"MCP artifact exceeds reviewed size limit: {server.get('id')}"
                )
            records.append(
                {"path": label, "content_hash": hashlib.sha256(raw).hexdigest()}
            )
    records.sort(key=lambda item: item["path"])
    return records


def _artifact_state(capability_type: str, row: Dict[str, Any]) -> Dict[str, Any]:
    if capability_type == "skill":
        snapshots = trusted_skill_reference_snapshots(row)
        records = [
            {"path": item["path"], "content_hash": item["content_hash"]}
            for item in snapshots
        ]
        payload: Dict[str, Any] = {"reference_snapshots": snapshots}
    elif capability_type == "mcp_server":
        records = _mcp_artifact_records(row)
        payload = {}
    else:
        return {"hash": None}
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {**payload, "hash": hashlib.sha256(encoded).hexdigest()}


def _safe_hash(capability_type: str, row: Dict[str, Any]) -> str:
    fields: Dict[str, Sequence[str]] = {
        "agent": (
            "agent_id",
            "name",
            "description",
            "instructions",
            "category",
            "domain_scope",
            "model_id",
        ),
        "skill": (
            "id",
            "name",
            "description",
            "instructions",
            "category",
            "trigger_condition",
            "skill_metadata",
            "storage_path",
            "model_id",
        ),
        "knowledge": (
            "id",
            "title",
            "content",
            "category",
            "source_type",
            "index_status",
            "chunk_size",
            "chunk_overlap",
            "split_strategy",
        ),
        "mcp_server": (
            "id",
            "name",
            "command",
            "args",
            "description",
            "is_builtin",
            "source_type",
            "runtime_type",
            "package_path",
            "detected_entrypoint",
        ),
        "mcp_tool": (
            "id",
            "server_id",
            "name",
            "description",
            "input_schema",
            "tool_metadata",
        ),
    }
    payload = {key: row.get(key) for key in fields.get(capability_type, ())}
    if capability_type == "mcp_server":
        # Pin only required variable names. Secret values remain deployment-
        # owned and are never emitted or included in source-control hashes.
        payload["env_keys"] = sorted(str(key) for key in (row.get("env") or {}))
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _display_name(capability_type: str, row: Dict[str, Any]) -> str:
    if capability_type == "knowledge":
        return str(row.get("title") or row.get("id") or "")
    return str(row.get("name") or row.get("agent_id") or row.get("id") or "")


def _enabled(capability_type: str, row: Dict[str, Any]) -> bool:
    if capability_type == "knowledge":
        return bool(row.get("is_indexed"))
    if capability_type == "mcp_tool":
        server = get_mcp_server(int(row.get("server_id") or 0))
        return bool(server and server.get("enabled"))
    return bool(row.get("enabled", True))


def _catalog_item(
    entry: Dict[str, Any],
    row: Dict[str, Any],
    artifact: Dict[str, Any],
) -> Dict[str, Any]:
    capability_type = str(entry["capability_type"])
    storage_hash = _safe_hash(capability_type, row)
    artifact_hash = artifact.get("hash")
    implementation_hash = (
        hashlib.sha256(f"{storage_hash}:{artifact_hash}".encode("utf-8")).hexdigest()
        if artifact_hash
        else storage_hash
    )
    return {
        **deepcopy(entry),
        "name": _display_name(capability_type, row),
        "enabled": _enabled(capability_type, row),
        "content_hash": implementation_hash,
        "storage_content_hash": storage_hash,
        "artifact_hash": artifact_hash,
    }


def assert_trusted_capability(
    capability_type: str,
    stable_id: Any,
) -> Dict[str, Any]:
    """Resolve one manifest identity and fail closed on any structural drift."""

    normalized = _normalize_type(capability_type)
    entry = get_trusted_metadata(normalized, stable_id)
    if not entry:
        raise CapabilityCatalogError(
            f"capability is not in the code-reviewed catalog: {normalized}/{stable_id}"
        )
    row = _runtime_object(normalized, entry["stable_id"])
    if not row:
        raise CapabilityCatalogError(
            f"trusted capability is missing from Draft storage: "
            f"{normalized}/{entry['stable_id']}"
        )
    actual_content_hash = _safe_hash(normalized, row)
    reviewed_content_hash = str(entry.get("reviewed_content_hash") or "")
    if not reviewed_content_hash or actual_content_hash != reviewed_content_hash:
        raise CapabilityCatalogError(
            f"trusted capability implementation hash drift: "
            f"{normalized}/{entry['stable_id']}"
        )
    artifact = _artifact_state(normalized, row)
    reviewed_artifact_hash = entry.get("reviewed_artifact_hash")
    if reviewed_artifact_hash is not None and artifact.get("hash") != str(
        reviewed_artifact_hash
    ):
        raise CapabilityCatalogError(
            f"trusted capability external artifact hash drift: "
            f"{normalized}/{entry['stable_id']}"
        )
    if normalized == "agent":
        configured_scope = str(row.get("domain_scope") or "")
        if configured_scope != entry["domain_scope"]:
            raise CapabilityCatalogError(
                f"trusted Agent domain drift: {entry['stable_id']}/"
                f"{configured_scope!r}/{entry['domain_scope']!r}"
            )
    if normalized == "mcp_tool":
        actual_server_id = int(row.get("server_id") or 0)
        if actual_server_id != int(entry.get("server_id") or 0):
            raise CapabilityCatalogError(
                f"trusted MCP Tool server drift: {entry['stable_id']}/"
                f"{actual_server_id}/{entry.get('server_id')}"
            )
        if entry.get("effect") != "read":
            raise CapabilityCatalogError(
                f"trusted MCP Tool is not read-only: {entry['stable_id']}"
            )
        declared_effect = str(
            (row.get("tool_metadata") or {}).get("effect")
            or (row.get("tool_metadata") or {}).get("operation")
            or ""
        ).strip().lower()
        if declared_effect and declared_effect != "read":
            raise CapabilityCatalogError(
                f"trusted MCP Tool Draft declares a non-read effect: "
                f"{entry['stable_id']}/{declared_effect}"
            )
    return {
        "metadata": entry,
        "object": row,
        "artifact": artifact,
        "item": _catalog_item(entry, row, artifact),
    }


def _raw_inventory() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "agent": list_agents(),
        "skill": list_skills(),
        "knowledge": list_knowledge_docs(),
        "mcp_server": list_mcp_servers(),
        "mcp_tool": list_mcp_tools(),
        "system_tool": [],
    }


def list_trusted_capabilities() -> Dict[str, Any]:
    """Return the safe control-plane catalog; never expose untrusted rows."""

    raw = _raw_inventory()
    raw_counts = {kind: len(rows) for kind, rows in raw.items()}
    manifest_counts = Counter(
        entry["capability_type"] for entry in _MANIFEST_ENTRIES
    )
    trusted_counts = {kind: 0 for kind in CAPABILITY_TYPES}
    items: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for entry in _MANIFEST_ENTRIES:
        capability_type = str(entry["capability_type"])
        stable_id = entry["stable_id"]
        try:
            resolved = assert_trusted_capability(capability_type, stable_id)
        except CapabilityCatalogError as exc:
            errors.append(
                {
                    "code": "trusted_catalog_object_invalid",
                    "capability_type": capability_type,
                    "stable_id": stable_id,
                    "detail": str(exc),
                }
            )
            continue
        trusted_counts[capability_type] += 1
        items.append(resolved["item"])
    for kind in CAPABILITY_TYPES:
        trusted_counts.setdefault(kind, 0)
        raw_counts.setdefault(kind, 0)
    untrusted_counts = {
        kind: max(0, raw_counts[kind] - trusted_counts[kind])
        for kind in CAPABILITY_TYPES
    }
    items.sort(key=lambda item: (str(item["capability_type"]), str(item["stable_id"])))
    return {
        "catalog_version": CATALOG_VERSION,
        "raw_counts": dict(sorted(raw_counts.items())),
        "manifest_counts": {
            kind: int(manifest_counts.get(kind, 0))
            for kind in sorted(CAPABILITY_TYPES)
        },
        "trusted_counts": dict(sorted(trusted_counts.items())),
        "untrusted_counts": dict(sorted(untrusted_counts.items())),
        "items": items,
        "errors": errors,
        "system_tool_model": "not_present",
        "supply_chain_policy": (
            "new capabilities require code review, structural validation and "
            "deployment before catalog admission"
        ),
    }


def set_trusted_capability_enabled(
    capability_type: str,
    stable_id: Any,
    enabled: bool,
) -> Dict[str, Any]:
    """Change only Draft enablement for one already trusted identity."""

    normalized = _normalize_type(capability_type)
    resolved = assert_trusted_capability(normalized, stable_id)
    row = resolved["object"]
    target = bool(enabled)
    if normalized == "agent":
        if str(row.get("agent_id")) == "router" and not target:
            raise CapabilityCatalogError("the singleton Router cannot be disabled")
        update_agent(int(row["id"]), enabled=target)
    elif normalized == "skill":
        set_skill_enabled(int(row["id"]), target)
    elif normalized == "knowledge":
        if target and str(row.get("index_status") or "").lower() not in {
            "ready",
            "indexed",
        }:
            raise CapabilityCatalogError(
                f"knowledge capability is not index-ready: {stable_id}/"
                f"{row.get('index_status')!r}"
            )
        set_knowledge_doc_indexed_flag(int(row["id"]), target)
    elif normalized == "mcp_server":
        toggle_mcp_server_enabled(int(row["id"]), target)
    else:
        raise CapabilityCatalogError(
            f"{normalized} has no independent Draft enable switch; "
            "use its reviewed parent capability"
        )
    return assert_trusted_capability(normalized, stable_id)["item"]


def _unique_ints(values: Iterable[Any], field: str) -> List[int]:
    result: List[int] = []
    for value in values:
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise CapabilityCatalogError(f"{field} contains an invalid id: {value!r}") from exc
        if normalized not in result:
            result.append(normalized)
    return result


def _require_binding_domain(
    agent_scope: str,
    capability_type: str,
    stable_id: Any,
) -> Dict[str, Any]:
    resolved = assert_trusted_capability(capability_type, stable_id)
    capability_scope = str(resolved["metadata"]["domain_scope"])
    if agent_scope not in VERTICAL_DOMAINS or capability_scope != agent_scope:
        raise CapabilityCatalogError(
            f"cross-domain binding is forbidden: {agent_scope}/"
            f"{capability_type}/{stable_id}/{capability_scope}"
        )
    return resolved


def _mcp_ids_from_names(names: Iterable[str]) -> List[int]:
    requested = [str(item or "").strip() for item in names]
    if not requested:
        return []
    exact: Dict[str, int] = {}
    for stable_id in trusted_capability_ids("mcp_server"):
        try:
            resolved = assert_trusted_capability("mcp_server", stable_id)
        except CapabilityCatalogError:
            continue
        display_name = str(resolved["object"].get("name") or "")
        if not display_name or display_name in exact:
            raise CapabilityCatalogError(
                f"trusted MCP binding name is missing or duplicated: "
                f"{display_name!r}"
            )
        exact[display_name] = int(stable_id)
    result: List[int] = []
    for name in requested:
        if name not in exact:
            raise CapabilityCatalogError(
                f"MCP server name is not an exact trusted catalog identity: {name!r}"
            )
        if exact[name] not in result:
            result.append(exact[name])
    return result


def set_trusted_agent_bindings(
    agent_id: str,
    skill_ids: Sequence[int],
    knowledge_doc_ids: Sequence[int],
    mcp_server_ids: Optional[Sequence[int]] = None,
    mcp_server_names: Optional[Sequence[str]] = None,
    system_tool_ids: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Replace Draft bindings after validating every identity and domain."""

    agent = assert_trusted_capability("agent", agent_id)
    row = agent["object"]
    if str(row.get("category") or "") in {"router", "orchestration"}:
        raise CapabilityCatalogError("the singleton Router cannot own capabilities")
    agent_scope = str(agent["metadata"]["domain_scope"])
    normalized_skills = _unique_ints(skill_ids or [], "skill_ids")
    normalized_docs = _unique_ints(knowledge_doc_ids or [], "knowledge_doc_ids")
    normalized_servers = _unique_ints(mcp_server_ids or [], "mcp_server_ids")
    for server_id in _mcp_ids_from_names(mcp_server_names or []):
        if server_id not in normalized_servers:
            normalized_servers.append(server_id)
    if list(system_tool_ids or []):
        raise CapabilityCatalogError(
            "no independently cataloged system Tool exists in the current data model"
        )

    for skill_id in normalized_skills:
        _require_binding_domain(agent_scope, "skill", skill_id)
    for doc_id in normalized_docs:
        _require_binding_domain(agent_scope, "knowledge", doc_id)
    server_rows: List[Dict[str, Any]] = []
    for server_id in normalized_servers:
        server = _require_binding_domain(agent_scope, "mcp_server", server_id)
        trusted_tools = [
            entry
            for entry in _MANIFEST_ENTRIES
            if entry["capability_type"] == "mcp_tool"
            and int(entry.get("server_id") or 0) == server_id
        ]
        if not trusted_tools:
            raise CapabilityCatalogError(
                f"trusted MCP server has no reviewed read Tool: {server_id}"
            )
        for tool in trusted_tools:
            _require_binding_domain(agent_scope, "mcp_tool", tool["stable_id"])
            if tool.get("effect") != "read":
                raise CapabilityCatalogError(
                    f"MCP Tool is not read-only: {tool['stable_id']}"
                )
        server_rows.append(server["object"])

    # All validation completes before the first Draft mutation.
    set_agent_skills(str(row["agent_id"]), normalized_skills)
    set_agent_knowledge_bindings(str(row["agent_id"]), normalized_docs)
    set_agent_tools(
        str(row["agent_id"]),
        [{"tool_name": str(server.get("name") or "")} for server in server_rows],
    )
    return {
        "agent_id": str(row["agent_id"]),
        "domain_scope": agent_scope,
        "skill_ids": get_agent_skills(str(row["agent_id"])),
        "knowledge_doc_ids": get_agent_knowledge_bindings(str(row["agent_id"])) or [],
        "mcp_server_names": [
            str(item.get("tool_name") or "")
            for item in get_agent_tools(str(row["agent_id"]))
            if str(item.get("tool_name") or "")
        ],
        "system_tool_ids": [],
        "effective_on": "next_published_release_new_session",
        "current_release_unchanged": True,
    }
