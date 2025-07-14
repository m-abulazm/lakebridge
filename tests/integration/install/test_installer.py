import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from databricks.labs.lakebridge.install import TranspilerInstaller, MavenInstaller, WorkspaceInstaller, WheelInstaller


@pytest.mark.skipif(os.environ.get("CI", "false") == "true", reason="Skipping in CI since we have no installed product")
def test_gets_installed_remorph_version(patched_transpiler_installer):
    version = patched_transpiler_installer.get_installed_version("remorph", False)
    check_valid_version(version)


def test_gets_maven_artifact_version() -> None:
    version = MavenInstaller.get_current_maven_artifact_version("com.databricks", "databricks-connect")
    assert version
    check_valid_version(version)


def test_downloads_from_maven():
    with TemporaryDirectory() as parent:
        path = Path(parent) / "pom.xml"
        success = MavenInstaller.download_artifact_from_maven(
            "com.databricks", "databricks-connect", "16.0.0", path, extension="pom"
        )
        assert success
        assert path.exists()
        assert path.stat().st_size == 5_684


def test_gets_pypi_artifact_version():
    version = WheelInstaller.get_latest_artifact_version_from_pypi("databricks-labs-remorph")
    check_valid_version(version)


@pytest.fixture()
def patched_transpiler_installer(tmp_path: Path):
    resources_folder = Path(__file__).parent.parent.parent / "resources" / "transpiler_configs"
    for transpiler in ("bladebridge", "morpheus"):
        target = tmp_path / transpiler
        target.mkdir(exist_ok=True)
        target = target / "lib"
        target.mkdir(exist_ok=True)
        target = target / "config.yml"
        source = resources_folder / transpiler / "lib" / "config.yml"
        shutil.copyfile(source, target)
    with patch.object(TranspilerInstaller, "transpilers_path", return_value=tmp_path):
        yield TranspilerInstaller


def test_lists_all_transpiler_names(patched_transpiler_installer):
    transpiler_names = patched_transpiler_installer.all_transpiler_names()
    assert transpiler_names == {'Morpheus', 'Bladebridge'}


def test_lists_all_dialects(patched_transpiler_installer):
    dialects = patched_transpiler_installer.all_dialects()
    assert dialects == {
        'athena',
        'bigquery',
        'datastage',
        'greenplum',
        'informatica (desktop edition)',
        'mssql',
        'netezza',
        'oracle',
        'redshift',
        'snowflake',
        'synapse',
        'teradata',
        'tsql',
    }


def test_lists_dialect_transpilers(patched_transpiler_installer):
    transpilers = patched_transpiler_installer.transpilers_with_dialect("snowflake")
    assert transpilers == {'Morpheus', 'Bladebridge'}
    transpilers = patched_transpiler_installer.transpilers_with_dialect("datastage")
    assert transpilers == {'Bladebridge'}


def check_valid_version(version: str):
    parts = version.split(".")
    for _, part in enumerate(parts):
        try:
            _ = int(part)
        except ValueError:
            assert False, f"{version} does not look like a valid semver"


def test_java_version() -> None:
    result = WorkspaceInstaller.find_java()
    match result:
        case None:
            # Fine, no Java available.
            pass
        case (java_home, tuple(version)):
            assert java_home.exists() and version >= (11, 0, 0, 0)
        case _:
            pytest.fail(f"Unexpected result from WorkspaceInstaller.find_java(): {result!r}")
