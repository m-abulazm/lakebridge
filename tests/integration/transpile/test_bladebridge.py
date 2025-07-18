import logging
from pathlib import Path
from unittest.mock import patch


from databricks.sdk import WorkspaceClient

from databricks.labs.lakebridge.config import TranspileConfig
from databricks.labs.lakebridge.install import TranspilerInstaller
from databricks.labs.lakebridge.transpiler.execute import transpile
from databricks.labs.lakebridge.transpiler.lsp.lsp_engine import LSPEngine

logger = logging.getLogger(__name__)


async def test_transpiles_informatica_with_sparksql(
    ws: WorkspaceClient, bladebridge_artifact: Path, tmp_path: Path
) -> None:
    labs_path = tmp_path / "labs"
    output_folder = tmp_path / "output"
    with patch.object(TranspilerInstaller, "labs_path", return_value=labs_path):
        await _transpile_informatica_with_sparksql(ws, bladebridge_artifact, output_folder)


async def _transpile_informatica_with_sparksql(
    ws: WorkspaceClient, bladebridge_artifact: Path, output_folder: Path
) -> None:
    bladebridge = TranspilerInstaller.transpilers_path() / "bladebridge"
    assert not bladebridge.exists()
    TranspilerInstaller.install_from_pypi("bladebridge", "databricks-bb-plugin", bladebridge_artifact)
    # check execution
    config_path = TranspilerInstaller.transpilers_path() / "bladebridge" / "lib" / "config.yml"
    lsp_engine = LSPEngine.from_config_path(config_path)
    input_source = Path(__file__).parent.parent.parent / "resources" / "functional" / "informatica"
    transpile_config = TranspileConfig(
        transpiler_config_path=str(config_path),
        source_dialect="informatica (desktop edition)",
        input_source=str(input_source),
        output_folder=str(output_folder),
        skip_validation=True,
        catalog_name="catalog",
        schema_name="schema",
        transpiler_options={"target-tech": "SPARKSQL"},
    )
    await transpile(ws, lsp_engine, transpile_config)
    # TODO: This seems to be flaky; debug logging to help diagnose the flakiness.
    files = [f.name for f in output_folder.iterdir()]
    logger.debug(f"Transpiled files: {files}")
    assert (output_folder / "m_employees_load.py").exists()
    assert (output_folder / "wf_m_employees_load.json").exists()
    assert (output_folder / "wf_m_employees_load_params.py").exists()
