from pathlib import Path
from unittest.mock import patch

from databricks.sdk import WorkspaceClient

from databricks.labs.lakebridge.config import TranspileConfig
from databricks.labs.lakebridge.install import TranspilerInstaller
from databricks.labs.lakebridge.transpiler.execute import transpile
from databricks.labs.lakebridge.transpiler.lsp.lsp_engine import LSPEngine


async def test_transpiles_all_dbt_project_files(ws: WorkspaceClient, tmp_path: Path) -> None:
    labs_path = tmp_path / "labs"
    output_folder = tmp_path / "output"
    with patch.object(TranspilerInstaller, "labs_path", return_value=labs_path):
        await _transpile_all_dbt_project_files(ws, output_folder)


async def _transpile_all_dbt_project_files(ws: WorkspaceClient, output_folder: Path) -> None:
    morpheus = TranspilerInstaller.transpilers_path() / "morpheus"
    assert not morpheus.exists()
    TranspilerInstaller.install_from_maven("morpheus", "com.databricks.labs", "databricks-morph-plugin")
    # check execution
    config_path = morpheus / "lib" / "config.yml"
    lsp_engine = LSPEngine.from_config_path(config_path)
    input_source = Path(__file__).parent.parent.parent / "resources" / "functional" / "dbt"

    transpile_config = TranspileConfig(
        transpiler_config_path=str(config_path),
        source_dialect="snowflake",
        input_source=str(input_source),
        output_folder=str(output_folder),
        skip_validation=False,
        catalog_name="catalog",
        schema_name="schema",
    )
    await transpile(ws, lsp_engine, transpile_config)
    assert (output_folder / "top-query.sql").exists()
    assert (output_folder / "dbt_project.yml").exists()
    assert (output_folder / "sub" / "sub-query.sql").exists()
    assert (output_folder / "sub" / "sub-query-bom.sql").exists()
    assert (output_folder / "sub" / "dbt_project.yml").exists()
