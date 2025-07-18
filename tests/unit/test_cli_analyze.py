from pathlib import Path
from unittest.mock import create_autospec, patch

from databricks.labs.blueprint.tui import MockPrompts
from databricks.labs.lakebridge import cli
from databricks.sdk import WorkspaceClient

from databricks.labs.lakebridge.contexts.application import ApplicationContext


from databricks.labs.bladespector.analyzer import Analyzer


def test_analyze():
    supported_tech = sorted(Analyzer.supported_source_technologies(), key=str.casefold)
    numbered_list = dict(enumerate(supported_tech))
    tech_enum = next((key for key, value in numbered_list.items() if value == "Informatica - PC"), 12)

    prompts = MockPrompts(
        {
            r"Select the source technology": tech_enum,
        }
    )
    with patch.object(ApplicationContext, "prompts", prompts):
        ws = create_autospec(WorkspaceClient)
        input_path = str(Path(__file__).parent.parent / "resources" / "functional" / "informatica")
        cli.analyze(ws, input_path, "/tmp/sample.xlsx")


def test_analyze_source_override():
    ws = create_autospec(WorkspaceClient)
    input_path = str(Path(__file__).parent.parent / "resources" / "functional" / "informatica")
    cli.analyze(ws, input_path, "/tmp/sample.xlsx", "Informatica - PC")
