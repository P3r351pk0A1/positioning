import csv
import json

import pytest

from positioning.cli import main
from positioning.pmi import run_pmi


def test_pmi_runner_writes_protocols(tmp_path):
    results = run_pmi(output_dir=tmp_path, repeat_count=10)

    assert all(item.passed for item in results)
    assert len(results) == 15
    assert (tmp_path / "pmi_results.json").exists()
    assert (tmp_path / "pmi_results.csv").exists()
    assert (tmp_path / "ПРОТОКОЛ_ПМИ.md").exists()

    payload = json.loads((tmp_path / "pmi_results.json").read_text(encoding="utf-8"))
    assert payload["summary"]["all_passed"]

    rows = list(csv.DictReader((tmp_path / "pmi_results.csv").open("r", encoding="utf-8")))
    assert len(rows) == 15


def test_pmi_cli_command_is_removed(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["pmi"])
    stderr = capsys.readouterr().err

    assert exc.value.code == 2
    assert "invalid choice" in stderr
    assert "pmi" in stderr
