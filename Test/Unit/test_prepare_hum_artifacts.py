import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

import prepare_hum_artifacts


def _column_index(cell_reference):
    letters = "".join(char for char in cell_reference if char.isalpha())
    index = 0
    for char in letters:
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    return index - 1


def _load_shared_strings(archive):
    path = "xl/sharedStrings.xml"
    if path not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read(path))
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    values = []
    for item in root.findall("x:si", namespace):
        parts = [node.text or "" for node in item.findall(".//x:t", namespace)]
        values.append("".join(parts))
    return values


def _cell_value(cell, shared_strings):
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    cell_type = cell.attrib.get("t")
    value = cell.find("x:v", namespace)
    if cell_type == "s":
        return shared_strings[int(value.text)]
    if value is None:
        inline = cell.find(".//x:t", namespace)
        return inline.text if inline is not None else None
    text = value.text
    if text is None:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def _read_xlsx_sheet(path):
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        sheet_names = [sheet.attrib["name"] for sheet in workbook.findall(".//x:sheet", namespace)]
        shared_strings = _load_shared_strings(archive)
        sheet_root = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))

        rows = []
        for row in sheet_root.findall(".//x:sheetData/x:row", namespace):
            values = []
            for cell in row.findall("x:c", namespace):
                column = _column_index(cell.attrib["r"])
                while len(values) <= column:
                    values.append(None)
                values[column] = _cell_value(cell, shared_strings)
            rows.append(values)

    return sheet_names, rows


@unittest.skipIf(importlib.util.find_spec("pyarrow") is None, "pyarrow is required for parquet tests")
@unittest.skipIf(
    importlib.util.find_spec("xlsxwriter") is None and importlib.util.find_spec("openpyxl") is None,
    "xlsxwriter or openpyxl is required for xlsx output",
)
class TestPrepareHumArtifacts(unittest.TestCase):
    def test_prepare_hum_artifacts_from_synthetic_parquet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            parquet_path = tmp_path / "hum_subject.parquet"
            output_dir = tmp_path / "artifacts"

            frame = pd.DataFrame(
                {
                    "date": pd.date_range("2024-01-01 00:00:00", periods=5, freq="5min"),
                    "id": ["subject-7"] * 5,
                    "CGM": [100.0, np.nan, 120.0, np.nan, 140.0],
                    "carbs": [0.0, 30.0, 0.0, 0.0, 15.0],
                    "insulin": [0.0, 0.1, 0.0, 0.2, 0.0],
                    "basal": [0.0, 0.1, 0.0, 0.0, 0.0],
                    "bolus": [0.0, 0.0, 0.0, 0.2, 0.0],
                    "meal_label": [None, "breakfast", None, None, "snack"],
                }
            )
            frame.to_parquet(parquet_path)

            prepare_hum_artifacts.main(
                [
                    str(parquet_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertTrue((output_dir / "model_state_results.xlsx").exists())
            self.assertTrue((output_dir / "insulin_input.csv").exists())
            self.assertTrue((output_dir / "iob.csv").exists())
            self.assertTrue((output_dir / "simulation_settings.json").exists())

            insulin = pd.read_csv(output_dir / "insulin_input.csv")
            self.assertEqual(len(insulin), 5)
            self.assertEqual(insulin.columns.tolist(), ["0"])
            self.assertEqual(insulin["0"].round(3).tolist(), [0.0, 20.0, 0.0, 40.0, 0.0])

            iob = pd.read_csv(output_dir / "iob.csv")
            self.assertEqual(len(iob), 5)
            self.assertTrue((iob["0"] == 0.0).all())

            with (output_dir / "simulation_settings.json").open() as handle:
                settings = json.load(handle)
            self.assertEqual(settings["settings"]["sampling_time"], 5.0)
            self.assertEqual(settings["settings"]["end_time"], 25.0)
            self.assertEqual(settings["patient"]["number_of_subjects"], 1)
            self.assertEqual(len(settings["inputs"]["meal_carb"]["magnitude"][0]), 2)
            self.assertEqual(len(settings["inputs"]["basal_insulin"]["magnitude"][0]), 1)
            self.assertEqual(len(settings["inputs"]["bolus_insulin"]["magnitude"][0]), 1)
            self.assertEqual(settings["settings"]["random_state"]["hum_subject_id"], "subject-7")

            jsonl_path = (
                output_dir
                / "PredictionSource"
                / "SimulationData"
                / "normal_day"
                / "normal_only_0_real_data.jsonl"
            )
            with jsonl_path.open() as handle:
                record = json.loads(handle.readline())

            self.assertEqual(record["patient_id"], "Patient_0")
            self.assertEqual(record["hum_subject_id"], "subject-7")
            self.assertEqual(record["sampling_minutes"], 5.0)
            self.assertEqual(record["cgm_coverage"], {"observed": 3, "total": 5, "fraction": 0.6})
            self.assertEqual(record["bg_mgdl"], [100.0, 110.0, 120.0, 130.0, 140.0])
            self.assertEqual(record["bg_mgdl_raw"], [100.0, None, 120.0, None, 140.0])
            self.assertEqual(
                record["faults_label"],
                ["None", "missing_signal", "None", "missing_signal", "None"],
            )
            self.assertEqual(len(record["carb_events"]), 2)
            self.assertEqual(record["carb_events"][0]["meal_type"], "breakfast")
            self.assertEqual(record["insulin_mUmin"]["magnitude"], [0.0, 20.0, 0.0, 40.0, 0.0])

            sheet_names, rows = _read_xlsx_sheet(output_dir / "model_state_results.xlsx")
            self.assertIn("Patient_0", sheet_names)
            header = rows[0]
            self.assertEqual(header[1:], ["IG (mmol/L)", "faults_label"])
            faults = [row[2] for row in rows[1:]]
            self.assertEqual(faults, ["None", "missing_signal", "None", "missing_signal", "None"])
            self.assertAlmostEqual(rows[1][1], 100.0 / 18.0)
            self.assertIsNone(rows[2][1])


if __name__ == "__main__":
    unittest.main()
