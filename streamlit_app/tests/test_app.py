"""UI tests for the CargoMind Streamlit app (key journeys and error states).

Run from the repository root:

    .venv/bin/python -m unittest discover -s streamlit_app/tests -t streamlit_app/tests
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))  # same import root Streamlit uses for the app's own scripts

from components import ui  # noqa: E402

TIMEOUT = 180


class InterpretationRules(unittest.TestCase):
    def test_critical_with_anomaly_recommends_investigation_without_causal_claim(self):
        text = ui.interpret("Critical", True)
        self.assertIn("associated with unusual operating conditions", text)
        self.assertIn("An anomaly is not a delay prediction", text)
        self.assertNotIn("caused", text.lower())

    def test_non_critical_without_anomaly_reports_both_independently(self):
        text = ui.interpret("On-Time", False)
        self.assertIn("On-Time", text)
        self.assertIn("no unusual operating conditions flagged", text)

    def test_missing_anomaly_result_is_not_reported_as_normal(self):
        text = ui.interpret("At-Risk", None)
        self.assertIn("No operating-conditions result is available", text)
        self.assertNotIn("no unusual operating conditions", text)

    def test_outputs_are_never_merged_into_a_score(self):
        for cls in ("On-Time", "At-Risk", "Critical"):
            for flag in (True, False, None):
                text = ui.interpret(cls, flag).lower()
                self.assertNotIn("risk score", text)
                self.assertNotIn("combined", text)


class PageJourneys(unittest.TestCase):
    def _run(self, page):
        at = AppTest.from_file(str(APP_DIR / "pages" / page), default_timeout=TIMEOUT)
        at.run()
        return at

    def test_every_page_keeps_the_synthetic_status_pill_without_a_warning_banner(self):
        for page in ("1_ML_Intelligence.py", "2_Port_Forecast.py", "3_Methodology.py"):
            with self.subTest(page=page):
                at = self._run(page)
                self.assertFalse(at.exception, at.exception)
                markdown = " ".join(m.value for m in at.markdown)
                self.assertIn("Synthetic · offline", markdown)  # small header indicator stays
                self.assertNotIn("Synthetic, offline demo", markdown)  # the long banner is gone

    def test_risk_prediction_naming(self):
        at = self._run("1_ML_Intelligence.py")
        markdown = " ".join(m.value for m in at.markdown)
        self.assertIn("Risk Prediction", markdown)
        self.assertNotIn("ML Intelligence", markdown)

    def test_ml_page_asks_for_a_port_before_showing_results(self):
        at = self._run("1_ML_Intelligence.py")
        self.assertFalse(at.exception)
        self.assertTrue(any("Choose an origin port" in i.value for i in at.info))
        self.assertEqual(len(at.metric), 0)

    def test_ml_page_shows_both_outputs_separately_for_a_selected_shipment(self):
        at = self._run("1_ML_Intelligence.py")
        at.selectbox[0].set_value("SGSIN").run()
        first_id = at.selectbox[1].options[0].split(" · ")[0]  # options are display labels; select by id
        at.selectbox[1].set_value(first_id).run()
        self.assertFalse(at.exception, at.exception)
        markdown = " ".join(m.value for m in at.markdown)
        self.assertIn("Predicted class", markdown)
        self.assertIn("Anomaly flag", markdown)
        self.assertIn("Delay outlook", markdown)
        self.assertIn("Unusual conditions at origin", markdown)
        self.assertIn("Anomaly score", [m.label for m in at.metric])
        self.assertTrue(any("not calibrated" in c.value for c in at.caption))
        self.assertIn("Shipment", markdown)  # context bar names the selection near the results

    def test_ml_filters_narrow_the_shipment_list(self):
        at = self._run("1_ML_Intelligence.py")
        at.selectbox[0].set_value("SGSIN").run()
        before = len(at.selectbox[1].options)
        at.multiselect[0].select(at.multiselect[0].options[0]).run()
        self.assertFalse(at.exception, at.exception)
        self.assertLess(len(at.selectbox[1].options), before)

    def test_ml_batch_table_and_reset(self):
        at = self._run("1_ML_Intelligence.py")
        at.selectbox[0].set_value("SGSIN").run()
        at.toggle[0].set_value(True).run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len(at.dataframe), 1)
        columns = list(at.dataframe[0].value.columns)
        self.assertIn("Predicted class (XGBoost)", columns)
        self.assertIn("Unusual conditions (Isolation Forest)", columns)
        self.assertNotIn("Risk score", columns)
        at.button[0].click().run()
        self.assertIsNone(at.selectbox[0].value)

    def test_forecast_page_names_the_producing_model(self):
        at = self._run("2_Port_Forecast.py")
        if at.error:  # port history missing in this copy: covered by the prompt/error test below
            self.skipTest("port history unavailable")
        at.selectbox[0].set_value("SGSIN").run()
        self.assertFalse(at.exception, at.exception)
        markdown = " ".join(m.value for m in at.markdown)
        self.assertIn("Next-hour forecast", markdown)
        self.assertIn("Model that produced this forecast", markdown)
        self.assertTrue("Transformer T_w (primary)" in markdown or "C_w LSTM (automatic fallback)" in markdown)
        self.assertIn("Automatic fallback", markdown)
        self.assertIn("benchmark only", markdown)  # EWMA is labelled as a benchmark, not the fallback

    def test_forecast_page_asks_for_a_port_before_forecasting(self):
        at = self._run("2_Port_Forecast.py")
        self.assertFalse(at.exception)
        # Either the port list is available and the user is prompted, or the data error state is shown.
        messages = [i.value for i in at.info] + [e.value for e in at.error]
        self.assertTrue(any("Select a port" in m or "unavailable" in m for m in messages), messages)

    def test_overview_dashboard_is_labelled_as_a_synthetic_replay(self):
        # Separate process: AppTest keeps a per-process page registry, and st.page_link on the main page only resolves
        # when app.py is the first script run there (as with `streamlit run`).
        script = ("import json; from streamlit.testing.v1 import AppTest\n"
                  f"at = AppTest.from_file('app.py', default_timeout={TIMEOUT}); at.run()\n"
                  "print(json.dumps({'exc': [e.message for e in at.exception], 'err': [e.value for e in at.error],"
                  " 'md': ' '.join(m.value for m in at.markdown)}))")
        out = subprocess.run([sys.executable, "-c", script], cwd=APP_DIR, capture_output=True, text=True, timeout=TIMEOUT)
        result = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(result["exc"], [])
        if result["err"]:  # port history missing: the page must say so instead of showing numbers
            self.assertIn("unavailable", result["err"][0])
            return
        self.assertIn("Synthetic replay as of", result["md"])
        self.assertIn("not live data", result["md"])
        self.assertIn("Ports monitored", result["md"])

    def test_methodology_page_states_the_limits(self):
        at = self._run("3_Methodology.py")
        text = " ".join(m.value for m in at.markdown)
        self.assertIn("Synthetic, offline data only", text)
        self.assertIn("never merged", text)


if __name__ == "__main__":
    unittest.main()
