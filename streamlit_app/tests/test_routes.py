"""Tests for the Alternative Route Advisor (services/routes.py and pages/4_Route_Advisor.py).

Run from the repository root:

    .venv/bin/python -m unittest discover -s streamlit_app/tests -t streamlit_app/tests
"""
import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from services import routes  # noqa: E402

HOUR = "2025-02-11 15:00+00:00"  # three strikes are in progress (MYPKG, INNSA, NLRTM) in the synthetic history


class RouteService(unittest.TestCase):
    def test_alternatives_use_only_direct_service_edges_without_repeats(self):
        g = routes.graph()
        for o in g.nodes:
            for d in g.nodes:
                if o == d:
                    continue
                advice = routes.advise(o, d, HOUR)
                for r in advice.alternatives:
                    self.assertEqual(len(r.ports), len(set(r.ports)), r.ports)  # no cycles or repeated ports
                    self.assertLessEqual(len(r.ports) - 1, routes.MAX_LEGS)
                    self.assertTrue(all(g.has_edge(a, b) for a, b in zip(r.ports, r.ports[1:])), r.ports)
                    self.assertFalse(set(r.ports[1:-1]) & set(advice.disrupted), r.ports)  # disrupted stopovers avoided

    def test_best_alternative_is_ranked_first_and_labelled(self):
        advice = routes.advise("INMAA", "NLRTM", HOUR)
        self.assertTrue(advice.alternatives)
        self.assertEqual(advice.alternatives[0].label, "Best alternative")
        scores = [r.score for r in advice.alternatives]
        self.assertEqual(scores, sorted(scores))
        self.assertIn("NLRTM", advice.disrupted)  # destination strike is reported, not avoided
        self.assertTrue(advice.endpoint_notes)

    def test_no_alternative_returns_an_explanation(self):
        advice = routes.advise("NLRTM", "AEJEA", HOUR, scenario_port="DEHAM")
        self.assertEqual(advice.alternatives, [])
        self.assertIn("No alternative path", advice.message)
        self.assertEqual(advice.scheduled.ports, ["NLRTM", "MYPKG", "AEJEA"])

    def test_same_origin_and_destination(self):
        advice = routes.advise("SGSIN", "SGSIN", HOUR)
        self.assertIsNone(advice.scheduled)
        self.assertEqual(advice.alternatives, [])
        self.assertIn("same port", advice.message)


class RoutePage(unittest.TestCase):
    def test_page_finds_routes(self):
        at = AppTest.from_file(str(APP_DIR / "pages" / "4_Route_Advisor.py"), default_timeout=180)
        at.run()
        self.assertFalse(at.exception, at.exception)
        at.button[0].click().run()
        self.assertFalse(at.exception, at.exception)
        text = " ".join(m.value for m in at.markdown)
        self.assertIn("Route comparison", text)
        self.assertIn("Scheduled route", text)
        self.assertTrue(any("Prototype recommendation" in c.value for c in at.caption))


if __name__ == "__main__":
    unittest.main()
