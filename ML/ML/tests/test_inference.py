"""Tests for ML inference (cargomind_ml.inference): XGBoost classifier, Isolation Forest detector, feature construction.

Saved models, results and ML/Dataset are only read. Run from the project root:

    .venv/bin/python -m unittest discover -s ML/ML/tests
"""
import json
import unittest
import warnings

import numpy as np
import pandas as pd

from cargomind_eda import io as eda_io
from cargomind_ml import config, inference as mli
from cargomind_ml.features import ANOMALY_FEATURES, CATEGORICAL, FEATURES, NUMERIC
from cargomind_ml.xgboost_model import BalancedXGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

TABLE = pd.read_pickle(config.RESULT_DIRS["data"] / "modelling_table.pkl")
TEST_PRED = pd.read_csv(config.RESULT_DIRS["metrics"] / "test_predictions.csv")
TEST_SCORES = pd.read_csv(config.RESULT_DIRS["anomaly"] / "test_anomaly_scores.csv")
RAW_SHIPMENTS = eda_io.load("shipment_records")


def stored_rows(ids):
    return TABLE.set_index("shipment_id").loc[list(ids)].reset_index()


def new_shipment(**overrides):
    """A made-up shipment (not in the dataset) on route R..., departing inside the data's coverage."""
    net = eda_io.load("port_network")
    r = net[(net["origin_port_id"] == "NLRTM") & (net["destination_port_id"] == "SGSIN")].iloc[0]
    row = {"shipment_id": "NEW000001", "route_id": r["route_id"], "origin_port_id": "NLRTM", "destination_port_id": "SGSIN",
           "carrier": "Coral Bay Logistics", "shipping_mode": "sea", "cargo_type": "general", "container_teu": 2.0,
           "shipment_priority": "medium", "departure_ts": "2025-02-05 10:00:00+00:00",
           "expected_arrival_ts": pd.Timestamp("2025-02-05 10:00:00+00:00") + pd.Timedelta(hours=float(r["planned_transit_hours"]))}
    row.update(overrides)
    return pd.DataFrame([row])


class Selection(unittest.TestCase):
    def test_selection_file_names_the_production_models(self):
        sel = mli.load_selection()
        self.assertEqual((sel["classifier"]["name"], sel["anomaly_detector"]["name"]), ("xgboost", "isolation_forest"))
        self.assertEqual((mli.PRIMARY_CLASSIFIER, mli.ANOMALY_DETECTOR), ("xgboost", "isolation_forest"))

    def test_research_artifacts_are_kept_and_loadable(self):
        for name in mli.RESEARCH_CLASSIFIERS:
            self.assertEqual(mli.load_classifier(name).name, name)
        self.assertEqual(mli.load_detector("one_class_svm").name, "one_class_svm")
        self.assertTrue((config.MODELS / "dummy_prior.joblib").exists())


class XGBoostClassifier(unittest.TestCase):
    def test_loads_with_the_boosting_preprocessing_and_feature_schema(self):
        clf = mli.load_classifier()
        self.assertEqual(clf.name, "xgboost")
        self.assertIsInstance(clf.pipeline.named_steps["model"], BalancedXGBClassifier)
        self.assertEqual(clf.config["preprocessing"], "boosting")
        tr = {name: (step, cols) for name, step, cols in clf.pipeline.named_steps["prep"].transformers}  # as configured
        self.assertEqual(tr["num"][0], "passthrough")  # boosting family: native NaN handling, no scaling
        self.assertEqual(list(tr["num"][1]), NUMERIC)
        self.assertEqual(list(tr["cat"][1]), CATEGORICAL)
        self.assertEqual(clf.config["features"], FEATURES)
        self.assertIn("xgboost.joblib@", clf.version)

    def test_reproduces_the_stored_test_predictions(self):
        res = mli.predict(stored_rows(TEST_PRED["shipment_id"]), anomaly=False)
        p = np.array([[r["classification"]["class_probabilities"][c] for c in config.CLASSES] for r in res])
        stored = TEST_PRED[[f"xgboost_p_{c}" for c in config.CLASSES]].to_numpy()
        np.testing.assert_allclose(p, stored, atol=1e-6)
        self.assertEqual([r["classification"]["predicted_class"] for r in res], TEST_PRED["xgboost_pred"].tolist())

    def test_three_classes_and_valid_probabilities(self):
        r = mli.predict(stored_rows(TEST_PRED["shipment_id"].iloc[:20]))
        for x in r:
            probs = x["classification"]["class_probabilities"]
            self.assertEqual(list(probs), config.CLASSES)
            self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)
            self.assertTrue(all(0 <= v <= 1 for v in probs.values()))
            self.assertEqual(x["classification"]["predicted_class"], max(probs, key=probs.get))


class IsolationForestDetector(unittest.TestCase):
    def test_loads_with_threshold_matching_its_config(self):
        det = mli.load_detector()
        self.assertEqual(det.name, "isolation_forest")
        self.assertEqual(det.threshold, json.loads((config.MODELS / "isolation_forest_config.json").read_text())["threshold"])
        self.assertEqual(det.config["features"], ANOMALY_FEATURES)

    def test_scores_and_flags_reproduce_the_stored_test_output_and_threshold_rule(self):
        res = mli.predict(stored_rows(TEST_SCORES["shipment_id"]))
        s = np.array([r["anomaly"]["anomaly_score"] for r in res])
        f = np.array([r["anomaly"]["anomaly_flag"] for r in res])
        np.testing.assert_allclose(s, TEST_SCORES["isolation_forest_score"], atol=1e-9)
        np.testing.assert_array_equal(f, TEST_SCORES["isolation_forest_flag"].to_numpy())
        np.testing.assert_array_equal(f, s > mli.load_detector().threshold)
        self.assertTrue(0 < f.mean() < 0.2)


class OutputContract(unittest.TestCase):
    def test_classification_and_anomaly_are_separate_sections(self):
        r = mli.predict(stored_rows(TEST_PRED["shipment_id"].iloc[:3]))[0]
        self.assertEqual(set(r), {"contract_version", "shipment_id", "prediction_timestamp_utc", "feature_cutoff_utc",
                                  "classification", "anomaly", "warnings"})
        self.assertEqual(r["contract_version"], mli.CONTRACT_VERSION)
        self.assertEqual(r["shipment_id"], TEST_PRED["shipment_id"].iloc[0])
        self.assertEqual(set(r["classification"]), {"model_name", "model_label", "model_version", "predicted_class",
                                                    "class_probabilities", "decision_rule", "probabilities_note"})
        self.assertEqual(set(r["anomaly"]), {"model_name", "model_label", "model_version", "anomaly_score", "anomaly_flag",
                                             "threshold", "score_definition", "threshold_rule"})
        self.assertNotIn("anomaly_flag", r["classification"])
        self.assertNotIn("predicted_class", r["anomaly"])
        json.dumps(r)

    def test_anomaly_section_can_be_switched_off_without_changing_classification(self):
        rows = stored_rows(TEST_PRED["shipment_id"].iloc[:5])
        a, b = mli.predict(rows), mli.predict(rows, anomaly=False)
        self.assertTrue(all(x["anomaly"] is None for x in b))
        self.assertEqual([x["classification"]["class_probabilities"] for x in a], [x["classification"]["class_probabilities"] for x in b])

    def test_shipment_id_and_cutoff_are_optional(self):
        r = mli.predict(stored_rows(TEST_PRED["shipment_id"].iloc[:2])[FEATURES])
        self.assertIsNone(r[0]["shipment_id"])
        self.assertIsNone(r[0]["feature_cutoff_utc"])


class InvalidFeatures(unittest.TestCase):
    def setUp(self):
        self.rows = stored_rows(TEST_PRED["shipment_id"].iloc[:4])

    def assertRejects(self, x, contains):
        with self.assertRaises(mli.FeatureValidationError) as cm:
            mli.predict(x)
        self.assertIn(contains, str(cm.exception))

    def test_missing_column_non_frame_and_empty(self):
        self.assertRejects(self.rows.drop(columns=["o_anchor_mean24"]), "missing feature columns")
        self.assertRejects(self.rows[FEATURES].to_numpy(), "DataFrame")
        self.assertRejects(self.rows.iloc[:0], "no rows")

    def test_non_numeric_infinite_and_missing_category(self):
        x = self.rows.copy()
        x["o_anchor_mean24"] = x["o_anchor_mean24"].astype(object)
        x.loc[0, "o_anchor_mean24"] = "high"
        self.assertRejects(x, "must be numeric")
        x = self.rows.copy()
        x.loc[1, "wx_o_gust_max24"] = np.inf
        self.assertRejects(x, "infinite")
        x = self.rows.copy()
        x.loc[2, "carrier"] = None
        self.assertRejects(x, "missing values")

    def test_first_day_departures_report_missing_24h_changes(self):
        first = TABLE[TABLE["departure_ts"] == TABLE["departure_ts"].min()].head(1)
        self.assertIn("o_util_change24", mli.predict(first)[0]["warnings"][0])

    def test_unseen_category_and_missing_inputs_are_reported(self):
        x = self.rows.copy()
        x.loc[0, "carrier"] = "Unknown Line"
        x.loc[1, "o_container_dwell_hours"] = np.nan
        r = mli.predict(x)
        self.assertTrue(any("not seen in training" in w for w in r[0]["warnings"]))
        self.assertTrue(any("missing input values ['o_container_dwell_hours']" in w for w in r[1]["warnings"]))
        self.assertEqual(r[2]["warnings"], [])

    def test_structural_missing_values_do_not_warn(self):
        direct = stored_rows(TEST_PRED["shipment_id"]).query("route_type == 'direct'").head(3)  # test period: 24 h history exists
        self.assertTrue(direct["h_util_mean24"].isna().all())
        self.assertTrue(all(not r["warnings"] for r in mli.predict(direct)))


class FeatureConstruction(unittest.TestCase):
    def test_features_equal_the_training_table_for_dataset_shipments(self):
        rng = np.random.default_rng(1)
        for sid in rng.choice(TABLE["shipment_id"], 6, replace=False):
            ship = RAW_SHIPMENTS[RAW_SHIPMENTS["shipment_id"] == sid][mli.SHIPMENT_INPUT_COLUMNS]
            got = mli.build_features_for_shipments(ship)[FEATURES].reset_index(drop=True)
            pd.testing.assert_frame_equal(got, stored_rows([sid])[FEATURES], check_dtype=False, atol=1e-9)

    def test_new_shipment_end_to_end(self):
        res = mli.predict_shipments(new_shipment())
        self.assertEqual(len(res), 1)
        r = res[0]
        self.assertEqual(r["shipment_id"], "NEW000001")
        self.assertEqual(r["feature_cutoff_utc"], "2025-02-05T09:00:00+00:00")
        self.assertIn(r["classification"]["predicted_class"], config.CLASSES)
        self.assertIsInstance(r["anomaly"]["anomaly_flag"], bool)

    def test_rejected_shipment_inputs(self):
        cases = [
            (new_shipment().assign(actual_delay_hours=3.0), "outcome columns"),
            (new_shipment().drop(columns=["carrier"]), "missing shipment columns"),
            (new_shipment(departure_ts="2025-03-20 10:00:00+00:00", expected_arrival_ts="2025-04-01 00:00:00+00:00"), "no port data"),
            (new_shipment(departure_ts="2025-02-05 10:30:00+00:00"), "on the hour"),
            (new_shipment(destination_port_id="USLAX"), "disagree with route_id"),
            (new_shipment(route_id="R999"), "unknown route_id"),
            (new_shipment(shipment_priority="urgent"), "shipment_priority"),
            (new_shipment(expected_arrival_ts="2025-02-04 00:00:00+00:00"), "after departure_ts"),
            (new_shipment(container_teu=-1), "positive"),
        ]
        for df, msg in cases:
            with self.assertRaises(mli.ShipmentInputError, msg=msg) as cm:
                mli.build_features_for_shipments(df)
            self.assertIn(msg, str(cm.exception))

    def test_dataset_files_are_not_modified(self):
        before = {p: p.stat().st_mtime_ns for p in eda_io.PROCESSED.glob("*.csv")}
        mli.predict_shipments(new_shipment())
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in eda_io.PROCESSED.glob("*.csv")})


class ArtifactIntegrity(unittest.TestCase):
    def test_unknown_models_are_refused(self):
        with self.assertRaises(mli.ModelArtifactError):
            mli.load_classifier("ensemble")
        with self.assertRaises(mli.ModelArtifactError):
            mli.load_detector("xgboost")


if __name__ == "__main__":
    unittest.main()
