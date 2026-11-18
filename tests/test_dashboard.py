"""Automated regression tests for RetainAI's Streamlit dashboard."""

from __future__ import annotations

import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app


class DashboardModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.telco, cls.customers, cls.segments, cls.uplift_metrics = app.load_data()
        (
            cls.churn_model,
            cls.feature_columns,
            cls.cox_model,
            cls.kaplan_meier,
        ) = app.load_models()

    def test_churn_probabilities_are_valid_and_match_saved_scoring(self) -> None:
        sample = self.telco.iloc[:64].copy()
        features = app.build_churn_matrix(sample, self.feature_columns)
        probabilities = self.churn_model.predict_proba(features)[:, 1]

        self.assertEqual(features.shape, (64, len(self.feature_columns)))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertTrue(np.array_equal(self.churn_model.classes_, np.array([0, 1])))

        saved_scores = app.score_customers().set_index("customerID")
        expected = saved_scores.loc[sample["customerID"], "ChurnRisk"].to_numpy()
        np.testing.assert_allclose(probabilities, expected, rtol=0, atol=1e-12)

    def test_shap_explanations_reconstruct_model_raw_scores(self) -> None:
        features = app.build_churn_matrix(self.telco.iloc[:8], self.feature_columns)
        explainer = shap.TreeExplainer(self.churn_model)
        values = explainer.shap_values(features)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]

        raw_scores = np.asarray(self.churn_model.predict(features, raw_score=True))
        reconstructed = values.sum(axis=1) + float(np.asarray(explainer.expected_value).reshape(-1)[-1])
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(reconstructed, raw_scores, rtol=1e-5, atol=1e-5)

    def test_conditional_survival_probabilities_are_bounded_monotone_and_horizon_limited(
        self,
    ) -> None:
        customer = self.customers.loc[self.customers["tenure"] == 70].iloc[0]
        survival, current_tenure, last_observed = app.predict_conditional_survival(
            customer, self.cox_model
        )

        self.assertEqual(current_tenure, 70)
        self.assertEqual(last_observed, 72)
        self.assertEqual(len(survival), 2)
        self.assertTrue(np.isfinite(survival.to_numpy()).all())
        self.assertTrue(((survival >= 0) & (survival <= 1)).all())
        self.assertTrue((np.diff(survival.to_numpy()) <= 1e-12).all())

        beyond_horizon = customer.copy()
        beyond_horizon["tenure"] = last_observed
        no_extrapolation, _, _ = app.predict_conditional_survival(
            beyond_horizon, self.cox_model
        )
        self.assertTrue(no_extrapolation.empty)

    def test_saved_clv_matches_monthly_revenue_margin_formula(self) -> None:
        clv_data = pd.read_csv(
            PROJECT_ROOT / "data" / "processed" / "customer_clv.csv"
        )
        expected_clv = app.calculate_customer_clv(
            clv_data["MonthlyCharges"],
            clv_data["ExpectedRemainingMonths"],
            profit_margin=0.40,
        )
        np.testing.assert_allclose(
            expected_clv,
            clv_data["CLV"],
            rtol=0,
            atol=0.01,
        )

    def test_saved_survival_models_have_valid_survival_probabilities(self) -> None:
        km_values = self.kaplan_meier.survival_function_.iloc[:, 0].to_numpy()
        self.assertTrue(np.isfinite(km_values).all())
        self.assertTrue(((km_values >= 0) & (km_values <= 1)).all())
        self.assertTrue((np.diff(km_values) <= 1e-12).all())
        self.assertEqual(int(self.cox_model.baseline_survival_.index.max()), 72)

    def test_all_pages_render_and_sidebar_filters_apply(self) -> None:
        test_app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=120).run()
        self.assertFalse(test_app.exception, [e.message for e in test_app.exception])
        self.assertEqual(
            [metric.label for metric in test_app.metric],
            [
                "Customers in view",
                "Observed churn rate",
                "Active, high-risk",
                "Expected value at risk",
            ],
        )

        for page in app.PAGES[1:]:
            with self.subTest(page=page):
                test_app.sidebar.radio[0].set_value(page).run()
                self.assertFalse(test_app.exception, [e.message for e in test_app.exception])

        test_app.sidebar.radio[0].set_value("Customer Overview").run()
        total_customers = int(test_app.metric[0].value.replace(",", ""))
        test_app.selectbox[0].set_value("Female").run()
        self.assertFalse(test_app.exception, [e.message for e in test_app.exception])
        female_count = int(test_app.metric[0].value.replace(",", ""))
        self.assertGreater(female_count, 0)
        self.assertLess(female_count, total_customers)

        test_app.selectbox[1].set_value("Month-to-month").run()
        test_app.selectbox[2].set_value("Fiber optic").run()
        test_app.selectbox[3].set_value("High-Value Customers").run()
        test_app.slider[0].set_range(0, 12).run()
        self.assertFalse(test_app.exception, [e.message for e in test_app.exception])
        expected_filtered_count = self.customers.loc[
            (self.customers["gender"] == "Female")
            & (self.customers["Contract"] == "Month-to-month")
            & (self.customers["InternetService"] == "Fiber optic")
            & (self.customers["SegmentName"] == "High-Value Customers")
            & self.customers["tenure"].between(0, 12)
        ].shape[0]
        self.assertEqual(
            int(test_app.metric[0].value.replace(",", "")),
            expected_filtered_count,
        )

        test_app.multiselect[0].set_value([]).run()
        self.assertFalse(test_app.exception, [e.message for e in test_app.exception])
        self.assertTrue(test_app.info)

        test_app.multiselect[0].set_value(["Active"]).run()
        self.assertFalse(test_app.exception, [e.message for e in test_app.exception])
        self.assertEqual(test_app.metric[1].value, "0.0%")

    def test_survival_page_labels_short_observation_horizon(self) -> None:
        test_app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=120).run()
        test_app.sidebar.radio[0].set_value("Survival Analysis").run()
        self.assertFalse(test_app.exception, [e.message for e in test_app.exception])
        captions = [item.value for item in test_app.caption]
        self.assertTrue(
            any("observation horizon" in caption for caption in captions),
            captions,
        )

        last_month_customer = self.customers.loc[
            self.customers["tenure"] == 72, "customerID"
        ].iloc[0]
        test_app.selectbox(key="survival_customer").set_value(last_month_customer).run()
        self.assertFalse(test_app.exception, [e.message for e in test_app.exception])
        self.assertTrue(
            any("beyond the model's last observed month" in item.value for item in test_app.warning)
        )

    def test_missing_artifacts_are_listed_and_rendered_as_an_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = app.required_project_files(
                Path(temporary_directory),
                Path(temporary_directory),
                Path(temporary_directory),
            )
        self.assertEqual(len(missing), 8)
        self.assertTrue(all(not path.exists() for path in missing))

        source = (
            "from pathlib import Path\n"
            "import app\n"
            f"app.render_missing_files_error({[str(path) for path in missing]!r})\n"
        )
        test_app = AppTest.from_string(source, default_timeout=30).run()
        self.assertFalse(test_app.exception, [e.message for e in test_app.exception])
        self.assertTrue(test_app.error)
        self.assertIn("required project files are missing", test_app.error[0].value)
        self.assertIn("Restore the listed data", test_app.error[0].value)

    def test_sidebar_dropdown_theme_has_accessible_text_contrast(self) -> None:
        config_path = PROJECT_ROOT / ".streamlit" / "config.toml"
        with config_path.open("rb") as config_file:
            theme = tomllib.load(config_file)["theme"]["sidebar"]

        def luminance(color: str) -> float:
            channels = [
                int(color[index : index + 2], 16) / 255
                for index in (1, 3, 5)
            ]
            linear = [
                channel / 12.92
                if channel <= 0.04045
                else ((channel + 0.055) / 1.055) ** 2.4
                for channel in channels
            ]
            return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

        contrast = (
            luminance(theme["textColor"]) + 0.05
        ) / (
            luminance(theme["secondaryBackgroundColor"]) + 0.05
        )
        self.assertGreaterEqual(contrast, 4.5)


class RetentionRoiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.customer = pd.DataFrame(
            {
                "customerID": ["fixture-1"],
                "ChurnRisk": [0.50],
                "CLV": [200.0],
                "MonthlyCharges": [100.0],
            }
        )

    def run_scenario(self, outreach_cost: float, budget: float = 1_000.0) -> dict[str, object]:
        return app.simulate_roi(
            active_customers=self.customer,
            budget=budget,
            discount_percent=5,
            discount_months=2,
            assumed_uplift_percent=10,
            outreach_cost=outreach_cost,
            minimum_risk_percent=30,
        )

    def test_profitable_scenario(self) -> None:
        result = self.run_scenario(outreach_cost=5)
        self.assertEqual(result["contacted"], 1)
        self.assertAlmostEqual(result["expected_savings"], 20.0)
        self.assertAlmostEqual(result["campaign_cost"], 15.0)
        self.assertAlmostEqual(result["net_profit"], 5.0)
        self.assertAlmostEqual(result["roi"], 1 / 3)

    def test_break_even_scenario(self) -> None:
        result = self.run_scenario(outreach_cost=10)
        self.assertAlmostEqual(result["expected_savings"], 20.0)
        self.assertAlmostEqual(result["campaign_cost"], 20.0)
        self.assertAlmostEqual(result["net_profit"], 0.0)
        self.assertAlmostEqual(result["roi"], 0.0)

    def test_zero_budget_contacts_no_customers(self) -> None:
        result = self.run_scenario(outreach_cost=5, budget=0)
        self.assertEqual(result["contacted"], 0)
        self.assertEqual(result["campaign_cost"], 0.0)
        self.assertEqual(result["expected_savings"], 0.0)
        self.assertEqual(result["net_profit"], 0.0)
        self.assertIsNone(result["roi"])

    def test_loss_scenario(self) -> None:
        result = self.run_scenario(outreach_cost=15)
        self.assertAlmostEqual(result["expected_savings"], 20.0)
        self.assertAlmostEqual(result["campaign_cost"], 25.0)
        self.assertAlmostEqual(result["net_profit"], -5.0)
        self.assertAlmostEqual(result["roi"], -0.2)

    def test_budget_limit_prioritizes_highest_risk_and_caps_uplift(self) -> None:
        customers = pd.concat(
            [
                self.customer,
                self.customer.assign(
                    customerID="fixture-2", ChurnRisk=0.8, CLV=300.0
                ),
            ],
            ignore_index=True,
        )
        result = app.simulate_roi(
            active_customers=customers,
            budget=20,
            discount_percent=0,
            discount_months=1,
            assumed_uplift_percent=30,
            outreach_cost=20,
            minimum_risk_percent=30,
        )
        self.assertEqual(result["eligible_count"], 2)
        self.assertEqual(result["contacted"], 1)
        selected = result["selected"]
        self.assertEqual(selected.iloc[0]["customerID"], "fixture-2")
        self.assertAlmostEqual(selected.iloc[0]["retention_uplift"], 0.3)
        self.assertEqual(result["budget_remaining"], 0.0)

    def test_invalid_budget_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.run_scenario(outreach_cost=5, budget=-1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
