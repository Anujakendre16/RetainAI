# RetainAI Streamlit dashboard test results

**Run date:** 2026-11-18  
**Environment:** Python 3.14.6, project `venv`, Streamlit 1.67.0  
**Command:** `.\venv\Scripts\python.exe -m unittest discover -s .\tests -p "test_*.py" -v`

## Result

**PASS - 15 tests passed, 0 failures, 0 errors.** The latest unittest run completed in
24.324 seconds.

| Test | Result |
|---|---|
| All five pages render; gender, contract, internet, segment, tenure, and customer-status filters apply; empty filter selection is handled | PASS |
| Churn scores are finite, within [0, 1], and match the saved-model scoring output | PASS |
| SHAP feature contributions are finite and reconstruct LightGBM raw scores | PASS |
| Conditional survival probabilities are bounded, non-increasing, and stop at the model observation limit | PASS |
| Kaplan-Meier survival probabilities are bounded and non-increasing | PASS |
| Saved CLV values agree with monthly charges x expected remaining months x 40% margin | PASS |
| Survival page labels its limited observation horizon and explains when no further estimate is available | PASS |
| Missing required files are identified and displayed as an actionable Streamlit error | PASS |
| Sidebar dropdown text/background contrast meets WCAG AA | PASS |
| ROI: profitable scenario | PASS |
| ROI: break-even scenario | PASS |
| ROI: zero-budget scenario | PASS |
| ROI: loss scenario | PASS |
| ROI audience respects budget, prioritizes highest predicted risk, and caps assumed uplift at predicted risk | PASS |
| Invalid negative ROI budget is rejected | PASS |

## Verified values and safeguards

- The sidebar theme uses foreground `#F3F6FB` on dropdown/widget surface `#1B2940`,
  measured contrast **13.48:1**.
- The saved Cox model's last observed follow-up month is **72**. The individual
  conditional curve is capped at 36 additional months and never extrapolates beyond
  month 72; a customer already at the horizon receives an explicit warning instead
  of a fabricated curve.
- The CLV check covers the saved customer CLV table and uses the existing 40% margin
  convention.
- Fixture ROI scenarios produced expected net profit of **+$5.00** (profitable),
  **$0.00** (break-even), **$0.00** with no customers contacted (zero budget), and
  **-$5.00** (loss).
- Missing data/model/report files are listed with a recovery instruction; model and
  notebook artifacts were not modified.

## Runtime notes

The test runner emitted Streamlit's expected "missing ScriptRunContext" warnings
because `AppTest` and pure helper tests run in-process without starting a browser
server. SHAP emitted its LightGBM output-format compatibility warning; the tests
passed and handle both supported SHAP return shapes. Neither warning represents an
application test failure.
