"""
Streamlit UI for reviewing and approving low-confidence FEWS explanations.

This app reads:
- human_review_queue.jsonl  (model-proposed items needing review)
- human_review_approved.jsonl (human-approved overrides)

Analysts can inspect shocks, drivers, and model explanations for each region/phase,
make corrections, and save an approved record that the core system will use as
authoritative ground truth on subsequent runs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from src import FEWSSystem
from src.config import HUMAN_REVIEW_QUEUE_FILE, HUMAN_REVIEW_APPROVED_FILE


QUEUE_PATH = Path(HUMAN_REVIEW_QUEUE_FILE)
APPROVED_PATH = Path(HUMAN_REVIEW_APPROVED_FILE)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load all JSON objects from a JSONL file. Returns empty list if file is missing."""
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """Append a JSON object as one line to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _find_matching_approved(
    region: str, ipc_phase: int, approved_records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Find approved records matching a region/phase pair."""
    region_lower = region.lower().strip()
    matches: List[Dict[str, Any]] = []
    for rec in approved_records:
        try:
            if int(rec.get("ipc_phase", -1)) != ipc_phase:
                continue
            if str(rec.get("region", "")).lower().strip() == region_lower:
                matches.append(rec)
        except (ValueError, TypeError):
            continue
    return matches


def main() -> None:
    st.set_page_config(page_title="FEWS Human Review Console", layout="wide")
    st.title("FEWS Human Review & FEWS Agent Console")
    st.write(
        "Review low-confidence explanations, correct shocks/drivers, and "
        "save human-approved overrides that the core system will use as ground truth. "
        "You can also chat with the FEWS agent for region-specific analysis."
    )

    # Global FEWSSystem instance shared across tabs
    if "fews_system" not in st.session_state:
        st.session_state["fews_system"] = None

    tab_review, tab_agent = st.tabs(
        ["Human review", "Chat with FEWS agent"]
    )

    # -------- TAB 1: HUMAN REVIEW --------
    with tab_review:
        queue_records = _load_jsonl(QUEUE_PATH)
        approved_records = _load_jsonl(APPROVED_PATH)

        # Sidebar: filters and stats
        st.sidebar.header("Review filters")
        status_filter = st.sidebar.selectbox(
            "Status",
            options=["Pending only", "All queue items"],
            index=0,
        )

        search_region = st.sidebar.text_input(
            "Filter by region name (contains, case-insensitive)", value=""
        ).strip()

        # Filter queue records
        filtered_queue: List[Dict[str, Any]] = []
        for rec in queue_records:
            if status_filter == "Pending only" and rec.get("status") != "pending":
                continue
            if search_region:
                if search_region.lower() not in str(rec.get("region", "")).lower():
                    continue
            filtered_queue.append(rec)

        st.sidebar.write(f"Queue items shown: {len(filtered_queue)}")
        st.sidebar.write(f"Total approved records: {len(approved_records)}")

        # Build a simple selection list
        if not filtered_queue:
            st.info("No queue items to review with current filters.")
            # Do not return; allow user to still access chat tab
        else:
            options = [
                f"{idx+1}. {rec.get('region', 'UNKNOWN')} | "
                f"IPC {rec.get('ipc_phase', '?')} | "
                f"{rec.get('geographic_full_name', '')}"
                for idx, rec in enumerate(filtered_queue)
            ]
            selected_label = st.selectbox("Select item to review", options=options)
            selected_index = options.index(selected_label)
            item = filtered_queue[selected_index]

            region = str(item.get("region", "UNKNOWN"))
            ipc_phase = int(item.get("ipc_phase", -1))
            geo_full = str(item.get("geographic_full_name", ""))

            st.subheader(f"Region: {region} (IPC Phase {ipc_phase})")
            st.text(f"Geographic full name: {geo_full}")
            st.text(f"Data quality: {item.get('data_quality', 'unknown')}")
            st.text(f"Created at: {item.get('created_at', '')}")

            # Show whether there is already an approved record for this region/phase
            matching_approved = _find_matching_approved(region, ipc_phase, approved_records)
            if matching_approved:
                st.warning(
                    f"⚠️ There are already {len(matching_approved)} approved record(s) "
                    f"for this region and IPC phase. Approving another will add a new version."
                )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Model-detected shocks")
                shocks_model = item.get("shocks_model", [])
                if not shocks_model:
                    st.write("No shocks recorded in this queue item.")
                else:
                    for idx, shock in enumerate(shocks_model, start=1):
                        st.markdown(
                            f"**{idx}. {shock.get('type', 'unknown').upper()}** — "
                            f"{shock.get('description', '')}"
                        )
                        st.caption(f"Evidence: {shock.get('evidence', '')[:300]}...")
                        st.caption(f"Confidence: {shock.get('confidence', 'unknown')}")

                st.markdown("### Model drivers")
                drivers_model = item.get("drivers_model", [])
                if drivers_model:
                    st.write(", ".join(drivers_model))
                else:
                    st.write("No drivers were recorded.")

                st.markdown("### Sources (model)")
                sources_model = item.get("sources_model", [])
                if sources_model:
                    st.write(", ".join(sorted(set(sources_model))))
                else:
                    st.write("No sources listed.")

            with col2:
                st.markdown("### Edit shocks (JSON)")
                default_shocks_json = json.dumps(
                    item.get("shocks_model", []), indent=2, ensure_ascii=False
                )
                shocks_human_input = st.text_area(
                    "Shocks JSON (edit to correct/add/remove shocks)",
                    value=default_shocks_json,
                    height=260,
                )

                st.markdown("### Edit drivers (comma-separated)")
                default_drivers_str = ", ".join(item.get("drivers_model", []))
                drivers_human_input = st.text_input(
                    "Drivers",
                    value=default_drivers_str,
                    help="Comma-separated list of drivers",
                )

            st.markdown("### Explanation (model output)")
            explanation_model = item.get("explanation_model", "")
            explanation_human_input = st.text_area(
                "Edit explanation (optional)",
                value=explanation_model,
                height=260,
            )

            reviewer_name = st.text_input(
                "Your name (for audit trail)", value="", placeholder="Analyst name"
            ).strip()

            if st.button("✅ Approve and save human-reviewed record"):
                # Parse shocks JSON
                try:
                    shocks_human = (
                        json.loads(shocks_human_input) if shocks_human_input.strip() else []
                    )
                    if not isinstance(shocks_human, list):
                        raise ValueError("Shocks JSON must be a list of objects.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to parse shocks JSON: {exc}")
                    st.stop()

                # Parse drivers
                drivers_human = [
                    d.strip() for d in drivers_human_input.split(",") if d.strip()
                ]

                record: Dict[str, Any] = {
                    "region": region,
                    "ipc_phase": ipc_phase,
                    "geographic_full_name": geo_full,
                    "shocks_human": shocks_human,
                    "drivers_human": drivers_human,
                    "explanation_human": explanation_human_input.strip(),
                    "sources_human": sources_model,
                    "reviewed_by": reviewer_name or None,
                    "reviewed_at": datetime.utcnow().isoformat(),
                }

                _append_jsonl(APPROVED_PATH, record)
                st.success("Human-reviewed record saved successfully.")
                st.experimental_rerun()

    # -------- TAB 2: FEWS AGENT (CLI-style actions) --------
    with tab_agent:
        st.subheader("FEWS agent")
        st.write(
            "Use the FEWS agent to identify at-risk regions, explain why a region is at risk, "
            "recommend interventions, or run a full analysis, mirroring the CLI options."
        )

        # Reuse global FEWSSystem
        cli_init_col1, cli_init_col2 = st.columns(2)
        with cli_init_col1:
            if st.session_state["fews_system"] is None:
                if st.button("Initialize FEWS system", key="cli_init_fews"):
                    with st.spinner("Initializing FEWS system (LLM + domain knowledge)..."):
                        system_cli = FEWSSystem()
                    st.session_state["fews_system"] = system_cli
                    st.success("FEWS system initialized.")
            else:
                st.success("FEWS system is initialized.", icon="✅")

        with cli_init_col2:
            if st.session_state["fews_system"] is not None:
                if st.button("Setup vector stores", key="cli_setup_vs"):
                    with st.spinner(
                        "Setting up vector stores... this may take a while on first run."
                    ):
                        st.session_state["fews_system"].setup_vector_stores()
                    st.success("Vector stores ready.", icon="📚")

        system_cli = st.session_state["fews_system"]
        if system_cli is None:
            st.info("Initialize the FEWS system above to access CLI-style actions.")
        else:
            st.markdown("### 1. Choose action")
            action = st.selectbox(
                "Action (matches CLI menu options)",
                options=[
                    "1 - Identify at-risk regions",
                    "2 - Explain why a region is at risk",
                    "3 - Recommend interventions for a region",
                    "4 - Full analysis (risk + why + interventions)",
                ],
            )

            # For actions that need a region, prepare a dropdown of regions from IPC data
            st.markdown("### 2. Region selection (for actions 2–4)")
            region_options: List[str] = []
            assessments_for_dropdown: List[Any] = []
            try:
                # Use min_phase=1 to list all regions present in IPC data
                assessments_for_dropdown = system_cli.ipc_parser.identify_at_risk_regions(
                    min_phase=1,
                    include_deteriorating=True,
                    include_projected=True,
                )
                region_options = [
                    f"{a.region} (IPC {a.current_phase}, {a.risk_level})"
                    for a in assessments_for_dropdown
                ]
            except Exception as exc:  # noqa: BLE001
                st.error(f"Error loading IPC regions: {exc}")

            selected_region_label = None
            selected_assessment = None
            if region_options:
                selected_region_label = st.selectbox(
                    "Region (ignored for action 1)",
                    options=region_options,
                    index=0,
                )
                # Map back to RegionRiskAssessment
                sel_idx = region_options.index(selected_region_label)
                selected_assessment = assessments_for_dropdown[sel_idx]

            if st.button("Run action", key="cli_run_action"):
                system = system_cli
                if action.startswith("1"):
                    # Function 1: Identify at-risk regions
                    with st.spinner("Identifying at-risk regions from IPC data..."):
                        at_risk = system.function1_identify_at_risk_regions()
                    if not at_risk:
                        st.warning("No at-risk regions identified (with current IPC data).")
                    else:
                        st.markdown("#### At-risk regions (sorted by risk level)")
                        rows = []
                        for a in at_risk:
                            rows.append(
                                {
                                    "Region": a.region,
                                    "IPC current": a.current_phase,
                                    "IPC ML1": a.projected_phase_ml1,
                                    "IPC ML2": a.projected_phase_ml2,
                                    "Risk level": a.risk_level,
                                    "Trend": a.trend,
                                    "Latest current date": a.latest_current_date,
                                    "Latest projection date": a.latest_projection_date,
                                    "Indicators": "; ".join(a.key_indicators),
                                }
                            )
                        st.dataframe(rows, use_container_width=True)

                else:
                    if selected_assessment is None:
                        st.warning(
                            "Select a region from the dropdown before running actions 2–4."
                        )
                    else:
                        region_name = selected_assessment.region
                        if action.startswith("2"):
                            # Explain why
                            with st.spinner(f"Explaining why {region_name} is at risk..."):
                                result = system.function2_explain_why(
                                    region_name, selected_assessment
                                )
                            st.markdown(f"#### Explanation for {region_name}")
                            st.write(f"IPC Phase: {result.get('ipc_phase')}")
                            st.write(f"Data quality: {result.get('data_quality')}")
                            st.markdown("##### Narrative")
                            st.write(result.get("explanation", ""))
                            if result.get("drivers"):
                                st.markdown("##### Drivers")
                                st.write(", ".join(result["drivers"]))
                            if result.get("sources"):
                                st.markdown("##### Sources")
                                st.write(", ".join(result["sources"]))

                        elif action.startswith("3"):
                            # Recommend interventions
                            with st.spinner(
                                f"Generating interventions for {region_name} "
                                "(running explanation first to get drivers)..."
                            ):
                                explanation = system.function2_explain_why(
                                    region_name, selected_assessment
                                )
                                drivers = explanation.get("drivers", [])
                                interventions = system.function3_recommend_interventions(
                                    region_name, selected_assessment, drivers
                                )
                            st.markdown(f"#### Intervention recommendations for {region_name}")
                            st.write(f"IPC Phase: {interventions.get('ipc_phase')}")
                            st.markdown("##### Recommendations")
                            st.write(interventions.get("recommendations", ""))
                            if interventions.get("sources"):
                                st.markdown("##### Sources")
                                st.write(", ".join(interventions["sources"]))
                            if interventions.get("limitations"):
                                st.markdown("##### Limitations")
                                st.write(interventions["limitations"])

                        else:
                            # Full analysis
                            with st.spinner(
                                f"Running full analysis for {region_name} "
                                "(risk assessment + why + interventions)..."
                            ):
                                # 1. Risk assessment (we already have selected_assessment)
                                assessment = selected_assessment

                                # 2. Explanation
                                explanation = system.function2_explain_why(
                                    region_name, assessment
                                )

                                # 3. Interventions
                                drivers = explanation.get("drivers", [])
                                interventions = system.function3_recommend_interventions(
                                    region_name, assessment, drivers
                                )

                            st.markdown(f"#### 1. Risk assessment — {region_name}")
                            st.write(f"Current IPC Phase: {assessment.current_phase}")
                            if assessment.projected_phase_ml1 is not None:
                                st.write(
                                    f"Projected (near-term) IPC ML1: {assessment.projected_phase_ml1}"
                                )
                            if assessment.projected_phase_ml2 is not None:
                                st.write(
                                    f"Projected (medium-term) IPC ML2: {assessment.projected_phase_ml2}"
                                )
                            st.write(f"Risk level: {assessment.risk_level}")
                            st.write(f"Trend: {assessment.trend}")
                            st.write(f"Indicators: {', '.join(assessment.key_indicators)}")

                            st.markdown("#### 2. Why is this region at risk?")
                            st.write(f"IPC Phase (from explanation): {explanation.get('ipc_phase')}")
                            st.write(f"Data quality: {explanation.get('data_quality')}")
                            st.markdown("##### Narrative")
                            st.write(explanation.get("explanation", ""))
                            if explanation.get("drivers"):
                                st.markdown("##### Drivers")
                                st.write(", ".join(explanation["drivers"]))
                            if explanation.get("sources"):
                                st.markdown("##### Sources")
                                st.write(", ".join(explanation["sources"]))

                            st.markdown("#### 3. Intervention recommendations")
                            st.write(f"IPC Phase (from interventions): {interventions.get('ipc_phase')}")
                            st.markdown("##### Recommendations")
                            st.write(interventions.get("recommendations", ""))
                            if interventions.get("sources"):
                                st.markdown("##### Sources")
                                st.write(", ".join(interventions["sources"]))
                            if interventions.get("limitations"):
                                st.markdown("##### Limitations")
                                st.write(interventions["limitations"])


if __name__ == "__main__":
    main()


