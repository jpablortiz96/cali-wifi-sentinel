from __future__ import annotations

import pandas as pd


def optimize_crews(impact_scores_df: pd.DataFrame, available_crews: int = 3) -> dict[str, object]:
    """Prioriza zonas para cuadrillas con una regla simple y transparente."""
    if impact_scores_df.empty:
        return {
            "recommended_zones": pd.DataFrame(),
            "waiting_zones": pd.DataFrame(),
            "coverage_territorial": "Sin datos",
            "riesgo_no_atencion": "No hay zonas priorizadas para evaluar.",
            "explanation": "No se puede optimizar sin un indice de impacto disponible.",
        }

    crews = max(int(available_crews), 1)
    remaining_df = impact_scores_df.sort_values(
        by=["final_impact_score", "technical_severity_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    selected_rows: list[pd.Series] = []
    selected_territories: list[str] = []

    while len(selected_rows) < crews and not remaining_df.empty:
        chosen_index = 0

        if "territorio" in remaining_df.columns and remaining_df["territorio"].notna().any():
            for index, candidate in remaining_df.iterrows():
                territory = candidate.get("territorio")
                same_territory_count = selected_territories.count(str(territory))
                if same_territory_count == 0 or float(candidate["final_impact_score"]) >= 90:
                    chosen_index = index
                    break

                alternative_df = remaining_df.iloc[index + 1 :]
                has_good_alternative = False
                for _, alternative in alternative_df.iterrows():
                    if (
                        str(alternative.get("territorio")) not in selected_territories
                        and float(alternative["final_impact_score"]) >= float(candidate["final_impact_score"]) - 10
                    ):
                        chosen_index = alternative.name
                        has_good_alternative = True
                        break

                if not has_good_alternative:
                    chosen_index = index
                    break

        chosen_row = remaining_df.loc[chosen_index]
        selected_rows.append(chosen_row)
        if pd.notna(chosen_row.get("territorio")):
            selected_territories.append(str(chosen_row.get("territorio")))
        remaining_df = remaining_df.drop(index=chosen_index).reset_index(drop=True)

    recommended_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    waiting_df = remaining_df.reset_index(drop=True)

    if not recommended_df.empty:
        recommended_df["razon_priorizacion"] = recommended_df.apply(
            lambda row: (
                f"Score {row['final_impact_score']:.2f}, clasificacion {row['classification']}, "
                f"severidad tecnica {row['technical_severity_score']:.2f}."
            ),
            axis=1,
        )
        recommended_df["riesgo_si_no_se_atiende"] = recommended_df["classification"].map(
            {
                "Critico": "Muy alto",
                "Alto": "Alto",
                "Medio": "Medio",
                "Bajo": "Bajo",
                "Observacion": "Bajo",
            }
        )

    if not waiting_df.empty:
        waiting_df["riesgo_si_no_se_atiende"] = waiting_df["classification"].map(
            {
                "Critico": "Muy alto",
                "Alto": "Alto",
                "Medio": "Medio",
                "Bajo": "Bajo",
                "Observacion": "Bajo",
            }
        )

    covered_territories = (
        recommended_df["territorio"].dropna().astype(str).nunique()
        if "territorio" in recommended_df.columns
        else 0
    )
    coverage_message = (
        f"Las cuadrillas cubren {covered_territories} territorios distintos."
        if covered_territories > 0
        else "No hay cobertura territorial diferenciada disponible."
    )

    critical_waiting = (
        waiting_df["classification"].isin(["Critico", "Alto"]).sum()
        if not waiting_df.empty
        else 0
    )
    risk_message = (
        f"Quedan {int(critical_waiting)} zonas de prioridad alta o critica en espera."
        if critical_waiting > 0
        else "No quedan zonas altas o criticas en espera con el escenario actual."
    )

    explanation = (
        "La priorizacion usa primero el score final de impacto y luego la severidad tecnica. "
        "Si existe territorio, intenta repartir cuadrillas entre territorios salvo criticidad extrema."
    )

    return {
        "recommended_zones": recommended_df,
        "waiting_zones": waiting_df,
        "coverage_territorial": coverage_message,
        "riesgo_no_atencion": risk_message,
        "explanation": explanation,
    }
