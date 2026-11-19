import argparse
import csv
import numpy as np
import pandas as pd
import statsmodels.api as sm

R_KJ = 8.314 / 1000.0


def run_regression(
    input_file: str,
    delta_n: float,
    temp: float = 298.0,
    write_csv: bool = False,
    descriptors_file: str = "solvent_descriptors.csv",
    output_file: str = "output.csv",
):
    # 1. Load input reaction data (Solvent ID, Solvent Name, log k)
    df_in = pd.read_csv(input_file, low_memory=False).iloc[:, :3].copy()
    df_in.columns = ["Marcus no", "Solvent", "Log k"]
    df_in["Marcus no"] = pd.to_numeric(df_in["Marcus no"], errors="coerce")
    df_in["Log k"] = pd.to_numeric(df_in["Log k"], errors="coerce")
    df_in = df_in.dropna(subset=["Marcus no", "Log k"]).copy()
    df_in["Marcus no"] = df_in["Marcus no"].astype(int)
    df_in["Solvent"] = df_in["Solvent"].fillna("").astype(str).str.strip()

    # 2. Load solvent descriptors (matched by Solvent ID)
    df_desc = pd.read_csv(descriptors_file, low_memory=False).iloc[:, :6].copy()
    df_desc.columns = ["Marcus no", "Solvent_desc", "as", "bs", "js", "is"]
    df_desc["Marcus no"] = pd.to_numeric(df_desc["Marcus no"], errors="coerce")
    for col in ["as", "bs", "js", "is"]:
        df_desc[col] = pd.to_numeric(df_desc[col], errors="coerce")
    df_desc = df_desc.dropna(subset=["Marcus no", "as", "bs", "js", "is"]).copy()
    df_desc["Marcus no"] = df_desc["Marcus no"].astype(int)

    # 3. Merge reaction data with solvent descriptors on Solvent ID
    df = pd.merge(
        df_in,
        df_desc[["Marcus no", "as", "bs", "js", "is"]],
        on="Marcus no",
        how="inner",
    )
    if df.empty:
        raise ValueError(
            f"No matching solvent data found between '{input_file}' and '{descriptors_file}'."
        )

    # 4. Free-energy transformation and OLS regression (Equation 7)
    df["ln_k"] = df["Log k"] * np.log(10)
    df["-RT ln(k_expt)"] = -R_KJ * temp * df["ln_k"]
    y_target = df["-RT ln(k_expt)"] - 0.5 * delta_n * df["as"] * df["bs"]

    X = pd.DataFrame(
        {
            "-RT ln k0": 1.0,
            "ΔB": -df["as"],
            "ΔA": -df["bs"],
            "ΔJ": -df["js"],
            "ΔI": -df["is"],
        }
    )

    model = sm.OLS(y_target, X).fit()

    # Reconstruct full calculated -RT ln(k) and R^2
    df["-RT ln(k_calc)"] = model.fittedvalues + 0.5 * delta_n * df["as"] * df["bs"]
    ssr = np.sum((df["-RT ln(k_expt)"] - df["-RT ln(k_calc)"]) ** 2)
    sst = np.sum((df["-RT ln(k_expt)"] - df["-RT ln(k_expt)"].mean()) ** 2)
    r_squared = 1.0 - (ssr / sst)

    # 5. Calculate %HB contribution (Equation S1)
    dB = float(model.params["ΔB"])
    dA = float(model.params["ΔA"])
    dJ = float(model.params["ΔJ"])
    dI = float(model.params["ΔI"])

    hb_term = -dB * df["as"] - dA * df["bs"] + 0.5 * delta_n * df["as"] * df["bs"]
    non_hb_term = -dJ * df["js"] - dI * df["is"]
    sd_hb = np.std(hb_term, ddof=1)
    sd_non_hb = np.std(non_hb_term, ddof=1)
    pct_hb = (sd_hb / (sd_hb + sd_non_hb)) * 100.0 if (sd_hb + sd_non_hb) > 0 else 0.0

    n_solvents = len(df)
    logk_range = float(df["Log k"].max() - df["Log k"].min())

    # 6. Print Table 2 summary to console
    params_order = ["ΔB", "ΔA", "ΔJ", "ΔI", "-RT ln k0"]
    print(f"\nRegression Summary ({input_file} | ΔN = {delta_n:.0f} | T = {temp:.0f} K)")
    print("-" * 52)
    for p in params_order:
        val = float(model.params[p])
        se = float(model.bse[p])
        print(f"{p:<12} {val:>8.1f} ± {se:<6.1f}")
    print("-" * 52)
    print(f"%HB:          {pct_hb:.0f}%")
    print(f"R^2:          {r_squared:.2f}")
    print(f"n:            {n_solvents}")
    print(f"range(log k): {logk_range:.1f}\n")

    # 7. Optionally write output.csv
    if write_csv:
        summary_rows = [
            ["Parameter", "Value", "SE", "", "Metric", "Value"],
            ["ΔB", f"{float(model.params['ΔB']):.1f}", f"{float(model.bse['ΔB']):.1f}", "", "%HB", f"{pct_hb:.0f}"],
            ["ΔA", f"{float(model.params['ΔA']):.1f}", f"{float(model.bse['ΔA']):.1f}", "", "R^2", f"{r_squared:.2f}"],
            ["ΔJ", f"{float(model.params['ΔJ']):.1f}", f"{float(model.bse['ΔJ']):.1f}", "", "n", f"{n_solvents}"],
            ["ΔI", f"{float(model.params['ΔI']):.1f}", f"{float(model.bse['ΔI']):.1f}", "", "range(log k)", f"{logk_range:.1f}"],
            ["-RT ln k0", f"{float(model.params['-RT ln k0']):.1f}", f"{float(model.bse['-RT ln k0']):.1f}", "", "deltaN", f"{delta_n:.0f}"],
            ["", "", "", "", "T (K)", f"{temp:.0f}"],
            [],
        ]

        out_table = df[
            [
                "Marcus no",
                "Solvent",
                "Log k",
                "as",
                "bs",
                "js",
                "is",
                "-RT ln(k_expt)",
                "-RT ln(k_calc)",
            ]
        ].copy()

        for col in ["as", "bs", "js", "is", "-RT ln(k_expt)", "-RT ln(k_calc)"]:
            out_table[col] = out_table[col].round(4)

        with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerows(summary_rows)
            writer.writerow(out_table.columns.tolist())
            writer.writerows(out_table.values.tolist())

        print(f"Detailed results written to '{output_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit solvent effects on reaction rates using the multiparameter H-bond equation."
    )
    parser.add_argument("input_file", help="Path to input CSV file (ID, Solvent, log k)")
    parser.add_argument("delta_n", type=float, help="Value of deltaN for the reaction")
    parser.add_argument(
        "-t",
        "--temp",
        type=float,
        default=298.0,
        help="Temperature in Kelvin (default: 298 K)",
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store_true",
        help="Write detailed summary and solvent table to 'output.csv'",
    )

    args = parser.parse_args()
    run_regression(
        input_file=args.input_file,
        delta_n=args.delta_n,
        temp=args.temp,
        write_csv=args.output,
    )