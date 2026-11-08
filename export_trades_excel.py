"""Export du backtest vers Excel : un trade par ligne, gagnants en vert, perdants en rouge.

Usage :
    from export_trades_excel import export_trades_to_excel
    export_trades_to_excel(report, "trades_backtest.xlsx", df=feats)
"""

import pandas as pd


GREEN_FILL = "C6EFCE"
GREEN_FONT = "006100"
RED_FILL = "FFC7CE"
RED_FONT = "9C0006"
NEUTRAL_FILL = "FFEB9C"
NEUTRAL_FONT = "9C6500"


def trades_to_dataframe(report, df=None):
    """Construit le DataFrame des trades depuis le rapport de backtest."""
    rows = []
    for i, tr in enumerate(report["trades"], start=1):
        row = {
            "trade": i,
            "side": "LONG" if tr.side == 1 else "SHORT",
            "qty": round(tr.qty, 6),
            "entry_price": round(tr.entry_price, 2),
            "exit_price": round(tr.exit_price, 2) if tr.exit_price is not None else None,
            "pnl_net": round(getattr(tr, "net_pnl", tr.pnl_fill(tr.exit_price) if tr.exit_price else 0.0), 2),
            "exit_reason": tr.exit_reason,
            "entry_index": tr.entry_index,
            "exit_index": tr.exit_index,
        }
        if df is not None and "timestamp" in df.columns:
            n = len(df)
            if 0 <= tr.entry_index < n:
                row["entry_time"] = str(df["timestamp"].iloc[tr.entry_index])
            if tr.exit_index is not None and 0 <= tr.exit_index < n:
                row["exit_time"] = str(df["timestamp"].iloc[tr.exit_index])
        rows.append(row)
    return pd.DataFrame(rows)


def export_trades_to_excel(report, path="trades_backtest.xlsx", df=None):
    """Genere le fichier Excel : feuille Trades (vert/rouge) + feuille Resume."""
    trades_df = trades_to_dataframe(report, df=df)

    col_order = [
        "trade", "entry_time", "exit_time", "side", "qty",
        "entry_price", "exit_price", "pnl_net", "exit_reason",
        "entry_index", "exit_index",
    ]
    trades_df = trades_df[[c for c in col_order if c in trades_df.columns]]

    summary = pd.DataFrame(
        [
            ("Capital initial ($)", report["initial_capital"]),
            ("Capital final ($)", round(report["final_capital"], 2)),
            ("Rendement total (%)", round(report["total_return_pct"], 3)),
            ("Nombre de trades", report["n_trades"]),
            ("Win rate (%)", round(report["win_rate"], 2)),
            ("Profit factor", round(report["profit_factor"], 3) if report["profit_factor"] != float("inf") else "infini"),
            ("Expectancy par trade ($)", round(report["expectancy_per_trade"], 4)),
            ("Max drawdown (%)", round(report["max_drawdown_pct"], 2)),
            ("Sharpe annualise", round(report["sharpe"], 2)),
            ("Frais totaux ($)", round(report["total_fees"], 2)),
            ("Mode d'execution", report.get("execution", "taker")),
            ("Fills maker", report.get("maker_fills", 0)),
            ("Annulations maker", report.get("maker_cancels", 0)),
        ],
        columns=["Metrique", "Valeur"],
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Resume", index=False)
        if trades_df.empty:
            pd.DataFrame({"info": ["Aucun trade sur la periode"]}).to_excel(
                writer, sheet_name="Trades", index=False
            )
        else:
            trades_df.to_excel(writer, sheet_name="Trades", index=False)
            ws = writer.sheets["Trades"]
            from openpyxl.styles import Font, PatternFill

            green = PatternFill(start_color=GREEN_FILL, end_color=GREEN_FILL, fill_type="solid")
            red = PatternFill(start_color=RED_FILL, end_color=RED_FILL, fill_type="solid")
            green_font = Font(color=GREEN_FONT)
            red_font = Font(color=RED_FONT)
            pnl_col = trades_df.columns.get_loc("pnl_net") + 1
            for row_idx, pnl in enumerate(trades_df["pnl_net"], start=2):
                cell = ws.cell(row=row_idx, column=pnl_col)
                if pnl > 0:
                    cell.fill = green
                    cell.font = green_font
                elif pnl < 0:
                    cell.fill = red
                    cell.font = red_font
            for col_idx, col in enumerate(trades_df.columns, start=1):
                width = 12
                if "time" in col:
                    width = 28
                elif col in ("entry_price", "exit_price", "pnl_net"):
                    width = 14
                ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = width
            ws.freeze_panes = "A2"
    print(f"Excel genere : {path} ({len(trades_df)} trades)")
    return path
