# =============================================================================
#  Healthcare Patient Analytics - End-to-End Python Project
#  Author  : Akansha Chowdhury
#  Dataset : Kaggle Healthcare Patient Analytics (healthcare_patient_analytics_seaborn.csv)
#  Purpose : Ingest, clean, engineer features, compute KPIs, and visualise
#            patient visit data across 5,000 records and 12 attributes.
# =============================================================================

# -- Standard library ----------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

# -- Third-party ---------------------------------------------------------------
import numpy  as np
import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import pearsonr

# -- Global plot style ---------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi"      : 120,
    "axes.spines.top" : False,
    "axes.spines.right": False,
})

# =============================================================================
#  SECTION 1 - SETUP & INGESTION
# =============================================================================

print("=" * 65)
print("  SECTION 1 - DATA INGESTION")
print("=" * 65)

# -- Load CSV ------------------------------------------------------------------
CSV_PATH = "project/healthcare_patient_analytics_seaborn.csv"
df = pd.read_csv(CSV_PATH)

# -- Health checks -------------------------------------------------------------
print(f"\n[INFO] Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("\n-- Column names --------------------------------------------------")
print(df.columns.tolist())

print("\n-- Data types ----------------------------------------------------")
print(df.dtypes)

print("\n-- First 5 rows --------------------------------------------------")
print(df.head())

print("\n-- Descriptive statistics ----------------------------------------")
print(df.describe().round(2))

# =============================================================================
#  SECTION 2 - DATA CLEANING
# =============================================================================

print("\n" + "=" * 65)
print("  SECTION 2 - DATA CLEANING")
print("=" * 65)

# -- 2a. Null value audit ------------------------------------------------------
null_counts = df.isnull().sum()
print(f"\n[NULL CHECK] Total null cells : {null_counts.sum()}")

# Rule: numeric nulls - median impute; categorical nulls - mode impute
for col in df.select_dtypes(include="number").columns:
    if null_counts[col] > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"  [IMPUTED] {col} - filled {null_counts[col]} nulls with median")

for col in df.select_dtypes(include="object").columns:
    if null_counts[col] > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"  [IMPUTED] {col} - filled {null_counts[col]} nulls with mode")

if null_counts.sum() == 0:
    print("  [PASS] No null values found - no imputation required.")

# -- 2b. Duplicate row removal -------------------------------------------------
rows_before = len(df)
df = df.drop_duplicates()
dupes_removed = rows_before - len(df)
print(f"\n[DEDUP] Duplicate rows removed: {dupes_removed}")
print(f"  Rows remaining: {len(df):,}")

# -- 2c. Unique patient_id check -----------------------------------------------
dup_ids = df[df.duplicated("patient_id", keep=False)]
if len(dup_ids) > 0:
    df = df.drop_duplicates(subset="patient_id", keep="first")
    print(f"[ID CHECK] Duplicate patient_ids resolved: {len(dup_ids)} rows - kept first occurrence.")
else:
    print("[ID CHECK] All patient_id values are unique - no action required.")

# -- 2d. Parse visit_date to datetime64 ---------------------------------------
df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")
unparsed = df["visit_date"].isnull().sum()
assert unparsed == 0, f"[ERROR] {unparsed} visit_date values failed to parse."
print(f"[DATE] visit_date cast to datetime64 - {unparsed} parse failures.")

# -- 2e. Cast categorical columns to category dtype ---------------------------
CAT_COLS = ["age_group", "gender", "region", "department", "treatment_type", "visit_type"]
df[CAT_COLS] = df[CAT_COLS].astype("category")
print(f"[DTYPE] {CAT_COLS} cast to category dtype.")

print(f"\n[INFO] Clean dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

# =============================================================================
#  SECTION 3 - FEATURE ENGINEERING
# =============================================================================

print("\n" + "=" * 65)
print("  SECTION 3 - FEATURE ENGINEERING")
print("=" * 65)

# -- 3a. Daily Treatment Cost --------------------------------------------------
# Formula: daily_treatment_cost = treatment_cost / length_of_stay_days
# Guard  : returns NaN for zero-LOS (same-day) records to avoid ZeroDivisionError
df["daily_treatment_cost"] = df.apply(
    lambda r: r["treatment_cost"] / r["length_of_stay_days"]
    if r["length_of_stay_days"] > 0 else np.nan,
    axis=1
)
zero_los_count = df["length_of_stay_days"].eq(0).sum()
print(f"\n[FEATURE] daily_treatment_cost computed.")
print(f"  Zero-LOS (same-day) records assigned NaN: {zero_los_count}")
print(f"  Valid daily_cost records: {df['daily_treatment_cost'].notna().sum():,}")

# -- 3b. High Readmission Risk Flag -------------------------------------------
# Threshold: readmission_risk >= 0.40
# Rationale: 0.40 represents the clinically significant risk inflection point
df["is_high_risk"] = (df["readmission_risk"] >= 0.40).astype(int)
hr_count = df["is_high_risk"].sum()
print(f"\n[FEATURE] is_high_risk flag applied (threshold >= 0.40).")
print(f"  High-risk patients: {hr_count:,} of {len(df):,} ({hr_count/len(df)*100:.1f}%)")

# -- 3c. High Cost Outlier Flag (IQR method) -----------------------------------
q1  = df["treatment_cost"].quantile(0.25)
q3  = df["treatment_cost"].quantile(0.75)
iqr = q3 - q1
iqr_fence = q3 + 1.5 * iqr
df["is_high_cost"] = (df["treatment_cost"] > iqr_fence).astype(int)
print(f"\n[FEATURE] is_high_cost flag applied (IQR upper fence = ${iqr_fence:,.0f}).")
print(f"  High-cost patients: {df['is_high_cost'].sum()} ({df['is_high_cost'].mean()*100:.1f}%)")

# -- 3d. Temporal Features ----------------------------------------------------
df["visit_month"]   = df["visit_date"].dt.month        # 1-12
df["visit_quarter"] = df["visit_date"].dt.quarter      # 1-4
df["visit_dow"]     = df["visit_date"].dt.dayofweek    # 0=Mon - 6=Sun
print(f"\n[FEATURE] Temporal features extracted: visit_month, visit_quarter, visit_dow.")
print(f"  Date range: {df['visit_date'].min().date()} - {df['visit_date'].max().date()}")

# =============================================================================
#  SECTION 4 - EXPLORATORY ANALYSIS & KPI CALCULATIONS
# =============================================================================

print("\n" + "=" * 65)
print("  SECTION 4 - EXPLORATORY ANALYSIS & KPI CALCULATIONS")
print("=" * 65)

# -- KPI 4a. Department-level operational summary ------------------------------
dept_summary = df.groupby("department", observed=True).agg(
    total_visits    = ("patient_id",           "count"),
    avg_cost        = ("treatment_cost",        "mean"),
    total_spend     = ("treatment_cost",        "sum"),
    avg_los         = ("length_of_stay_days",   "mean"),
    avg_recovery    = ("recovery_score",        "mean"),
    high_risk_count = ("is_high_risk",          "sum"),
    high_risk_rate  = ("is_high_risk",          "mean"),
).reset_index()
dept_summary["high_risk_pct"] = (dept_summary["high_risk_rate"] * 100).round(1)
dept_summary["cost_cv"]       = (
    df.groupby("department", observed=True)["treatment_cost"].std().values /
    df.groupby("department", observed=True)["treatment_cost"].mean().values * 100
).round(1)

print("\n[KPI 4a] Department-Level Summary:")
print(dept_summary[
    ["department", "total_visits", "avg_cost", "total_spend", "avg_los", "avg_recovery", "high_risk_pct"]
].sort_values("avg_cost", ascending=False).to_string(index=False))

# -- KPI 4b. Age group cost & risk profile -------------------------------------
age_summary = df.groupby("age_group", observed=True).agg(
    total    = ("patient_id",     "count"),
    avg_cost = ("treatment_cost", "mean"),
    avg_rec  = ("recovery_score", "mean"),
    hr_pct   = ("is_high_risk",   "mean"),
).reset_index()
age_summary["hr_pct"] = (age_summary["hr_pct"] * 100).round(1)

print("\n[KPI 4b] Age Group - Cost & Risk Profile:")
print(age_summary.sort_values("avg_cost", ascending=False).to_string(index=False))

# -- KPI 4c. Visit type comparison ---------------------------------------------
visit_summary = df.groupby("visit_type", observed=True).agg(
    count    = ("patient_id",          "count"),
    avg_cost = ("treatment_cost",      "mean"),
    avg_los  = ("length_of_stay_days", "mean"),
    hr_pct   = ("is_high_risk",        "mean"),
).reset_index()
visit_summary["hr_pct"]     = (visit_summary["hr_pct"] * 100).round(1)
visit_summary["volume_pct"] = (visit_summary["count"] / visit_summary["count"].sum() * 100).round(1)

print("\n[KPI 4c] Visit Type Comparison (Routine vs Emergency):")
print(visit_summary.to_string(index=False))

# -- KPI 4d. LOS vs Recovery Score correlation ---------------------------------
r_val, p_val = pearsonr(df["length_of_stay_days"], df["recovery_score"])
print(f"\n[KPI 4d] LOS vs Recovery Score - Pearson r = {r_val:.4f}  (p = {p_val:.4f})")
print(f"  Interpretation: {'No meaningful correlation' if abs(r_val) < 0.1 else 'Moderate/strong correlation'}")

# -- KPI 4e. Treatment type recovery -------------------------------------------
treat_rec = df.groupby("treatment_type", observed=True)["recovery_score"].agg(
    ["mean", "min", "max", "count"]
).sort_values("mean", ascending=False).round(2)
print("\n[KPI 4e] Avg Recovery Score by Treatment Type:")
print(treat_rec.to_string())

# -- KPI 4f. Regional cost summary ---------------------------------------------
region_summary = df.groupby("region", observed=True).agg(
    avg_cost = ("treatment_cost", "mean"),
    hr_pct   = ("is_high_risk",   "mean"),
).reset_index()
region_summary["hr_pct"] = (region_summary["hr_pct"] * 100).round(1)

print("\n[KPI 4f] Regional Avg Cost & High-Risk Rate:")
print(region_summary.sort_values("avg_cost", ascending=False).to_string(index=False))

# =============================================================================
#  SECTION 5 - VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 65)
print("  SECTION 5 - VISUALIZATIONS")
print("=" * 65)

# -----------------------------------------------------------------------------
#  PLOT 1 - Horizontal Bar Chart: Average Treatment Cost by Department
# -----------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(9, 5))

# Sort departments by descending avg cost for ranked display
dept_plot = dept_summary.sort_values("avg_cost", ascending=True)
bars = ax1.barh(
    dept_plot["department"],
    dept_plot["avg_cost"],
    color=sns.color_palette("Blues_d", len(dept_plot)),
    edgecolor="white",
    height=0.55,
)

# Hospital average reference line
hosp_avg_cost = df["treatment_cost"].mean()
ax1.axvline(
    x=hosp_avg_cost, color="#ef4444", linewidth=1.4,
    linestyle="--", label=f"Hospital Avg  ${hosp_avg_cost:,.0f}"
)

# Value labels on bars
for bar in bars:
    width = bar.get_width()
    ax1.text(
        width + 200, bar.get_y() + bar.get_height() / 2,
        f"${width:,.0f}", va="center", ha="left", fontsize=10, color="#1f2328"
    )

ax1.set_xlabel("Average Treatment Cost (USD)", fontsize=11)
ax1.set_title("Average Treatment Cost by Department", fontsize=13, fontweight="bold", pad=14)
ax1.legend(fontsize=10)
ax1.set_xlim(0, dept_plot["avg_cost"].max() * 1.18)
plt.tight_layout()
plt.savefig("project/outputs/plots/plot1_avg_cost_by_dept.png", dpi=150, bbox_inches="tight")
plt.show()
print("[PLOT 1] Saved -> project/outputs/plots/plot1_avg_cost_by_dept.png")

# -----------------------------------------------------------------------------
#  PLOT 2 - Stacked Bar Chart: High vs Standard Readmission Risk by Department
# -----------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 5))

# Build stacked data: % high-risk and % standard-risk per department
risk_plot = dept_summary[["department", "high_risk_pct"]].copy()
risk_plot["standard_pct"] = 100 - risk_plot["high_risk_pct"]
risk_plot = risk_plot.sort_values("high_risk_pct", ascending=False)

x     = np.arange(len(risk_plot))
width = 0.55

bars_hr  = ax2.bar(x, risk_plot["high_risk_pct"],   width, label="High Risk  (- 0.40)", color="#ef4444", edgecolor="white")
bars_std = ax2.bar(x, risk_plot["standard_pct"],     width, bottom=risk_plot["high_risk_pct"], label="Standard Risk", color="#93c5fd", edgecolor="white")

# Label high-risk % inside each red segment
for i, (hr, dept) in enumerate(zip(risk_plot["high_risk_pct"], risk_plot["department"])):
    ax2.text(i, hr / 2, f"{hr}%", ha="center", va="center", fontsize=10, fontweight="bold", color="white")

# Hospital avg risk reference line
hosp_hr_pct = df["is_high_risk"].mean() * 100
ax2.axhline(y=hosp_hr_pct, color="#1d4ed8", linewidth=1.4, linestyle="--",
            label=f"Hospital Avg  {hosp_hr_pct:.1f}%")

ax2.set_xticks(x)
ax2.set_xticklabels(risk_plot["department"], fontsize=10)
ax2.set_ylabel("Percentage of Patients (%)", fontsize=11)
ax2.set_ylim(0, 115)
ax2.set_title("Readmission Risk Composition by Department", fontsize=13, fontweight="bold", pad=14)
ax2.legend(fontsize=10, loc="upper right")
plt.tight_layout()
plt.savefig("project/outputs/plots/plot2_readmission_risk_by_dept.png", dpi=150, bbox_inches="tight")
plt.show()
print("[PLOT 2] Saved -> project/outputs/plots/plot2_readmission_risk_by_dept.png")

# -----------------------------------------------------------------------------
#  PLOT 3 - Scatter Plot with Regression Line: LOS vs Recovery Score
# -----------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(9, 6))

# Colour points by department
dept_list   = df["department"].cat.categories.tolist()
palette_map = dict(zip(dept_list, sns.color_palette("tab10", len(dept_list))))
# Convert category to str before mapping to avoid pandas MultiIndex bug
colors      = df["department"].astype(str).map(palette_map)

ax3.scatter(
    df["length_of_stay_days"],
    df["recovery_score"],
    c=colors.tolist(), alpha=0.35, s=18, linewidths=0
)

# OLS regression line (numpy polyfit)
m, b = np.polyfit(df["length_of_stay_days"], df["recovery_score"], deg=1)
x_line = np.linspace(df["length_of_stay_days"].min(), df["length_of_stay_days"].max(), 200)
ax3.plot(x_line, m * x_line + b, color="#1d4ed8", linewidth=2.0,
         label=f"Regression line  (r = {r_val:.3f})")

# Mean crosshair reference lines
ax3.axvline(x=df["length_of_stay_days"].mean(), color="#57606a", linewidth=0.9,
            linestyle=":", alpha=0.8, label=f"Avg LOS = {df['length_of_stay_days'].mean():.2f} days")
ax3.axhline(y=df["recovery_score"].mean(), color="#57606a", linewidth=0.9,
            linestyle=":", alpha=0.8, label=f"Avg Recovery = {df['recovery_score'].mean():.2f}")

# Department colour legend
legend_patches = [
    mpatches.Patch(color=palette_map[d], label=d) for d in dept_list
]
ax3.legend(handles=legend_patches, fontsize=9, loc="upper right",
           title="Department", title_fontsize=9)

# Regression line separate legend entry
reg_line = plt.Line2D([0], [0], color="#1d4ed8", linewidth=2.0,
                      label=f"Regression  r = {r_val:.3f}")
handles2, _ = ax3.get_legend_handles_labels()
ax3.legend(
    handles=legend_patches + [reg_line],
    fontsize=9, loc="upper right", title="", ncol=1
)

ax3.set_xlabel("Length of Stay (Days)", fontsize=11)
ax3.set_ylabel("Recovery Score (0 - 100)", fontsize=11)
ax3.set_title("Length of Stay vs Patient Recovery Score", fontsize=13, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig("project/outputs/plots/plot3_los_vs_recovery.png", dpi=150, bbox_inches="tight")
plt.show()
print("[PLOT 3] Saved -> project/outputs/plots/plot3_los_vs_recovery.png")

# -----------------------------------------------------------------------------
#  PLOT 4 - Donut Chart: Patient Distribution by Visit Type
# -----------------------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(7, 6))

vt_counts = df["visit_type"].value_counts()
labels    = vt_counts.index.tolist()
sizes     = vt_counts.values
colors_d  = ["#3b82f6", "#ef4444"]   # blue = Routine, red = Emergency
explode   = (0.04, 0.04)

wedges, texts, autotexts = ax4.pie(
    sizes,
    labels=None,
    autopct="%1.1f%%",
    startangle=90,
    colors=colors_d,
    explode=explode,
    pctdistance=0.72,
    wedgeprops={"edgecolor": "white", "linewidth": 2.5},
)

for at in autotexts:
    at.set_fontsize(13)
    at.set_fontweight("bold")
    at.set_color("white")

# Inner white circle to create donut effect
centre_circle = plt.Circle((0, 0), 0.48, color="white")
ax4.add_patch(centre_circle)

# Centre annotation: total patients
ax4.text(0, 0.08, f"{len(df):,}", ha="center", va="center",
         fontsize=18, fontweight="bold", color="#1f2328")
ax4.text(0, -0.18, "Total Patients", ha="center", va="center",
         fontsize=10, color="#57606a")

# Legend with counts
legend_labels = [f"{lbl}  ({cnt:,}  -  {cnt/len(df)*100:.1f}%)"
                 for lbl, cnt in zip(labels, sizes)]
ax4.legend(
    wedges, legend_labels,
    loc="lower center", bbox_to_anchor=(0.5, -0.08),
    fontsize=11, frameon=False
)

ax4.set_title("Patient Distribution by Visit Type", fontsize=13, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig("project/outputs/plots/plot4_visit_type_donut.png", dpi=150, bbox_inches="tight")
plt.show()
print("[PLOT 4] Saved -> project/outputs/plots/plot4_visit_type_donut.png")

# =============================================================================
#  SECTION 6 - FINAL SUMMARY METRICS (Console Report)
# =============================================================================

# -- Compute all summary values ------------------------------------------------
total_patients  = len(df)
overall_avg_cost= df["treatment_cost"].mean()
overall_avg_los = df["length_of_stay_days"].mean()
overall_avg_rec = df["recovery_score"].mean()
total_spend     = df["treatment_cost"].sum()
hr_rate_pct     = df["is_high_risk"].mean() * 100
hr_total        = df["is_high_risk"].sum()

highest_cost_dept  = dept_summary.sort_values("avg_cost", ascending=False).iloc[0]
highest_risk_dept  = dept_summary.sort_values("high_risk_pct", ascending=False).iloc[0]
highest_cost_region= region_summary.sort_values("avg_cost", ascending=False).iloc[0]

routine_ct  = int(visit_summary.loc[visit_summary["visit_type"] == "Routine",  "count"].values[0])
emergency_ct= int(visit_summary.loc[visit_summary["visit_type"] == "Emergency","count"].values[0])

routine_hr  = int(df[(df["visit_type"] == "Routine")   & (df["is_high_risk"] == 1)].shape[0])
emergency_hr= int(df[(df["visit_type"] == "Emergency") & (df["is_high_risk"] == 1)].shape[0])

cost_cv      = df["treatment_cost"].std() / df["treatment_cost"].mean() * 100
highest_age_cost = df.groupby("age_group", observed=True)["treatment_cost"].mean().idxmax()
lowest_rec_age   = df.groupby("age_group", observed=True)["recovery_score"].mean().idxmin()

# -- Print formatted report ----------------------------------------------------
DIVIDER = "=" * 65
SEP     = "-" * 65

print(f"\n{DIVIDER}")
print(f"  HEALTHCARE PATIENT ANALYTICS - FINAL SUMMARY REPORT")
print(f"  Author : Akansha Chowdhury")
print(DIVIDER)

print(f"\n  DATASET")
print(SEP)
print(f"  Total patient records       :  {total_patients:>10,}")
print(f"  Source attributes (raw)     :  {12:>10}")
print(f"  Engineered features added   :  {df.shape[1] - 12:>10}")
print(f"  Date range                  :  {df['visit_date'].min().date()} - {df['visit_date'].max().date()}")
print(f"  Null cells in source cols   :  {df[['patient_id','visit_date','age_group','gender','region','department','treatment_type','visit_type','length_of_stay_days','treatment_cost','recovery_score','readmission_risk']].isnull().sum().sum():>10}")
print(f"  Duplicate rows removed      :  {dupes_removed:>10}")

print(f"\n  HOSPITAL-WIDE KPIs")
print(SEP)
print(f"  Total network spend         : ${total_spend:>14,.0f}")
print(f"  Average treatment cost      : ${overall_avg_cost:>14,.2f}")
print(f"  Cost std deviation (CV)     : ${df['treatment_cost'].std():>14,.2f}  ({cost_cv:.1f}%)")
print(f"  Average length of stay      :  {overall_avg_los:>9.2f} days")
print(f"  Average recovery score      :  {overall_avg_rec:>9.2f} / 100")
print(f"  High readmission risk rate  :  {hr_rate_pct:>9.1f}%  ({hr_total:,} patients)")

print(f"\n  VISIT TYPE DISTRIBUTION")
print(SEP)
print(f"  Routine visits              :  {routine_ct:>6,}  ({routine_ct/total_patients*100:.1f}%)")
print(f"  Emergency visits            :  {emergency_ct:>6,}  ({emergency_ct/total_patients*100:.1f}%)")
print(f"  Routine:Emergency ratio     :  {routine_ct/emergency_ct:.2f} : 1")

print(f"\n  DEPARTMENT FINDINGS")
print(SEP)
print(f"  Highest avg cost dept       :  {highest_cost_dept['department']:<20} ${highest_cost_dept['avg_cost']:,.2f}/visit")
print(f"  Highest readmission risk    :  {highest_risk_dept['department']:<20} {highest_risk_dept['high_risk_pct']:.1f}%")
for _, row in dept_summary.sort_values("avg_cost", ascending=False).iterrows():
    print(f"    {row['department']:<22}  avg cost ${row['avg_cost']:,.0f}  |  risk {row['high_risk_pct']:.1f}%  |  avg LOS {row['avg_los']:.2f}d")

print(f"\n  READMISSION RISK INSIGHTS")
print(SEP)
print(f"  High-risk patients (>= 0.40):  {hr_total:,} of {total_patients:,}")
print(f"  High-risk via Routine visits:  {routine_hr:,}  ({routine_hr/hr_total*100:.1f}% of high-risk cohort)")
print(f"  High-risk via Emergency     :  {emergency_hr:,}  ({emergency_hr/hr_total*100:.1f}% of high-risk cohort)")
print(f"  Gen. Medicine Routine rate  :  {df[(df['department']=='General Medicine') & (df['visit_type']=='Routine')]['is_high_risk'].mean()*100:.1f}%")

print(f"\n  LOS vs RECOVERY CORRELATION")
print(SEP)
print(f"  Pearson r                   :  {r_val:>+.4f}")
print(f"  p-value                     :  {p_val:>.4f}")
print(f"  Interpretation              :  {'No meaningful correlation - LOS reduction carries no recovery risk.' if abs(r_val) < 0.1 else 'Statistically significant relationship detected.'}")

print(f"\n  COST-EFFICIENCY FLAGS")
print(SEP)
print(f"  Highest per-visit age group :  {highest_age_cost:<10} ${df.groupby('age_group', observed=True)['treatment_cost'].mean()[highest_age_cost]:,.2f}")
print(f"  Lowest recovery age group   :  {lowest_rec_age:<10} {df.groupby('age_group', observed=True)['recovery_score'].mean()[lowest_rec_age]:.2f}/100")
print(f"  Highest cost region         :  {highest_cost_region['region']:<10} ${highest_cost_region['avg_cost']:,.2f}/visit")
print(f"  Lowest cost region          :  {region_summary.sort_values('avg_cost').iloc[0]['region']:<10} ${region_summary.sort_values('avg_cost').iloc[0]['avg_cost']:,.2f}/visit")

print(f"\n  OUTPUT FILES")
print(SEP)
print(f"  Plot 1 - Avg Cost by Dept   :  project/outputs/plots/plot1_avg_cost_by_dept.png")
print(f"  Plot 2 - Risk by Dept       :  project/outputs/plots/plot2_readmission_risk_by_dept.png")
print(f"  Plot 3 - LOS vs Recovery    :  project/outputs/plots/plot3_los_vs_recovery.png")
print(f"  Plot 4 - Visit Type Donut   :  project/outputs/plots/plot4_visit_type_donut.png")

print(f"\n{DIVIDER}")
print(f"  Analysis complete.")
print(DIVIDER)

# =============================================================================
#  END OF SCRIPT
# =============================================================================
