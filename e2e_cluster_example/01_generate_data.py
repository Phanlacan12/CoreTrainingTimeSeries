import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_epidemic_data(
    num_locations=5, num_diseases=2, start_date="2023-01-01", days=30, output_dir="data"
):
    os.makedirs(output_dir, exist_ok=True)

    locations = [f"LOC_{i:03d}" for i in range(1, num_locations + 1)]
    diseases = [f"DIS_{i:02d}" for i in range(1, num_diseases + 1)]

    # Generate static data
    static_records = []
    for loc in locations:
        for dis in diseases:
            series_id = f"{loc}_{dis}"
            static_records.append(
                {
                    "series_id": series_id,
                    "timestamp": pd.to_datetime(start_date),
                    "disease_type": np.random.choice(
                        ["Viral", "Bacterial", "Parasitic"]
                    ),
                    "danger_level": np.random.choice(
                        ["Low", "Medium", "High", "Critical"]
                    ),
                    "population_density": np.random.uniform(100, 5000),
                    "description": f"Notes for {series_id}",
                }
            )

    df_static = pd.DataFrame(static_records)
    df_static.to_csv(f"{output_dir}/epidemic_static.csv", index=False)
    print(f"Generated {len(df_static)} static records.")

    # Generate time series data
    start_dt = pd.to_datetime(start_date)
    hours = days * 24

    ts_records = []
    for loc in locations:
        for dis in diseases:
            series_id = f"{loc}_{dis}"
            base_cases = np.random.randint(500, 1000)
            phase_shift = np.random.uniform(0, 2 * np.pi)

            for h in range(hours):
                current_time = start_dt + timedelta(hours=h)

                # simulate daily and weekly seasonal pattern
                daily_factor = np.sin(
                    (current_time.hour * 2 * np.pi / 24) + phase_shift
                )
                weekly_factor = np.cos((h * 2 * np.pi / 168) + phase_shift)
                hour_factor = daily_factor + 1.5  # shift positive for covariates

                # target case generation with seasonality and noise
                amplitude = base_cases * 0.3
                seasonal_component = amplitude * (daily_factor + weekly_factor)
                cases = int(
                    base_cases
                    + seasonal_component
                    + np.random.normal(0, base_cases * 0.05)
                )
                cases = max(50, cases)  # ensure strongly positive

                if np.random.random() < 0.02:  # 2% chance of spike
                    cases += np.random.randint(200, 500)

                ts_records.append(
                    {
                        "series_id": series_id,
                        "timestamp": current_time,
                        "cases": cases,
                        # Future Covariates
                        "day_of_week": current_time.dayofweek,
                        "hour_of_day": current_time.hour,
                        "is_holiday": 1
                        if current_time.dayofweek >= 5
                        else 0,  # simplified holiday
                        "temperature": np.random.uniform(20, 35) + hour_factor * 2,
                        "humidity": np.random.uniform(40, 90),
                        # Past Covariates
                        "mobility_index": np.random.uniform(0.5, 2.0) * hour_factor,
                        "vaccination_rate": min(
                            1.0, np.random.uniform(0.1, 0.9) + h * 0.0001
                        ),
                        "nearby_cases": max(0, cases + int(np.random.normal(0, 5))),
                    }
                )

    df_ts = pd.DataFrame(ts_records)
    df_ts.to_csv(f"{output_dir}/epidemic_cases_ts.csv", index=False)
    print(f"Generated {len(df_ts)} time series records.")

    # Generate Cluster Mapping (3 clusters)
    df_cluster = df_static[["series_id"]].copy()
    df_cluster["cluster_id"] = [
        f"Cluster_{np.random.randint(1, 4)}" for _ in range(len(df_cluster))
    ]
    df_cluster.to_csv(f"{output_dir}/cluster_mapping.csv", index=False)
    print(f"Generated cluster mappings for 3 clusters.")

    return df_ts, df_static, df_cluster


if __name__ == "__main__":
    generate_epidemic_data()
