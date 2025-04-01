from pathlib import Path

import pandas as pd


def main():
    # Read the original CSV
    df = pd.read_csv("datasets/labels.csv")

    # Process each dataset separately
    datasets = ["Lipo", "Desmoid", "GIST"]

    for dataset in datasets:
        # Filter for current dataset
        mask = df["Dataset"] == dataset
        df_dataset = df[mask]

        # Select only the desired columns
        df_new = df_dataset[["Subject", "Diagnosis", "Diagnosis_binary"]]

        # Create output directory if it doesn't exist
        output_dir = Path(f"datasets/{dataset.lower()}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save to separate CSV file with dataset-specific name
        output_file = output_dir / f"{dataset.lower()}_labels.csv"
        df_new.to_csv(output_file, index=False)
        print(f"Created {output_file} with {len(df_new)} entries")

        # Save individual labels for each subject
        for _, row in df_new.iterrows():
            subject = row["Subject"]
            subject_dir = output_dir / subject
            if subject_dir.exists():
                # Create single-row DataFrame with just this subject's data
                subject_df = pd.DataFrame([row])
                label_file = subject_dir / f"{subject.lower()}_label.csv"
                subject_df.to_csv(label_file, index=False)
                print(f"Created {label_file}")


if __name__ == "__main__":
    main()
