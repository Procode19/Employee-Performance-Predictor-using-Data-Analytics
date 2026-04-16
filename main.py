#!/usr/bin/env python3
"""
Employee Performance Predictor - Industrial Pipeline Runner
Runs full ML pipeline: data → training → prediction
"""

import subprocess
import sys
import os
from datetime import datetime


def print_header(title):
    print("\n" + "=" * 80)
    print(f"🚀 {title}")
    print("=" * 80)


def run_command(command, description):
    print_header(description)
    print(f"⚙️ Executing: {command}\n")

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Stream output live
    for line in process.stdout:
        print(line, end="")

    process.wait()

    # Handle errors
    if process.returncode != 0:
        print("\n❌ ERROR OCCURRED:\n")
        for line in process.stderr:
            print(line, end="")
        sys.exit(process.returncode)

    print("\n✅ Step Completed Successfully\n")


def main():

    print_header("EMPLOYEE PERFORMANCE PREDICTOR PIPELINE")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Ensure correct directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    print(f"📁 Working Directory: {project_dir}")

    # ================= STEP 1 =================
    run_command(
        "python src/generate_data.py",
        "STEP 1: Generating Synthetic Employee Dataset"
    )

    # ================= STEP 2 =================
    run_command(
        "python src/train_model.py",
        "STEP 2: Training ML Model (Random Forest / Classification)"
    )

    # ================= STEP 3 =================
    run_command(
        "python src/predict_performance.py",
        "STEP 3: Running Sample Predictions"
    )

    # ================= FINAL =================
    print_header("PIPELINE COMPLETED SUCCESSFULLY")

    print("📊 Outputs Generated:")
    print("✔ data/employee_features.csv")
    print("✔ models/employee_perf_model.pkl")
    print("✔ prediction results printed above")

    print("\n🚀 Next Steps:")
    print("1. Run Streamlit Dashboard:")
    print("   streamlit run app/app.py")

    print("\n2. Improve model performance (fix imbalance issue)")
    print("3. Deploy project (Streamlit Cloud / Render / HuggingFace)")

    print(f"\n⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()