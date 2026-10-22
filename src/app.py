
from pathlib import Path

import pandas as pd
import streamlit as st

from hardware_profiler import get_hardware_summary


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Hardware-Aware GenAI Inference Optimizer",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Hardware-Aware GenAI Inference Optimizer")
st.write(
    "Compare measured language-model configurations and identify "
    "the best speed-memory trade-off for selected constraints."
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==================================================
# HARDWARE DETECTION
# ==================================================

hardware = get_hardware_summary()

st.header("Detected Local Hardware")

c1, c2, c3, c4 = st.columns(4)

c1.metric("CPU Cores", hardware["cpu_cores"])
c2.metric("Logical Cores", hardware["logical_cores"])
c3.metric("Total RAM", f'{hardware["total_ram_gb"]:.2f} GB')
c4.metric("Available RAM", f'{hardware["available_ram_gb"]:.2f} GB')

st.caption(f'Operating system: {hardware["operating_system"]}')
st.caption(f'Processor: {hardware["processor"]}')

if hardware["gpu_available"]:
    st.success(f'GPU detected: {hardware["gpu_name"]}')

    if hardware["gpu_memory_gb"] is not None:
        st.write(f'GPU memory: {hardware["gpu_memory_gb"]:.2f} GB')
else:
    st.info("No supported GPU acceleration detected by the profiler.")

st.write(f'**Detected runtime:** {hardware["recommended_runtime"]}')


# ==================================================
# LOCAL INFERENCE GUIDANCE
# ==================================================

st.divider()
st.header("Hardware-Aware Model Guidance")

available_ram = hardware["available_ram_gb"]
gpu_memory = hardware["gpu_memory_gb"]
runtime = hardware["recommended_runtime"]

st.write(f"Currently available system RAM: **{available_ram:.2f} GB**")

if available_ram < 2:
    st.warning(
        "Very little system RAM is currently available. Avoid loading "
        "a local language model until more memory is free."
    )
elif runtime == "CUDA" and gpu_memory is not None and gpu_memory >= 6:
    st.info(
        "A CUDA GPU with at least 6 GB VRAM is detected. Consider testing "
        "a compatible quantized model after checking free VRAM and RAM."
    )
elif runtime == "MPS":
    st.info(
        "Apple Metal acceleration is detected. Check model compatibility "
        "and memory requirements before running inference."
    )
elif available_ram >= 4:
    st.info(
        "CPU inference may be feasible with a small quantized model. "
        "Test actual memory usage and generation speed."
    )
else:
    st.warning(
        "Available RAM is limited. Free memory before attempting local inference."
    )

st.caption(
    "This assessment is a resource-based estimate, not a guarantee "
    "that a model will fit. Local inference is not yet integrated."
)


# ==================================================
# BENCHMARK FILE LOADER
# ==================================================

st.divider()
st.header("Model and Benchmark Selection")

# Each tuple contains:
# filename, expected model name, model ID, column-name mapping
benchmark_files = [
    (
        "benchmark_summary.csv",
        "Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        {
            "average_speed": "average_speed",
            "average_time": "average_time",
            "average_peak_memory": "average_peak_memory",
        },
    ),
    (
        "qwen05b_benchmark_summary.csv",
        "Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        {
            "avg_tokens_per_second": "average_speed",
            "avg_generation_time_seconds": "average_time",
            "avg_peak_gpu_memory_gb": "average_peak_memory",
        },
    ),
    (
        "tinyllama_benchmark_summary.csv",
        "TinyLlama-1.1B-Chat",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        {
            "avg_tokens_per_second": "average_speed",
            "avg_generation_time_seconds": "average_time",
            "avg_peak_gpu_memory_gb": "average_peak_memory",
        },
    ),
]

frames = []
missing_files = []

for filename, model_name, model_id, column_map in benchmark_files:
    file_path = PROJECT_ROOT / filename

    if not file_path.exists():
        missing_files.append(filename)
        continue

    try:
        model_df = pd.read_csv(file_path)

        # Rename the model's benchmark columns to a common schema.
        missing_columns = [
            source
            for source in column_map
            if source not in model_df.columns
        ]

        if missing_columns:
            st.warning(
                f"{filename} is missing columns: "
                + ", ".join(missing_columns)
            )
            continue

        model_df = model_df.rename(columns=column_map).copy()

        if "quantization" not in model_df.columns:
            st.warning(f"{filename} has no quantization column.")
            continue

        model_df["model_name"] = model_name
        model_df["model_id"] = model_id
        model_df["source_file"] = filename

        frames.append(model_df)

    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.warning(f"Could not read {filename}: {error}")

if missing_files:
    st.warning(
        "These benchmark files were not found: "
        + ", ".join(missing_files)
        + ". Place them in the main project folder beside src."
    )

if not frames:
    st.error(
        "No valid benchmark summaries were loaded. "
        "Check that the CSV files are in the project root."
    )
    st.stop()

df = pd.concat(frames, ignore_index=True, sort=False)

numeric_columns = [
    "average_speed",
    "average_time",
    "average_peak_memory",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(
    subset=numeric_columns + ["model_name", "quantization"]
)

df = df[
    (df["average_speed"] > 0)
    & (df["average_time"] > 0)
    & (df["average_peak_memory"] > 0)
].copy()

if df.empty:
    st.error("No valid benchmark measurements remain after validation.")
    st.stop()

df["configuration"] = (
    df["model_name"] + " | " + df["quantization"].astype(str)
)

st.success(
    f"Loaded {len(df)} measured configurations across "
    f"{df['model_name'].nunique()} models."
)


# ==================================================
# MODEL FILTER
# ==================================================

model_options = ["All benchmarked models"] + sorted(
    df["model_name"].unique().tolist()
)

selected_model = st.selectbox(
    "Choose a model to compare",
    model_options,
)

if selected_model == "All benchmarked models":
    view_df = df.copy()
else:
    view_df = df[df["model_name"] == selected_model].copy()

st.subheader("Measured Benchmark Results")

display_df = view_df[
    [
        "model_name",
        "quantization",
        "average_speed",
        "average_time",
        "average_peak_memory",
    ]
].rename(
    columns={
        "model_name": "Model",
        "quantization": "Quantization",
        "average_speed": "Average speed (tokens/sec)",
        "average_time": "Average generation time (sec)",
        "average_peak_memory": "Peak allocated GPU memory (GB)",
    }
)

st.dataframe(display_df, use_container_width=True, hide_index=True)



# ==================================================
# HARDWARE PROFILES AND CONSTRAINTS
# ==================================================

st.divider()
st.header("Hardware Profile Simulator")

st.caption(
    "Explore hypothetical hardware limits. These profiles are simulations, "
    "not measurements from other laptops. Benchmark results remain from "
    "the Google Colab Tesla T4."
)

profile_options = {
    "Custom constraints": None,
    "CPU-only / low-memory example": {
        "memory": 0.5,
        "speed": 1.0,
    },
    "Low-memory GPU example": {
        "memory": 1.0,
        "speed": 5.0,
    },
    "Higher-memory GPU example": {
        "memory": 4.0,
        "speed": 10.0,
    },
}

selected_profile = st.selectbox(
    "Choose a simulated profile",
    list(profile_options.keys()),
)

profile = profile_options[selected_profile]

default_memory = profile["memory"] if profile else 1.0
default_speed = profile["speed"] if profile else 10.0

st.sidebar.header("Hardware Constraints")

max_memory = st.sidebar.slider(
    "Maximum GPU memory (GB)",
    min_value=0.25,
    max_value=16.0,
    value=float(default_memory),
    step=0.25,
    key=f"max_memory_{selected_profile}",
)

min_speed = st.sidebar.slider(
    "Minimum generation speed (tokens/sec)",
    min_value=1.0,
    max_value=30.0,
    value=float(default_speed),
    step=1.0,
    key=f"min_speed_{selected_profile}",
)

if selected_profile != "Custom constraints":
    st.info(
        f"Simulated profile: **{selected_profile}**. "
        "The selected limits are illustrative, not actual hardware readings."
    )



# ==================================================
# RECOMMENDATION ENGINE
# ==================================================

st.divider()
st.header("Benchmark-Based Recommendation")

st.caption(
    "All saved model performance measurements were collected on a "
    "Google Colab NVIDIA Tesla T4 GPU."
)

# Determine whether the selected profile represents CPU-only hardware.
cpu_only_profile = selected_profile == "CPU-only / low-memory example"

if cpu_only_profile:
    st.warning(
        "No CPU inference benchmarks are available yet. "
        "The Tesla T4 measurements cannot establish which configuration "
        "will run fastest on a CPU-only laptop."
    )
    st.info(
        "Next step: benchmark at least one quantized model using a CPU "
        "inference runtime before making a CPU-specific recommendation."
    )

else:
    eligible = view_df[
        (view_df["average_peak_memory"] <= max_memory)
        & (view_df["average_speed"] >= min_speed)
    ]

    if eligible.empty:
        st.warning(
            "No tested configuration meets both selected constraints. "
            "Increase the memory limit or lower the minimum speed."
        )
    else:
        best = eligible.sort_values(
            "average_speed",
            ascending=False,
        ).iloc[0]

        r1, r2, r3 = st.columns(3)

        r1.metric("Recommended Model", best["model_name"])
        r2.metric("Quantization", best["quantization"])
        r3.metric(
            "Measured T4 Speed",
            f'{best["average_speed"]:.2f} tokens/sec',
        )

        st.write(
            f'**Peak allocated GPU memory on T4:** '
            f'{best["average_peak_memory"]:.3f} GB'
        )

        st.subheader("Why was this configuration selected?")

        st.write(
            f'- Measured T4 speed: {best["average_speed"]:.2f} tokens/sec.'
        )
        st.write(
            f'- Measured T4 peak allocated GPU memory: '
            f'{best["average_peak_memory"]:.3f} GB.'
        )
        st.write(
            f'- It is the fastest eligible configuration among '
            f'{len(eligible)} saved benchmark configurations.'
        )

        if selected_profile != "Custom constraints":
            st.info(
                "This profile is simulated. The recommendation filters "
                "Tesla T4 benchmark results and does not predict actual "
                "performance on the simulated hardware."
            )



# ==================================================
# PERFORMANCE CHARTS
# ==================================================

st.divider()
st.header("Performance Comparison")

chart_df = view_df.set_index("configuration")

st.subheader("Generation Speed (tokens/sec)")
st.bar_chart(chart_df[["average_speed"]])

st.subheader("Peak Allocated GPU Memory (GB)")
st.bar_chart(chart_df[["average_peak_memory"]])

st.subheader("Generation Time (seconds)")
st.bar_chart(chart_df[["average_time"]])


# ==================================================
# EFFICIENCY ANALYSIS
# ==================================================

st.divider()
st.header("Efficiency Analysis")

fastest = view_df.loc[view_df["average_speed"].idxmax()]
lowest_memory = view_df.loc[
    view_df["average_peak_memory"].idxmin()
]

e1, e2 = st.columns(2)

e1.metric(
    "Fastest Measured Configuration",
    fastest["configuration"],
    f'{fastest["average_speed"]:.2f} tokens/sec',
)

e2.metric(
    "Lowest Measured GPU Memory",
    lowest_memory["configuration"],
    f'{lowest_memory["average_peak_memory"]:.3f} GB',
)


# ==================================================
# EXPORT
# ==================================================

st.divider()
st.header("Export Results")

export_df = view_df[
    [
        "model_name",
        "model_id",
        "quantization",
        "average_speed",
        "average_time",
        "average_peak_memory",
        "source_file",
    ]
].rename(
    columns={
        "model_name": "model",
        "average_speed": "average_tokens_per_second",
        "average_time": "average_generation_time_seconds",
        "average_peak_memory": "average_peak_gpu_memory_allocated_gb",
    }
)

st.download_button(
    label="Download Combined Benchmark Results (CSV)",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="combined_benchmark_results.csv",
    mime="text/csv",
)


# ==================================================
# REPEATED-RUN RELIABILITY
# ==================================================

st.divider()
st.header("Repeated-Run Reliability Analysis")

st.warning(
    "The individual 4-bit NF4 runs in the original reliability file "
    "were reconstructed and are not verified raw measurements. "
    "Do not use those individual values as experimental evidence."
)

reliability_path = PROJECT_ROOT / "reliability_summary.csv"
runs_path = PROJECT_ROOT / "reliability_runs.csv"
speed_image = PROJECT_ROOT / "repeated_speed.png"
time_image = PROJECT_ROOT / "repeated_time.png"

if reliability_path.exists():
    try:
        reliability = pd.read_csv(reliability_path)

        st.subheader("Reliability Summary")
        st.dataframe(
            reliability,
            use_container_width=True,
            hide_index=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Generation Speed Across Runs")
            if speed_image.exists():
                st.image(str(speed_image), use_container_width=True)
            else:
                st.info("repeated_speed.png was not found.")

        with col2:
            st.subheader("Generation Time Across Runs")
            if time_image.exists():
                st.image(str(time_image), use_container_width=True)
            else:
                st.info("repeated_time.png was not found.")

        if runs_path.exists():
            with st.expander("View individual reliability runs"):
                runs = pd.read_csv(runs_path)
                st.dataframe(
                    runs,
                    use_container_width=True,
                    hide_index=True,
                )

    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.warning(f"Could not load reliability files: {error}")
else:
    st.info("reliability_summary.csv was not found.")


# ==================================================
# LIMITATIONS
# ==================================================

st.divider()

st.caption(
    "Measurements were collected on a Google Colab Tesla T4 GPU. "
    "PyTorch-allocated memory is not total GPU memory consumption. "
    "Hardware guidance is an estimate; local model execution and "
    "local inference benchmarking are not yet integrated."
)
