
from hardware_profiler import get_hardware_summary

for key, value in get_hardware_summary().items():
    print(f"{key}: {value}")
