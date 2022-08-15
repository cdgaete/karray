

def _format_bytes(size: int):
    """
    Format bytes to human readable format.

    Thanks to: https://stackoverflow.com/a/70211279
    """
    power_labels = {40: "TB", 30: "GB", 20: "MB", 10: "KB"}
    for power, label in power_labels.items():
        if size >= 2 ** power:
            approx_size = size / 2 ** power
            return f"{approx_size:.1f} {label}"
    return f"{size} bytes"