import requests
import pandas as pd
import os
import time
import numpy as np

BLOCK = 512
STATE_FILE = "scan_state.txt"
INDEX_FILE = "tar_index.csv"


def parse_tar_header(block: bytes):
    if block == b"\0" * BLOCK:
        return None

    name = block[0:100].rstrip(b"\0").decode("utf-8", errors="ignore")
    size_oct = block[124:136].rstrip(b"\0").decode("ascii", errors="ignore")

    try:
        size = int(size_oct.strip() or "0", 8)
    except ValueError:
        return None

    return name, size


def load_offset():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return int(f.read().strip())
    return 0


def save_offset(offset: int):
    with open(STATE_FILE, "w") as f:
        f.write(str(offset))


def append_records(records):
    df = pd.DataFrame(records)
    header = not os.path.exists(INDEX_FILE)
    df.to_csv(INDEX_FILE, mode="a", header=header, index=False)


def scan_tar_http_chunked(
    url: str,
    max_headers_per_run=200,
    timeout=10,
    sleep_between=0.1,
):
    """
    Scan up to `max_headers_per_run` TAR entries, then exit cleanly.
    Can be resumed safely.
    """
    session = requests.Session()
    offset = load_offset()
    records = []
    scanned = 0
    next_call_needed = True

    print(f"Resuming scan at byte offset {offset}")

    while scanned < max_headers_per_run:
        try:
            headers = {"Range": f"bytes={offset}-{offset + BLOCK - 1}"}
            r = session.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            block = r.content

        except requests.RequestException as e:
            print(f"Network error at offset {offset}: {e}")
            break

        if block == b"\0" * BLOCK:
            print("End of TAR archive reached.")
            save_offset(offset)
            next_call_needed = False
            break

        parsed = parse_tar_header(block)
        if parsed is None:
            print("Invalid TAR header encountered.")
            save_offset(offset)
            next_call_needed = False
            break

        name, size = parsed

        data_offset = offset + BLOCK
        records.append(
            {
                "filename": name,
                "data_offset": data_offset,
                "data_size": size,
            }
        )

        padded = ((size + BLOCK - 1) // BLOCK) * BLOCK
        offset += BLOCK + padded
        scanned += 1
        if scanned % 20 == 0:
            print(scanned)

        time.sleep(sleep_between)

    if records:
        append_records(records)
        save_offset(offset)

    print(f"Scanned {scanned} entries this run.")
    print(f"Next offset: {offset}")
    return next_call_needed



def download_file(url: str, df: pd.DataFrame, filename: str, out_path: str | None = None):
    """
    Download a single file from the TAR using byte ranges.
    """
    if out_path is None:
        out_path = f"./{filename}.hdf5"
    row = df[df["filename"] == f"emg2pose_data/{filename}.hdf5"]
    if row.empty:
        raise ValueError("File not found in index")

    offset = int(row.iloc[0]["data_offset"])
    size = int(row.iloc[0]["data_size"])

    headers = {"Range": f"bytes={offset}-{offset + size - 1}"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(r.content)


url = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar"
df = pd.read_csv("tar_index.csv")

'''
To downlaod the files,
	1. See the filenames in the downloaed_hdf5s foler
	2. For each filename, run this script and issue this line:
		download_file(url, df, [the filename])
	Note that the filename should NOT contain the ".hdf5" suffix.
'''
