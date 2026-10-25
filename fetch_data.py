"""Download only the public modelling table from the cited Figshare version."""
from pathlib import Path
from urllib.request import urlopen
import hashlib

URL = "https://ndownloader.figshare.com/files/56221685"
MD5 = "861f16a5411ba3a69211067bad0007b9"
dest = Path(__file__).resolve().parent / "data-figshare.csv"
if dest.exists():
    payload = dest.read_bytes()
    if hashlib.md5(payload).hexdigest() != MD5:
        raise RuntimeError("Existing data differs from the specified Figshare file; no overwrite.")
else:
    payload = urlopen(URL, timeout=120).read()
    if hashlib.md5(payload).hexdigest() != MD5:
        raise RuntimeError("Figshare download checksum mismatch; no file written.")
    dest.write_bytes(payload)
print(f"Verified {dest.name}: {len(payload)} bytes")
