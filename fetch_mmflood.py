"""Read selected files from the official MMFlood ZIP using HTTP ranges."""
import argparse
import bz2
from concurrent.futures import ThreadPoolExecutor
import io
from inflate64 import Inflater
import struct
import zlib
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import LZMADecompressor, ZipFile

URL = "https://zenodo.org/records/6534637/files/mmflood.zip?download=1"


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.offset = 0
        with self.request("-1") as response:
            self.size = int(response.headers["Content-Range"].split("/")[-1])

    def request(self, byte_range):
        url = URL + "&chunk=" + byte_range
        response = urlopen(Request(url, headers={"Range": f"bytes={byte_range}"}), timeout=35)
        if response.status != 206:
            response.close()
            raise OSError("The source did not honor the byte range.")
        return response

    def seekable(self):
        return True

    def tell(self):
        return self.offset

    def seek(self, offset, whence=0):
        self.offset = (0, self.offset, self.size)[whence] + offset
        return self.offset

    def read(self, size=-1):
        end = min(self.size, self.offset + size) if size >= 0 else self.size
        if end <= self.offset:
            return b""
        with self.request(f"{self.offset}-{end - 1}") as response:
            data = response.read()
        self.offset += len(data)
        return data


EVENTS = ("EMSR107", "EMSR184", "EMSR280", "EMSR265")


def fetch(directory):
    directory.mkdir(parents=True, exist_ok=True)
    remote = RemoteZip()
    with ZipFile(remote) as archive:
        entries = {item.filename: item for item in archive.infolist()}
        jobs = []
        for event in EVENTS:
            matches = sorted(name for name in entries if name.startswith("mmflood/" + event + "-")
                             and "/mask/" in name and name.endswith(".tif")
                             and name.replace("/mask/", "/s1_raw/") in entries)
            if not matches:
                raise ValueError(f"No paired data for {event}")
            # Spread samples through each event's sorted chip names.
            chosen = [matches[i * len(matches) // min(15, len(matches))] for i in range(min(15, len(matches)))]
            for name in chosen:
                jobs.extend((entries[name], entries[name.replace("/mask/", "/s1_raw/")]))
        print(f"Downloading {len(jobs)//2} paired satellite chips", flush=True)

    def extract(item):
        tag = "Label" if "/mask/" in item.filename else "Radar"
        dest = directory / (Path(item.filename).stem + "_" + tag + ".tif")
        if dest.exists():
            return
        start = item.header_offset
        # Read one range containing the local file header and compressed payload.
        end = min(remote.size - 1, start + 30 + len(item.filename.encode()) + 65535 + item.compress_size)
        with remote.request(f"{start}-{end}") as response:
            payload = response.read()
        if payload[:4] != b"PK\x03\x04":
            raise OSError("Invalid ZIP header")
        name_size, extra_size = struct.unpack_from("<HH", payload, 26)
        offset = 30 + name_size + extra_size
        content = payload[offset:offset + item.compress_size]
        if item.compress_type == 8:
            content = zlib.decompress(content, -15)
        elif item.compress_type == 9:
            content = Inflater().inflate(content)
        elif item.compress_type == 12:
            content = bz2.decompress(content)
        elif item.compress_type == 14:
            content = LZMADecompressor().decompress(content)
        elif item.compress_type != 0:
            raise OSError(f"Unsupported ZIP compression: {item.compress_type}")
        if zlib.crc32(content) != item.CRC:
            raise OSError("ZIP checksum mismatch")
        dest.write_bytes(content)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(extract, jobs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    fetch(parser.parse_args().directory)
