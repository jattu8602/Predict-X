"""Fetch and verify the two NASA FD001 files needed by the Render image."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from tempfile import TemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
CHECKSUMS = {
    "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
}


def main() -> None:
    with TemporaryFile() as downloaded, urlopen(URL, timeout=60) as response:
        copyfileobj(response, downloaded)
        with ZipFile(downloaded) as outer:
            nested = next(name for name in outer.namelist() if PurePosixPath(name).name == "CMAPSSData.zip")
            with ZipFile(BytesIO(outer.read(nested))) as archive:
                files = {filename: archive.read(filename) for filename in CHECKSUMS}

    for filename, contents in files.items():
        if sha256(contents).hexdigest() != CHECKSUMS[filename]:
            raise ValueError(f"NASA download checksum did not match for {filename}.")

    raw = Path("data/raw")
    raw.mkdir(parents=True, exist_ok=True)
    for filename, contents in files.items():
        (raw / filename).write_bytes(contents)
    print("Downloaded and verified the official NASA FD001 training and demo readings.")


if __name__ == "__main__":
    main()
