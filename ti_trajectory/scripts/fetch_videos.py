# -*- coding: utf-8 -*-
"""
从 Figshare 远程 ZIP（TI 路口 video.zip, file id=57423067）流式提取三个视频。
ZIP 内条目为 deflate(method=8) 压缩，且 mp4 的 moov atom 位于文件末尾，
因此必须顺序解压整个条目才能得到可解码的 mp4。

用法：
    python fetch_videos.py                 # 提取全部三个
    python fetch_videos.py video-1         # 只提取 video-1
    python fetch_videos.py --frames 1000   # 提取后用 ffmpeg 抽前 N 帧
"""
import argparse
import io
import os
import struct
import sys
import time
import urllib.request
import zipfile

FIGSHARE_FILE_ID = 57423067
BASE_URL = f"https://ndownloader.figshare.com/files/{FIGSHARE_FILE_ID}"
UA = {"User-Agent": "Mozilla/5.0"}
CHUNK = 4 * 1024 * 1024


def http_size(url):
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Length"])


class HttpFile(io.RawIOBase):
    """只读、支持 seek 的远程文件，用于喂给 zipfile。"""

    def __init__(self, url, size):
        self.url = url
        self.size = size
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0:
            return b""
        start = min(self.pos, self.size)
        end = min(self.pos + n, self.size) - 1
        if end < start:
            return b""
        req = urllib.request.Request(
            self.url, headers={**UA, "Range": f"bytes={start}-{end}"}
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = resp.read()
        self.pos += len(data)
        return data


def extract(targets, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    size = http_size(BASE_URL)
    print(f"[remote] zip size = {size:,} bytes", flush=True)
    zf = zipfile.ZipFile(HttpFile(BASE_URL, size))
    names = {i.filename: i for i in zf.infolist()}
    for key in targets:
        member = f"video/20250529_{key}.mp4"
        if member not in names:
            print(f"[skip] {member} 不在包内", flush=True)
            continue
        dst = os.path.join(out_dir, f"20250529_{key}.mp4")
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            print(f"[done] 已存在 {dst}", flush=True)
            continue
        info = names[member]
        t0 = time.time()
        got = 0
        with zf.open(info) as src, open(dst, "wb") as out:
            while True:
                buf = src.read(CHUNK)
                if not buf:
                    break
                out.write(buf)
                got += len(buf)
                el = time.time() - t0
                print(
                    f"  [{key}] {got/1e9:.2f}/{info.file_size/1e9:.2f} GB "
                    f"{got/1e6/max(el,1e-9):.1f} MB/s",
                    end="\r",
                    flush=True,
                )
        print(f"\n[ok] {dst}  {os.path.getsize(dst):,} bytes  "
              f"{time.time()-t0:.0f}s", flush=True)


def grab_frames(mp4, n_frames, out_dir, fps_note=""):
    """用 ffmpeg 抽前 n_frames 帧为 PNG。"""
    import imageio_ffmpeg
    import subprocess

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-i", mp4,
        "-frames:v", str(n_frames),
        "-q:v", "2",
        os.path.join(out_dir, "f_%05d.jpg"),
    ]
    print(f"[ffmpeg] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    print(f"[ok] frames -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", default=["video-1", "video-2", "video-3"])
    ap.add_argument("--out", default="raw/video")
    ap.add_argument("--frames", type=int, default=0,
                    help=">0 时抽取每个视频的前 N 帧为 jpg")
    ap.add_argument("--frames-dir", default="raw/frames")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, args.out)
    extract(args.keys, out_dir)

    if args.frames > 0:
        for k in args.keys:
            mp4 = os.path.join(out_dir, f"20250529_{k}.mp4")
            if os.path.exists(mp4):
                grab_frames(mp4, args.frames, os.path.join(root, args.frames_dir, k))


if __name__ == "__main__":
    sys.exit(main())
