from collections.abc import Callable, Iterable, Mapping
import os
from typing import Any
import wget
import gzip
import threading
import subprocess
from tqdm import tqdm
from data_process import load_shsmu_site


# unzip (*.pdb.gz in dir) to (new dir *.pdb)
def unzip(origin_dir: str, outdir: str):
    if os.path.exists(origin_dir):
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        files = os.listdir(origin_dir)
        for file in files:
            if ".gz" in file:
                filename = file.replace(".gz", "")
                gzip_file = gzip.GzipFile(origin_dir + file)
                with open(outdir + filename, "wb+") as f:
                    f.write(gzip_file.read())


"""
    Downloading data
"""


# download pretraining data from rcsb
class DownThread(threading.Thread):
    def __init__(self, url: str, outpath: str):
        threading.Thread.__init__(self)
        self.url = url
        self.outpath = outpath
        self.result = 0
        self.timeout = 60
        self.trytime = 3

    def download_wget_os(self, url: str, outpath: str):
        cmd = (
            "wget -c "
            + url
            + " -P "
            + outpath
            + " -t "
            + str(self.trytime)
            + " -T "
            + str(self.timeout)
        )
        status_subprocess = subprocess.call(cmd, shell=True)
        if status_subprocess == 0:
            return "success"
        elif status_subprocess == 1:
            return "linkerror"
        elif status_subprocess == 4:
            return "timeout"
        else:
            return "unknownerror"

    def run(self):
        self.result = self.download_wget_os(self.url, self.outpath)

    def get_result(self):
        return self.result

    def get_url(self):
        return self.url


def wtriting_to_log(file_name: str, contents, mode=1):
    if mode == 0:
        with open(file_name, "w") as f:
            f.writelines("")
    else:
        with open(file_name, "a") as f:
            f.writelines(contents + "\n")


def download_pretarining_data(list_path: str, outpath: str):
    max_process_number = 11
    filename_extension = ".pdb.gz"
    base_url = "https://files.rcsb.org/download/"

    with open(list_path, "r") as f:
        file_list = f.readline().strip().split(",")

    threads_group = []
    succeed_count = 0
    linkerror_count = 0
    timeout_count = 0
    other_count = 0
    total = len(file_list)
    wtriting_to_log(outpath + "timeout.log", "", mode=0)
    wtriting_to_log(outpath + "succeed.log", "", mode=0)
    wtriting_to_log(outpath + "wrang.log", "", mode=0)

    for i in tqdm(range(total)):
        url = base_url + file_list[i].strip() + filename_extension
        new_thread = DownThread(url, outpath)
        threads_group.append(new_thread)
        new_thread.setDaemon(True)
        new_thread.start()

        while len(threading.enumerate()) > max_process_number:
            pass

        for thread in threads_group:
            if thread.get_result() == "success":
                succeed_count += 1
                wtriting_to_log(outpath + "succeed.log", thread.get_url())
            elif thread.get_result() == "timeout":
                timeout_count += 1
                wtriting_to_log(outpath + "timeout.log", thread.get_url())
            elif thread.get_result() == "linkerror":
                linkerror_count += 1
                wtriting_to_log(outpath + "wrang.log", thread.get_url())
            elif thread.get_result() == "unknownerror":
                other_count += 1
                wtriting_to_log(outpath + "wrang.log", thread.get_url())

            if thread.get_result() != 0:
                threads_group.remove(thread)
                break
    threads_group.join()


def redownload_error_pretarining_data(
    url_list_timeout: str, url_list_wrang: str, outpath: str
):
    max_process_number = 11
    url_list = []
    with open(url_list_timeout, "r") as f:
        for line in f.readlines():
            url_list.append(line.strip())
    with open(url_list_wrang, "r") as f:
        for line in f.readlines():
            url_list.append(line.strip())

    threads_group = []
    succeed_count = 0
    linkerror_count = 0
    timeout_count = 0
    other_count = 0
    total = len(url_list)
    wtriting_to_log(outpath + "timeout.log", "", mode=0)
    wtriting_to_log(outpath + "succeed.log", "", mode=0)
    wtriting_to_log(outpath + "wrang.log", "", mode=0)

    for i in tqdm(range(total)):
        url = url_list[i]
        new_thread = DownThread(url, outpath)
        threads_group.append(new_thread)
        new_thread.setDaemon(True)
        new_thread.start()

        while len(threading.enumerate()) > max_process_number:
            pass

        for thread in threads_group:
            if thread.get_result() == "success":
                succeed_count += 1
                wtriting_to_log(outpath + "succeed.log", thread.get_url())
            elif thread.get_result() == "timeout":
                timeout_count += 1
                wtriting_to_log(outpath + "timeout.log", thread.get_url())
            elif thread.get_result() == "linkerror":
                linkerror_count += 1
                wtriting_to_log(outpath + "wrang.log", thread.get_url())
            elif thread.get_result() == "unknownerror":
                other_count += 1
                wtriting_to_log(outpath + "wrang.log", thread.get_url())

            if thread.get_result() != 0:
                threads_group.remove(thread)
                break
    while len(threads_group) > 0:
        for thread in threads_group:
            if thread.get_result() == "success":
                succeed_count += 1
                wtriting_to_log(outpath + "succeed.log", thread.get_url())
            elif thread.get_result() == "timeout":
                timeout_count += 1
                wtriting_to_log(outpath + "timeout.log", thread.get_url())
            elif thread.get_result() == "linkerror":
                linkerror_count += 1
                wtriting_to_log(outpath + "wrang.log", thread.get_url())
            elif thread.get_result() == "unknownerror":
                other_count += 1
                wtriting_to_log(outpath + "wrang.log", thread.get_url())
            if thread.get_result() != 0:
                threads_group.remove(thread)


# download the allosteric site form shsmu.edu.cn (feature->site)
def download_shsmu_as(allosteric_site_path: str, outpath: str):
    base_url = (
        "https://mdl.shsmu.edu.cn/ASD2023Common/static_file//site/allo_site_pdb_gz/"
    )
    filename_extension = ".pdb.gz"
    data = load_shsmu_site(allosteric_site_path)
    filename = []
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    else:
        files = os.listdir(outpath)
        for file in files:
            filename.append(file.replace(filename_extension, "").upper())
    for item in tqdm(data):
        if item["allosteric_site"] not in filename:
            wget.download(
                base_url + item["allosteric_site"] + filename_extension,
                outpath + item["allosteric_site"] + filename_extension,
            )


# download corresponding pdbs of the shsmu allosteric site
def download_rcsb(allosteric_site_path: str, outpath: str):
    base_url = "https://files.rcsb.org/download/"
    filename_extension = ".pdb.gz"
    data = load_shsmu_site(allosteric_site_path)
    filename = []
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    else:
        files = os.listdir(outpath)
        for file in files:
            filename.append(file.replace(filename_extension, "").upper())

    for item in tqdm(data):
        if item["allosteric_pdb"] not in filename:
            wget.download(
                base_url + item["allosteric_pdb"] + filename_extension,
                outpath + item["allosteric_pdb"] + filename_extension,
            )


# download_shsmu_as(allosteric_site_path='../data/allosteric_site_shsmu.json', outpath='../data/allosteric_site/')
# download_rcsb(allosteric_site_path='../data/allosteric_site_shsmu.json', outpath='../data/rcsb/')

# unzip('../data/rcsb/', '../data/shsmu_allosteric_site/rscb_pdb/')
# download_pretarining_data('../data/pretrain/list_file_protein_xray_max3A_1.txt', '../data/pretrain/')
redownload_error_pretarining_data(
    "/mnt/g/Little-LL/pertrain/timeout.log",
    "/mnt/g/Little-LL/pertrain/wrang.log",
    "/mnt/g/Little-LL/pertrain/",
)
