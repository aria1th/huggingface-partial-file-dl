import requests
import os
from functools import cache
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from tenacity import retry, stop_after_attempt, wait_exponential
from pandas import read_parquet

def create_headers(start: int, end: int):
    headers = {'Range': f'bytes={start}-{end}','Authorization': "Bearer "+ os.environ['HF_TOKEN']}
    return headers

def create_auth_headers():
    headers = {'Authorization': "Bearer "+ os.environ['HF_TOKEN']}
    return headers
@cache
def get_access_json(repo_type: str, repository: str, arcname: str):
    if arcname.endswith('.tar'):
        arcname = arcname.replace('.tar', '.json')
    access_path = f"https://huggingface.co/{repo_type}s/{repository}/resolve/main/{arcname}"
    response = requests.get(access_path, headers=create_auth_headers(), timeout=10)
    response.raise_for_status()
    # into json
    access_json = json.loads(response.text)
    return access_json

def get_files_parquet(repo_name, save_path, repo_type:str = "dataset"):
    try:
        df = read_parquet(os.path.join(save_path, 'files.parquet'))
        return df
    except Exception as e:
        if not isinstance(e, FileNotFoundError):
            print(e)
    print('Downloading files.parquet in', save_path)
    header = create_auth_headers()
    access_path = f"https://huggingface.co/{repo_type}s/{repo_name}/resolve/main/files.parquet"
    response = requests.get(access_path, headers=header, timeout=10)
    response.raise_for_status()
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, 'files.parquet') if os.path.isdir(save_path) else save_path
    with open(save_path, 'wb') as f:
        f.write(response.content)
    df = read_parquet(save_path)
    return df

def get_unique_arcfiles(repo_name, repo_type:str = "dataset", save_path='.'):
    df = get_files_parquet(repo_name, save_path, repo_type)
    bool_array = df['mimetype'].str.contains('video/') == False #example filter
    filtered_df = df[bool_array]
    unique_package_files = filtered_df['package_file'].unique()
    return unique_package_files

def get_access_pointer(repo_type: str, repository: str, arcname: str, filename: str):
    access_json = get_access_json(repo_type, repository, arcname)
    pointer = access_json['files'][filename]
    start = pointer['offset']
    end = start + pointer['size'] - 1
    access_path = f"https://huggingface.co/{repo_type}s/{repository}/resolve/main/{arcname}"
    header = create_headers(start, end)
    return access_path, header, pointer['size']

def filtering_access_pointer(repo_type: str, repository: str, arcname: str, func: callable):
    access_json = get_access_json(repo_type, repository, arcname)
    with tqdm(total=len(access_json['files']), desc='Filtering') as pbar:
        for filename, pointer in access_json['files'].items():
            if func(filename):
                start = pointer['offset']
                end = start + pointer['size'] - 1
                access_path = f"https://huggingface.co/{repo_type}s/{repository}/resolve/main/{arcname}"
                header = create_headers(start, end)
                yield access_path, header, filename, pointer['size']
            pbar.update(1)
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_file(access_path, headers, filename, save_path='.', check_size:int=None):
    if not os.path.isabs(save_path):
        save_path = os.path.join(os.getcwd(), save_path)
    if not os.path.isabs(filename):
        filename = os.path.join(save_path, filename)
    if os.path.exists(filename):
        if check_size:
            if os.path.getsize(filename) == check_size:
                return filename
            else:
                os.remove(filename)
        else:
            return filename
    response = requests.get(access_path, headers=headers, timeout=10)
    response.raise_for_status()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        f.write(response.content)
    return filename

def get_files_archive(repo_type: str, repo_id: str, arc_name: str, filter_func: callable, save_path='.', threads=1):
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for access_path, header, filename, filesize in filtering_access_pointer(repo_type, repo_id, arc_name, filter_func):
            #et_file(access_path, header, filename, save_path)
            futures.append(executor.submit(get_file, access_path, header, filename, save_path, filesize))
        for future in tqdm(futures, desc='Downloading'):
            future.result()

def get_files_from_archives(repo_type: str, repo_id: str, arc_names: list, filter_func: callable, save_path='.', threads=1):
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        with tqdm(total=len(arc_names), desc='Archives',position=0) as pbar:
            for arc_name in arc_names:
                for access_path, header, filename, filesize in filtering_access_pointer(repo_type, repo_id, arc_name, filter_func):
                    #et_file(access_path, header, filename, save_path)
                    futures.append(executor.submit(get_file, access_path, header, filename, save_path, filesize))
                pbar.update(1)
            for future in tqdm(futures, desc='Downloading'):
                future.result()
if __name__ == "__main__":
    repo_type = 'dataset'
    repo_id='deepghs/fancaps_full'
    arc_name='images/0000.tar'
    parquet_path="."
    # set os.environ['HF_TOKEN'] before running
    # url, header = get_access_pointer(repo_type, repo_id, arc_name, file_name)
    # get_file(url, header, file_name)
    get_files_from_archives(
        repo_type=repo_type,
        repo_id=repo_id,
        arc_names=get_unique_arcfiles(repo_id, repo_type, parquet_path),
        filter_func=lambda x: x.rsplit('.', 1)[-1] in ['jpg', 'jpeg', 'png', 'webp'],
        save_path='example',
        threads=24
    )
