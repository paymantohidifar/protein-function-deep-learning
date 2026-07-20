import os
from tqdm import tqdm
import requests
import tarfile


def download_data(base_url: str, filename:str, output_path: str) -> None:
    
    # Define browser-like headers to prevent 403/406 blocking by the server
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    url = base_url + filename

    print(f"Initiating stream from: {url}")
    
    # Wrap in context manager to guarantee connection closing on exit/failure
    with requests.get(url, stream=True, headers=headers, timeout=60) as response:
    
        response.raise_for_status() # Gracefully capture HTTP errors
        total_size = int(response.headers.get('content-length', 0)) # Track total size for the progress bar
        chunk_size = 8192
        
        with open(output_path, "wb") as file, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))


def extract_tar_archive(archive_path: str, target_dir: str) -> None:
    """Safely extract a .tgz or .tar.gz archive to a target directory."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    print(f"Extracting {archive_path} to {target_dir}...")
    
    # Open the archive using 'r:gz' mode (read with gzip decompression)
    with tarfile.open(archive_path, "r:gz") as tar:
        # Enforce security guardrails against Directory Traversal exploits (Tar損loit)
        # By ensuring paths remain strictly inside the target destination directory
        tar.extractall(path=target_dir, filter='data')
        
    print("Extraction complete.")