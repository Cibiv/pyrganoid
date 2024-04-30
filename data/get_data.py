import urllib.request
import hashlib
from pathlib import Path
import gzip
import sys

try:
    import polars as pl
    import polars.datatypes as dt
    polars_available = True
    LT48_schema = {'sample': dt.Categorical(), 'lid': dt.Utf8(), 'nreads': dt.Int64(),
                   'category': dt.Utf8(), 'day': dt.Utf8(), 'replicate': dt.Int32(), 'effect': dt.Utf8()}

    LT47_2D_schema = {'Sample': dt.Categorical(), 'LID': dt.Utf8(), 'Library': dt.Categorical(), 'nLineages': dt.Int64()}
    LT47_2D_preprocess = [pl.col("Sample").alias("sample"), pl.col("LID").alias("lid"), pl.col("Library").alias("lib"), pl.col("nLineages").alias("nreads")]

except ModuleNotFoundError:
    polars_available = False

LT47_URL = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE151383&format=file&file=GSE151383%5FLT47%2Etsv%2Egz'
LT47_MD5 = '02d1c7accd95ac3e54137b1f5fb52f5c'

LT48_URL = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM6599035&format=file&file=GSM6599035%5Flt48%2Ecsv%2Egz'
LT48_MD5 = 'f2b2a484e6271a5f279e18f3f1089631'

BLOCKSIZE = 16384

def verify_hash(path, hash):
    reading = True
    with open(path, mode='rb') as f:
        h = hashlib.md5(usedforsecurity=False)
        while reading:
            data = f.read(BLOCKSIZE)
            if len(data) == 0:
                reading = False
            else:
                h.update(data)

    return hash == h.hexdigest()

def download(name, url, hash, path):
    path = Path(path)
    if path.exists():
        if verify_hash(path, hash):
            print(f'{name} ({path}) already downloaded')
            return

    downloading = True

    print(f'Downloading {name} ({path})')
    with urllib.request.urlopen(url) as f:
        with open(path, mode='wb') as out:
            while downloading:
                b = f.read(BLOCKSIZE)
                if len(b) == 0:
                    downloading = False
                else:
                    out.write(b)

    print(f'Downloaded  {name} ({path})')
    print(f'Verifying MD5 hash of {path}')
    if not verify_hash(path, hash):
        print(f'ERROR: MD5 hash of {path} is incorrect')
        sys.exit(-1)
    print(f'Verified  MD5 hash of {path}')

def decompress(name, input_path, output_path):
    print(f'Decompressing {name} ({input_path})')
    decompressing = True
    with gzip.open(input_path, mode='rb') as f:
        with open(output_path, mode='wb') as out:
            while decompressing:
                b = f.read(BLOCKSIZE)
                if len(b) == 0:
                    decompressing = False
                else:
                    out.write(b)
    print(f'Decompressed  {name} ({input_path})')

def to_parquet(name, input_path, output_path, separator=',', schema=None, preprocess=None):
    import polars as pl
    print(f'Converting to parquet {name} ({input_path})')
    df = pl.read_csv(input_path, separator=separator, schema=schema)
    if preprocess:
        df = df.select(preprocess)
    df.write_parquet(output_path)
    print(f'Converted to parquet {name} ({output_path})')


def main():
    print('Checking GSE151383')
    compressed_path = Path('LT47.tsv.gz')
    decompressed_path = Path('LT47.tsv')
    data_path = Path('LT47.parquet')

    if not data_path.exists():
        if not decompressed_path.exists():
            if not compressed_path.exists():
                download('GSE151383', LT47_URL, LT47_MD5, compressed_path)
            decompress('GSE151383', compressed_path, decompressed_path)
            compressed_path.unlink()
        to_parquet('GSE151383', decompressed_path, data_path, separator='\t')
        decompressed_path.unlink()
    print('Done GSE151383')
    print()


    print('Checking GSM6599035')
    compressed_path = Path('LT48.csv.gz')
    decompressed_path = Path('LT48.csv')
    data_path = Path('LT48.parquet')

    if not data_path.exists():
        if not decompressed_path.exists():
            if not compressed_path.exists():
                download('GSM6599035', LT48_URL, LT48_MD5, compressed_path)
            decompress('GSM6599035', compressed_path, decompressed_path)
            compressed_path.unlink()
        to_parquet('GSM6599035', decompressed_path, data_path, schema=LT48_schema)
        decompressed_path.unlink()

    print('Done GSM6599035')
    print()

    # the processed LT47_2D dataset is unavailable from GEO, but can be reproduced from the raw data GSE151384
    # For convenience, the LT47_2D.tsv.gz is commited to this git repository
    print('Checking LT47_2D')
    data_path = Path('LT47_2D.parquet')
    if not data_path.exists():
        print(f'{data_path} does not exist, aborting.')
        sys.exit(-1)
    print('Done LT47_2D')

    # the processed LT47_3D dataset from GEO (GSE151383) does not contain the lib1/lib2 information, but can be reproduced from the raw data
    # For convenience, the LT47_3D.parquet is commited to this git repository
    print('Checking LT47_3D')
    data_path = Path('LT47_3D.parquet')
    if not data_path.exists():
        print(f'{data_path} does not exist, aborting.')
        sys.exit(-1)
    print('Done LT47_3D')

if __name__ == '__main__':
    main()
