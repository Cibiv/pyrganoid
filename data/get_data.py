import urllib.request
import hashlib
from pathlib import Path

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


def download_lt47():
    path = Path('LT47.tsv.gz')

    if path.exists():
        if verify_hash(path, LT47_MD5):
            print('LT47.tsv.gz already downloaded')
            return

    downloading = True

    print('Downloading GSE151383 (LT47)')

    with urllib.request.urlopen(LT47_URL) as f:
        with open(path, mode='wb') as out:
            while downloading:
                b = f.read(BLOCKSIZE)
                if len(b) == 0:
                    downloading = False
                else:
                    out.write(b)

    print('Downloaded  GSE151383 (LT47)')
    print('Verifying MD5 hash of LT47.tsv.gz')
    if not verify_hash(path, LT47_MD5):
        print('MD5 hash of LT47.tsv.gz failed')
    return

def download_lt48():
    path = Path('LT48.csv.gz')

    if path.exists():
        if verify_hash(path, LT48_MD5):
            print(f'{path} already downloaded')
            return

    downloading = True

    print('Downloading GSM6599035 (LT48)')

    with urllib.request.urlopen(LT48_URL) as f:
        with open(path, mode='wb') as out:
            while downloading:
                b = f.read(BLOCKSIZE)
                if len(b) == 0:
                    downloading = False
                else:
                    out.write(b)

    print(f'Downloaded  GSM6599035 ({path})')
    print('Verifying MD5 hash of LT47.tsv.gz')
    if not verify_hash(path, LT48_MD5):
        print(f'MD5 hash of {path} failed')
    return


def main():
    download_lt47()
    download_lt48()

if __name__ == '__main__':
    main()
