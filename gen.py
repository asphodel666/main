import random
import hashlib
import sys

def generate_hashes(input_file, encoding, hash_function, count, output_file):
    supported_hashes = {
        'MD4': 'md4',
        'MD5': 'md5',
        'SHA-1': 'sha1',
        'SHA-256': 'sha256',
        'SHA-512': 'sha512',
    }

    if hash_function not in supported_hashes:
        print(f"Unsupported hash function: {hash_function}")
        return

    try:
        with open(input_file, 'r', encoding=encoding) as f:
            passwords = f.read().splitlines()
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    hashes = []
    hash_func = getattr(hashlib, supported_hashes[hash_function])

    for password in passwords:
        hash_value = hash_func(password.encode(encoding)).hexdigest()
        hashes.append(hash_value)

    while len(hashes) < count:
        # Генерация случайного пароля
        length = random.randint(1, 15)  # Длина от 1 до 15
        random_password = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=length))
        random_hash = hash_func(random_password.encode(encoding)).hexdigest()
        hashes.append(random_hash)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(hashes))
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python gen.py <input_file> <encoding> <hash_function> <count> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    encoding = sys.argv[2]
    hash_function = sys.argv[3]
    count = int(sys.argv[4])
    output_file = sys.argv[5]

    generate_hashes(input_file, encoding, hash_function, count, output_file)
