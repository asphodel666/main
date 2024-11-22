
import hashlib
import sys
from multiprocessing import Pool, cpu_count
import time

def compute_hash(password, encoding, hash_function):
    """Вычисляет хеш для заданного пароля."""
    hash_func = getattr(hashlib, hash_function.lower())  # Приводим к нижнему регистру
    return hash_func(password.encode(encoding)).hexdigest()

def process_chunk(chunk, hash_list, encoding, hash_function):
    """Обрабатывает часть данных, возвращает совпадения."""
    matches = []
    #print(hash_list)
    for password in chunk:
        password = password.strip()  # Убираем лишние пробелы и символы новой строки
        hash_value = compute_hash(password, encoding, hash_function)
        #print(hash_value)
        # Приводим хеши к нижнему регистру и убираем лишние пробелы
        if any(hash_value.strip().lower() == h.strip().lower() for h in hash_list):
            matches.append(f"{password}:{hash_value}")
    return matches

def split_list(data, num_chunks):
    """Разделяет данные на равные части."""
    avg = len(data) // num_chunks
    return [data[i * avg:(i + 1) * avg] for i in range(num_chunks)]

def crack_passwords(wordlist_file, encoding, hash_function, hashlist_file):
    """Основная функция для восстановления паролей."""
    # Загрузка словаря паролей
    with open(wordlist_file, 'r', encoding=encoding) as f:
        passwords = f.readlines()

    # Загрузка хешей
    with open(hashlist_file, 'r', encoding='utf-8') as f:
        hash_list = set(f.read().splitlines())

    # Печать хешей для проверки
    #print("Loaded hashes:")
    #for h in hash_list:
    #   print(h)

    # Распараллеливание по ядрам
    num_cores = cpu_count()
    chunks = split_list(passwords, num_cores)

    start_time = time.time()

    with Pool(processes=num_cores) as pool:
        results = pool.starmap(process_chunk, [(chunk, hash_list, encoding, hash_function) for chunk in chunks])

    end_time = time.time()

    # Объединяем результаты из всех процессов
    matches = [item for sublist in results for item in sublist]

    # Вывод результатов
    if matches:
        for match in matches:
            print(match)
    else:
        print("No matches found.")

    print(f"\nSpeed: {len(passwords) / (end_time - start_time):.2f} candidates/second")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python crack.py <wordlist_file> <encoding> <hash_function> <hashlist_file>")
        sys.exit(1)

    wordlist_file = sys.argv[1]
    encoding = sys.argv[2]
    hash_function = sys.argv[3]
    hashlist_file = sys.argv[4]

    crack_passwords(wordlist_file, encoding, hash_function, hashlist_file)
