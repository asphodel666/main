from Crypto.Hash import MD4
import random
import sys

def hash_password(password, hash_function):
    """Возвращает хэш пароля, используя заданную хэш-функцию."""
    if hash_function == "md4":
        hash_obj = MD4.new()
    else:
        try:
            hash_obj = hashlib.new(hash_function)
        except ValueError:
            print(f"Ошибка: Хэш-функция {hash_function} не поддерживается.")
            sys.exit(1)
    
    hash_obj.update(password.encode('utf-8'))
    return hash_obj.hexdigest()

def generate_hashes(input_file, encoding, hash_function, num_hashes, output_file):
    """Генерирует хэши для списка паролей."""
    try:
        with open(input_file, 'r', encoding=encoding) as f:
            passwords = f.read().splitlines()
    except FileNotFoundError:
        print(f"Ошибка: Файл {input_file} не найден.")
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"Ошибка: Кодировка {encoding} не поддерживается для файла {input_file}.")
        sys.exit(1)

    # Генерируем хэши для паролей
    hashes = []
    for password in passwords:
        hashes.append(hash_password(password, hash_function))
    
    # Добавляем псевдослучайные хэши, если нужно
    while len(hashes) < num_hashes:
        random_password = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=12))
        hashes.append(hash_password(random_password, hash_function))
    
    # Обрезаем список до указанного количества хэшей
    hashes = hashes[:num_hashes]

    # Записываем хэши в выходной файл
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(hashes))

    print(f"Файл {output_file} с хэш-значениями успешно создан.")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Использование: python3 gen.py <input_file> <encoding> <hash_function> <num_hashes> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    encoding = sys.argv[2]
    hash_function = sys.argv[3].lower()
    try:
        num_hashes = int(sys.argv[4])
    except ValueError:
        print("Ошибка: Количество хэш-значений должно быть числом.")
        sys.exit(1)
    output_file = sys.argv[5]
    
    generate_hashes(input_file, encoding, hash_function, num_hashes, output_file)
