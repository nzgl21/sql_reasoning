import sqlite3
from pathlib import Path


def get_db_ddl(db_id: str, db_dir: Path) -> str:
    """
    Подключается к SQLite базе {db_id}.sqlite и забирает чистый DDL всех таблиц.
    """
    # Файл .sqlite гарантированно есть у всех 166 баз Spider:
    db_path = db_dir / db_id / f"{db_id}.sqlite"
    
    if not db_path.exists():
        raise FileNotFoundError(f"Файл базы {db_path} не найден")

    # Подключаемся к базе SQLite
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Запрашиваем DDL всех таблиц (исключая внутренние системные таблицы sqlite_*)
    query = """
        SELECT sql 
        FROM sqlite_master 
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%';
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    # Собираем чистые DDL statements
    ddl_statements = [row[0].strip() for row in rows if row[0]]
    
    # Объединяем через точку с запятой и перенос строки
    return ";\n\n".join(ddl_statements) + (";" if ddl_statements else "")