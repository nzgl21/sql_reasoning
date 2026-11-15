import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

import sqlglot

from config import PROJECT_PATH


def extract_sql_from_completion(text: str) -> str:
    """Извлекает SQL-запрос из текста ответа модели.
    
    Поддерживает:
    1. Блоки ```sql ... ```
    2. Разделение тегами <think>...</think>
    3. Обычный SQL-текст
    """
    # 1. Поиск блока ```sql ... ```
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2. Если блока нет, берем текст после закрывающего тега </think>
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # 3. Убираем случайные другие кодовые блоки ``` ... ```
    text = re.sub(r"^```\s*|\s*```$", "", text.strip())
    return text.strip()


def execute_sql_safely(db_path: Path | str, sql: str, timeout_sec: float = 2.0) -> tuple[set | None, str | None]:
    """Безопасно выполняет SQL-запрос в базе данных SQLite.
    
    - Режим: Read-Only (mode=ro) для защиты данных.
    - Таймаут: прерывание долгих запросов (декартовых произведений).
    - Возвращает (set_of_rows, error_message).
    """
    db_path = str(db_path)

    def _run():
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            # Приводим к множеству кортежей для независимости от порядка строк
            return set(cursor.fetchall())

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run)
        try:
            return future.result(timeout=timeout_sec), None
        except TimeoutError:
            return None, "Query Timeout"
        except Exception as e:
            return None, str(e)


def execution_reward(completions: list[str], query: list[str], db_id: list[str], **kwargs) -> list[float]:
    """Главная функция награды: проверка корректности выполнения SQL на реальной базе.
    
    Возвращает:
        1.0 - если запрос выполнился без ошибок и результат совпал с эталонным.
        0.0 - если произошла ошибка или результат не совпал.
    """
    rewards = []
    database_dir = PROJECT_PATH / "data" / "raw" / "spider_data" / "database"

    for completion, gold_sql, d_id in zip(completions, query, db_id):
        pred_sql = extract_sql_from_completion(completion)
        db_path = database_dir / d_id / f"{d_id}.sqlite"

        if not db_path.exists():
            rewards.append(0.0)
            continue

        # 1. Выполняем эталонный запрос
        gold_res, gold_err = execute_sql_safely(db_path, gold_sql)
        if gold_err is not None:
            rewards.append(0.0)
            continue

        # 2. Выполняем запрос модели
        pred_res, pred_err = execute_sql_safely(db_path, pred_sql)

        # 3. Сравниваем результаты
        if pred_err is None and pred_res == gold_res:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """Награда за соблюдение формата рассуждений (<think>...</think> и ```sql...```).
    
    Возвращает 0.2 при соблюдении формата, иначе 0.0.
    """
    rewards = []
    pattern = r"<think>.*?</think>\s*```sql.*?```"
    for completion in completions:
        if re.search(pattern, completion.strip(), re.DOTALL):
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards


def syntax_reward(completions: list[str], **kwargs) -> list[float]:
    """Награда за синтаксическую валидность SQL-запроса через sqlglot.
    
    Возвращает 0.1, если SQL хотя бы синтаксически корректен.
    """
    rewards = []
    for completion in completions:
        pred_sql = extract_sql_from_completion(completion)
        try:
            sqlglot.parse_one(pred_sql)
            rewards.append(0.1)
        except Exception:
            rewards.append(0.0)
    return rewards
