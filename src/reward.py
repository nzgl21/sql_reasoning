import itertools
from pathlib import Path
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any

import sqlglot

from config import PROJECT_PATH


def to_text(completion: Any) -> str:
    """Извлекает строку, если ответ пришел в чат-формате [{'role': ..., 'content': ...}]."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if len(completion) > 0 and isinstance(completion[0], dict) and "content" in completion[0]:
            return completion[0]["content"]
        return " ".join(str(c) for c in completion)
    if isinstance(completion, dict) and "content" in completion:
        return completion["content"]
    return str(completion)


def extract_sql_from_completion(text: str) -> str:
    """Извлекает SQL-запрос из текста ответа модели."""
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


def execute_sql_safely(
    db_path: Path | str, sql: str, timeout_sec: float = 2.0
) -> tuple[list[tuple] | None, str | None]:
    """Безопасно выполняет SQL-запрос в базе SQLite (Read-Only + таймаут)."""
    db_path = str(db_path)

    def _run():
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            return cursor.fetchall()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run)
        try:
            return future.result(timeout=timeout_sec), None
        except TimeoutError:
            return None, "Query Timeout"
        except Exception as e:
            return None, str(e)


def match_results(pred_rows: list[tuple], gold_rows: list[tuple]) -> bool:
    """Сравнивает результаты запроса с учетом подмножества или перестановки столбцов.
    
    Возвращает True (1 балл), если:
    1. Результаты полностью совпадают.
    2. В ответе модели не все столбцы (столбцы модели - подмножество эталона).
    3. В ответе модели есть лишние столбцы (эталон - подмножество модели).
    4. Столбцы совпадают, но перепутан их порядок.
    """
    if not pred_rows or not gold_rows:
        return pred_rows == gold_rows

    # Число строк должно совпадать (выборка сущностей верна)
    if len(pred_rows) != len(gold_rows):
        return False

    # 1. Быстрое точное совпадение
    if set(pred_rows) == set(gold_rows):
        return True

    pred_cols = len(pred_rows[0])
    gold_cols = len(gold_rows[0])

    # 2. Модель вернула НЕ ВСЕ столбцы (pred_cols < gold_cols)
    if pred_cols < gold_cols:
        pred_set = set(pred_rows)
        for indices in itertools.permutations(range(gold_cols), pred_cols):
            projected_gold = {tuple(row[i] for i in indices) for row in gold_rows}
            if projected_gold == pred_set:
                return True

    # 3. Модель вернула лишние столбцы (pred_cols > gold_cols)
    elif pred_cols > gold_cols:
        gold_set = set(gold_rows)
        for indices in itertools.permutations(range(pred_cols), gold_cols):
            projected_pred = {tuple(row[i] for i in indices) for row in pred_rows}
            if projected_pred == gold_set:
                return True

    # 4. Количество столбцов одинаковое, но перепутан порядок
    else:
        gold_set = set(gold_rows)
        for indices in itertools.permutations(range(pred_cols)):
            projected_pred = {tuple(row[i] for i in indices) for row in pred_rows}
            if projected_pred == gold_set:
                return True

    return False


def execution_reward(
    completions: list[Any], query: list[str], db_id: list[str], **kwargs
) -> list[float]:
    """Главная награда: проверка корректности выполнения SQL на реальной базе."""
    rewards = []
    database_dir = PROJECT_PATH / "data" / "raw" / "spider_data" / "database"
    
    

    for completion, gold_sql, d_id in zip(completions, query, db_id):
        raw_text = to_text(completion)
        pred_sql = extract_sql_from_completion(raw_text)
        db_path = database_dir / d_id / f"{d_id}.sqlite"
        
        

        if not db_path.exists():
            rewards.append(0.0)
            continue

        # 1. Выполняем эталонный запрос
        gold_res, gold_err = execute_sql_safely(db_path, gold_sql)
        if gold_err is not None:
            print("Ошибка")
            rewards.append(0.0)
            continue

        # 2. Выполняем запрос модели
        pred_res, pred_err = execute_sql_safely(db_path, pred_sql)

        # 3. Сравниваем результаты (с учетом подмножества столбцов)
        if pred_err is None and match_results(pred_res, gold_res):
            rewards.append(1.0)
        else:
            print("Ошибка в совпадении")
            print("LLM: ", pred_sql)
            print("Эталон: ", gold_sql)
            print("----------------------------")
            rewards.append(0.0)
    print(rewards)
    return rewards


def format_reward(completions: list[Any], **kwargs) -> list[float]:
    """Награда за соблюдение формата рассуждений (<think>...</think> и ```sql...```)."""
    rewards = []
    pattern = r"<think>.*?</think>\s*```sql.*?```"
    for completion in completions:
        raw_text = to_text(completion)
        if re.search(pattern, raw_text.strip(), re.DOTALL):
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards


def syntax_reward(completions: list[Any], **kwargs) -> list[float]:
    """Награда за синтаксическую валидность SQL-запроса через sqlglot."""
    rewards = []
    for completion in completions:
        raw_text = to_text(completion)
        pred_sql = extract_sql_from_completion(raw_text)
        try:
            sqlglot.parse_one(pred_sql)
            rewards.append(0.1)
        except Exception as e:
            print(e)
            rewards.append(0.0)
    return rewards