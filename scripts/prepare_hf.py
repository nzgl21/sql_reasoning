import json
from pathlib import Path
from datasets import Dataset, Features, Value
from connection import get_db_ddl


def create_ds(spider_data_path: Path) -> Dataset:
    """Извлекает db_id, query, question, sql_context и возвращает Hugging Face Dataset."""
    file_path = spider_data_path / "train_spider.json"
    databases_dir = spider_data_path / "database"

    with open(file_path, "r", encoding="utf-8") as f:
        raws = json.load(f)

    # 1. Кэш DDL, чтобы не подключаться к одной базе сотни раз
    ddl_cache = {}
    cleaned_rows = []

    # 2. Извлекаем только нужные поля (защита от падения на поле "sql")
    for raw in raws:
        db_id = raw["db_id"]
        
        if db_id not in ddl_cache:
            ddl_cache[db_id] = get_db_ddl(db_id, databases_dir)

        cleaned_rows.append({
            "db_id": db_id,
            "question": raw["question"],
            "query": raw["query"],
            "sql_context": ddl_cache[db_id],
        })

    # 3. Описываем схему (обратите внимание на 'string')
    features = Features({
        "db_id": Value("string"),
        "question": Value("string"),
        "query": Value("string"),
        "sql_context": Value("string"),
    })

    # 4. Создаем и возвращаем Dataset
    return Dataset.from_list(cleaned_rows, features=features)




if __name__ == "__main__":
    # Укажите ваш путь к папке spider_data
    PROJECT_ROOT = Path(__file__).parent.parent
    spider_path = PROJECT_ROOT / "data" / "raw" / "spider_data"
    
    ds = create_ds(spider_path)
    print(ds)

    ds.save_to_disk(PROJECT_ROOT/ "data" / "processed" / "train")
    