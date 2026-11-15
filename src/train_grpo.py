from pathlib import Path
import torch
from datasets import load_from_disk
from trl.trainer import GRPOTrainer, GRPOConfig


from config import PROJECT_PATH
from model import base_model, peft_config, tokenizer
from reward import execution_reward, format_reward, syntax_reward


def format_prompt(example: dict) -> dict:
    """Форматирует диалог в виде system и user сообщений."""
    return {
        "prompt": [
            {
                "role": "system",
                "content": (
                    "Ты эксперт по SQL. Прежде чем писать код ты думаешь."
                    "Сначала пиши свои рамышления внутри тэгов <think> и </think>. "
                    "Дальше пиши финальный код SQL вннутри ```sql и ```."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"### Database Schema:\n{example['sql_context']}\n\n"
                    f"### Question:\n{example['question']}"
                ),
            },
        ]
    }


def main():
    # 1. Пути к данным и выходам
    data_path = PROJECT_PATH / "data" / "processed" / "train"
    output_dir = PROJECT_PATH / "outputs" / "grpo"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Обработанный датасет не найден по пути: {data_path}. "
            "Сначала выполните подготовку данных: python scripts/prepare_hf.py"
        )

    # 2. Загрузка и подготовка датасета
    print(f"Загрузка датасета из {data_path}...")
    dataset = load_from_disk(str(data_path))
    dataset = dataset.map(format_prompt)

    # 3. Настройка параметров GRPO (Reinforcement Learning)
    is_cuda = torch.cuda.is_available()
    use_bf16 = is_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = is_cuda and not use_bf16
    
    print(
        f"\n[Hardware Check] "
        f"CUDA: {'🟢 OK' if is_cuda else '🔴 No'} | "
        f"BF16: {'✅' if use_bf16 else '❌'} | "
        f"FP16: {'✅' if use_fp16 else '❌'}"
    )    

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=1e-5,               # Для RL необходим очень низкий learning rate
        beta=0.04,                        # Коэффициент регуляризации KL-дивергенции
        num_generations=4,                # G: количество генерируемых вариантов на 1 вопрос
        temperature=0.8,                  # Разнообразие в рассуждениях
        max_completion_length=1024,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,   # Эффективный размер батча = 1 * 8 * 4 = 32 ответа
        num_train_epochs=1,
        warmup_steps=0.1,
        logging_steps=1,
        save_steps=50,
        bf16=use_bf16,
        fp16=use_fp16,
        run_name="first_run",
        report_to="tensorboard",
        max_steps=100,
        
        loss_type = "cispo"
    )

    # 4. Инициализация тренера GRPO с функциями наград
    print("Инициализация GRPOTrainer с верификаторами...")
    trainer = GRPOTrainer(
        model=base_model,
        reward_funcs=[execution_reward, format_reward],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,

    )

    # 5. Запуск обучения
    print("🚀 Старт GRPO обучения...")
    trainer.train()

    # 6. Сохранение финальной модели
    final_model_dir = output_dir / "final_model"
    print(f"Сохранение модели в {final_model_dir}...")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    print("Обучение успешно завершено!")


if __name__ == "__main__":
    main()
