import time
import torch  # type: ignore
import numpy as np  # type: ignore

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk   # type: ignore


def get_dataset(data_path, column="text"):
    dataset = load_from_disk(data_path)
    if isinstance(dataset, dict) and "train" in dataset:
        return dataset["train"]
    return dataset


def sliding_window_reshape_batch(batch, window_size: int, stride: int, pad_token_id: int):
    """
    Applies sliding window reshaping to a batch of sequences.
    Input:

    - batch: Batch of sequences with shape (batch_size, sequence_length).
        Useful for hf datasets parallelization.
    - window_size: Size of the sliding window.
    - stride: Stride of the sliding window.
    - pad_token_id: Token ID for padding.

    Output:

    - Dictionary with the reshaped input_ids and attention_mask.
        This is a standard format for hf datasets.

    """
    batch_input_ids = batch["input_ids"]

    all_input_ids = []
    all_attention_masks = []

    for input_ids in batch_input_ids:
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # Ensure 2D shape

        n = input_ids.size(-1)  # Sequence length
        n_windows = (n - 1) // stride + 1  # Compute number of windows

        # Initialize output tensors
        input_ids_out = torch.full((n_windows, window_size), pad_token_id, dtype=torch.long)
        attention_mask_out = torch.zeros((n_windows, window_size), dtype=torch.long)

        # Fill the windows
        for i, start in enumerate(range(0, n, stride)):
            end = min(start + window_size, n)
            length = end - start
            input_ids_out[i, :length] = input_ids[0, start:end]
            attention_mask_out[i, :length] = 1  # Mask only valid tokens

        all_input_ids.extend(input_ids_out)
        all_attention_masks.extend(attention_mask_out)

    return {"input_ids": all_input_ids, "attention_mask": all_attention_masks}


def preprocess_dataset(dataset, model_checkpoint, column="text", max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Eliminar truncation and max_length
    def tokenize_fn(examples):
        return tokenizer(examples[column], truncation=False, return_token_type_ids=False)

    # Check de output dimension
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=[column])
    dataset = dataset.map(
        lambda batch: sliding_window_reshape_batch(
            batch,
            window_size=512,
            stride=256,
            pad_token_id=tokenizer.pad_token_id
        ),
        batched=True
    )
    return dataset, tokenizer


def compute_metrics(eval_pred):
    """
    Calcula la accuracy para los tokens en los que labels != -100.
    Se espera que eval_pred sea una tupla (logits, labels).
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # Creamos una máscara para ignorar los tokens con label -100
    mask = labels != -100
    if mask.sum() == 0:
        accuracy = 0
    else:
        accuracy = (predictions[mask] == labels[mask]).mean()
    return {"accuracy": accuracy}


def continual_pretraining(
    data_path,
    model_checkpoint,
    output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    learning_rate=1e-4,
    logging_steps=50,
    save_steps=1000,
    weight_decay=1e-3,
    warmup_steps=5000,
    evaluation_strategy="epoch",
    eval_accumulation_steps=50,
    prediction_loss_only=True,
    disable_tqdm=False,
    report_to="tensorboard",
    test_size=0.1,
    seed=42,
    column="text",
    optim="adamw_torch"
):

    import os
    print(f"Process local rank: {os.environ.get('LOCAL_RANK')}")
    print(f"Process global rank: {os.environ.get('RANK')}")
    print(f"World size: {os.environ.get('WORLD_SIZE')}")

    dataset = get_dataset(data_path)
    dataset, tokenizer = preprocess_dataset(dataset, model_checkpoint, column=column)

    split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    """
    Experiment to do:
    [x] optimizer: no
    optimizer: adamw_torch   https://github.com/huggingface/transformers/blob/
    d1b92369ca193da49f9f7ecd01b08ece45c2c9aa/src/transformers/training_args.py#L148
    optimizer: adamw_apex_fused
    optimizer: adamw_torch_fused
    optimizer: adamw_anyprecision
    optimizer: adafactor
    optimizer: adamw_bnb_8bit
    optimizer: sgd
    [x] compile: false
    compile: true
    [x] model_precision: bf16
    [x] model_precision: fp16
    [x] model_precision: no
    """

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_steps=save_steps,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        evaluation_strategy=evaluation_strategy,
        eval_accumulation_steps=eval_accumulation_steps,
        prediction_loss_only=prediction_loss_only,
        disable_tqdm=disable_tqdm,
        report_to=report_to,
        optim=optim,
        # use_liger_kernel=True,  # Testing just in case
    )

    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    # model = torch.compile(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset.select(range(min(1000, len(eval_dataset)))),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")

    eval_result = trainer.evaluate()
    eval_loss = eval_result.get("eval_loss")
    perplexity = np.exp(eval_loss) if eval_loss is not None and eval_loss < 100 else float("inf")
    print(f"Evaluation Loss: {eval_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
