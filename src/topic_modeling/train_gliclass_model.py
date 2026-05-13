import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.csv as pv
import pandas as pd
import os
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from gliclass.training import TrainingArguments, Trainer
from gliclass.data_processing import GLiClassDataset, DataCollatorWithPadding, AugmentationConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
import torch
import numpy
import gzip

if __name__ =="__main__":

    # our model fine-tuning uses the v1 model of GLiClass, and our annotated data subsample
    df = pd.read_excel("sample_dataset_labels.xlsx", sheet_name="sample_dataset_labels")
    labels = ["sports", "movies/tv shows", "art/design", "video games", "books/literature", "politics", "technology", "science", "business", "lifestyle", "music", "travel", "social/general/other"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GLiClassModel.from_pretrained("knowledgator/gliclass-base-v1.0").to(device)
    tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-base-v1.0")

    # each row of the annotated data needs to be converted into a specific format specified by GLiClass:
    # - the text of the actual tweet
    # - the list of all possible labels
    # - our actual manually assigned labels from our annotations
    train_data = []
    for index, row in df.iterrows():
        label_list = [label for label in labels if row[label] == 1]
        train_data.append({
            "text": row['tweet'],
            "all_labels": labels,
            "true_labels": label_list
        })
    print(train_data[0])
    print("num examples:", len(train_data))

    # fine-tuning of the model according to GLiClass tutorial: https://huggingface.co/blog/Ihor/refreshing-zero-shot-classification#how-to-fine-tune 
    max_length = 1024
    problem_type = "multi_label_classification"

    # training arguments for our fine-tuning
    training_args = TrainingArguments(
        output_dir='models/test',
        learning_rate=1e-5,
        weight_decay=0.01,
        others_lr=1e-5,
        others_weight_decay=0.01,
        lr_scheduler_type='linear',
        warmup_ratio=0.0,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=8,
        eval_strategy="epoch",
        save_steps = 1000,
        save_total_limit=10,
        dataloader_num_workers=8,
        logging_steps=10,
        use_cpu = False,
        report_to="none",
        fp16=False,
        )

    train_data, test_data = train_test_split(
        train_data,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    augment_config = AugmentationConfig(enabled=False)

    train_dataset = GLiClassDataset(
        train_data,
        tokenizer,
        augment_config,
        max_length=max_length,
        problem_type=problem_type,
    )
    test_dataset = GLiClassDataset(
        train_data,
        tokenizer,
        augment_config,
        max_length=max_length,
        problem_type=problem_type,
    )

    data_collator = DataCollatorWithPadding(device=device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

    # saving fine-tuned model as a pretrained model for use later
    model.save_pretrained("./gliclass_finetuned_EECS767")
    tokenizer.save_pretrained("./gliclass_finetuned_EECS767")
