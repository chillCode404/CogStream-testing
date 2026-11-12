# enable expandable segment for pytorch to reuse fragmented gpu mem segments
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import json
import re
import numpy as np
import random

# Monkey-patch torch.load to fallback to weights_only=False for trusted local checkpoints (PyTorch 2.6+ rng_state.pth compatibility)
_original_torch_load = torch.load

def _safe_torch_load(*args, **kwargs):
    try:
        return _original_torch_load(*args, **kwargs)
    except Exception as e:
        # If weights_only caused unpickling error, retry with weights_only=False
        if "weights_only" in kwargs:
            kwargs["weights_only"] = False
            return _original_torch_load(*args, **kwargs)
        elif len(args) < 3:  # weights_only is the 3rd positional or keyword argument
            return _original_torch_load(*args, weights_only=False, **kwargs)
        raise e

torch.load = _safe_torch_load
from torch.distributed import init_process_group
from transformers import (
    AutoModelForCausalLM,
    Trainer, TrainingArguments,
    LogitsProcessor, LogitsProcessorList, AutoProcessor
)
from peft import LoraConfig, get_peft_model, PeftModel
import random
from transformers import TrainerCallback
import argparse

def count_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total param: {total_params:,}")
    print(f"trainable param: {trainable_params:,}")
    print(f"trainable ratio: {trainable_params / total_params * 100:.4f}%")
    return trainable_params, total_params

# torchrun --nproc_per_node=8 first_stage_training.py
class EpochLossCallback(TrainerCallback):
    def __init__(self):
        self.epoch_losses = []
        self.stage_losses = []
    
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            epoch_loss = logs["loss"]
            self.epoch_losses.append(epoch_loss)
            self.stage_losses.append(epoch_loss)
            print(f"Epoch {state.epoch:.2f}: Average Loss = {epoch_loss:.4f}")

def load_json(folder_path):
    all_data = []
    for json_file in os.listdir(folder_path):
        if json_file.endswith(".json"):
            file_path = os.path.join(folder_path, json_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.append((json_file, data))
    
    return all_data
class StopTrainingCallback(TrainerCallback):
    def __init__(self, max_epochs=5):
        self.max_epochs = max_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch >= self.max_epochs:
            print(f"Epoch {state.epoch} reached {self.max_epochs}, stopping training.")
            control.should_terminate = True
        return control

# 1. 数据预处理函数
def format_example(example, include_demo=True):
    """
    Convert data sample to formatted prompt with instructions to output a list of indices
    of historical QA pairs that are helpful for answering the current question, prefixed
    with yes (requires additional visual info) or no (fully answerable by historical QA).
    """
    system_prompt = """<|im_start|>system
You are a QA-pair filtering assistant. Your task is to identify which of the historical QA pairs are helpful for answering the current question and determine if the historical QA pairs alone are sufficient to answer it.

A QA pair is considered helpful if it provides:
- Relevant background information, context, or details
- Additional facts or insights that can be used to answer the current question
- Matching roles, scenarios, or domain knowledge that could support the answer

Output a single bracketed sequence:
- Start with 'yes' if the historical QA pairs are insufficient to fully answer the question (additional visual information may be needed).
- Start with 'no' if the current question can be fully answered using only the historical QA pairs (no additional visual information needed).
- Follow with the indices (starting from 0) of the helpful QA pairs, e.g., [yes,0,5] or [no,0,5].
- If no QA pairs are helpful, output [yes] or [no] based on the question's dependency.
- Do not add extra text or explanation — only output the bracketed sequence.
<|im_end|>"""

    demo_section = ""
    if include_demo:
        demo_section = """\nExample:
Current Question: What causes earthquakes?
Historical QA Pairs:
0. Q: How to measure earthquakes? A: Using the Richter scale
1. Q: What is tectonic plate? A: Massive rock slabs beneath crust
2. Q: What is the weather like today? A: Sunny and warm
→ Output: [no,1]
------------------------------
Example:
Current Question: What does an earthquake look like?
Historical QA Pairs:
0. Q: How to measure earthquakes? A: Using the Richter scale
1. Q: What is tectonic plate? A: Massive rock slabs beneath crust
2. Q: What is the weather like today? A: Sunny and warm
→ Output: [yes]
------------------------------"""

    user_content = f"""{demo_section}
Current Question: {example['current_Q']}

Historical QA Pairs (ordered by time):"""

    for i, (q, a) in enumerate(zip(example['hist_Qs'], example['hist_As'])):
        user_content += f"\n{i}. Q: {q}\n   A: {a}"

    user_content += "\nGenerate a bracketed sequence (e.g., [yes,0,5] or [no,0,5]) indicating the dependency (yes or no) and the indices of helpful QA pairs. Only output the bracketed sequence."
    helpful_indices = [str(i) for i, label in enumerate(example['labels']) if label == 1]
    target = f",{','.join(helpful_indices)}]" if helpful_indices else "]"
    if example["if_visual"]:
        target = "[yes" + target
    else:
        target = "[no" + target
    full_text = (
        f"{system_prompt}"
        f"<|im_start|>user\n{user_content}<|im_end|>"
        f"<|im_start|>assistant\n{target}<|im_end|>"
    )

    
    return {
        "text": full_text,
        "prompt_part": f"{system_prompt}<|im_start|>user\n{user_content}<|im_end|><|im_start|>assistant\n",
        "target_part": f"{target}<|im_end|>"
    }

class CustomDataCollator:
    def __init__(self,
                 tokenizer,
                 max_length: int = 2048,
                 log_file="batch_log.txt",
                 eval_mode=False):
        self.eval_mode = eval_mode
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.log_file = log_file

    def __call__(self, batch_of_items):
        expansions = []
        log_entries = []

        for item_idx, item in enumerate(batch_of_items):
            if self.eval_mode:
                sample = self._make_sample(item["current_Q"], 
                                            item["hist_Qs"], item["hist_As"],
                                            item["labels_01"], item["if_visual"])
                expansions.append(sample)
            else:
                current_ID = item["ID"]
                if current_ID == 1:
                    sample = self._make_sample(item["current_Q"], 
                                            item["hist_Qs"], item["hist_As"],
                                            item["labels_01"], item["if_visual"])
                    expansions.append(sample)
                    log_entries.append(f"[Item {item_idx}] ID=1 => 1 sample.")
                
                elif current_ID == 2:
                    sample_normal = self._make_sample(item["current_Q"], 
                                                    item["hist_Qs"], item["hist_As"],
                                                    item["labels_01"], item["if_visual"])
                    expansions.append(sample_normal)
                    rev_Qs = list(reversed(item["hist_Qs"]))
                    rev_As = list(reversed(item["hist_As"]))
                    rev_labels = list(reversed(item["labels_01"]))
                    sample_rev = self._make_sample(item["current_Q"], rev_Qs, rev_As, rev_labels, item["if_visual"])
                    expansions.append(sample_rev)
                    log_entries.append(f"[Item {item_idx}] ID=2 => 2 samples.")
                
                else:
                    sample_normal = self._make_sample(item["current_Q"], 
                                                    item["hist_Qs"], item["hist_As"],
                                                    item["labels_01"], item["if_visual"])
                    expansions.append(sample_normal)

                    unique_shuffles = set()
                    while len(unique_shuffles) < 3:
                        indices = list(range(len(item["hist_Qs"])))
                        random.shuffle(indices)
                        shuf_Qs = [item["hist_Qs"][i] for i in indices]
                        shuf_As = [item["hist_As"][i] for i in indices]
                        if len(indices) != len(item["labels_01"]):
                            print(item["current_Q"])
                            print(indices)
                            print(item["labels_01"])
                            continue
                        shuf_labels = [item["labels_01"][i] for i in indices]
                        unique_shuffles.add(tuple(zip(shuf_Qs, shuf_As, shuf_labels)))
                    
                    for shuf_tuple in unique_shuffles:
                        sq, sa, sl = zip(*shuf_tuple)
                        expansions.append(
                            self._make_sample(item["current_Q"], list(sq), list(sa), list(sl), item["if_visual"])
                        )
                    log_entries.append(f"[Item {item_idx}] ID={current_ID} => total {1 + 2} samples.")

        tokenized_batch = []
        target = []
        for exp_idx, s in enumerate(expansions):
            formatted_data = format_example(s)
            prompt_part = formatted_data["prompt_part"]
            target_part = formatted_data["target_part"]
            target.append(target_part)
            encoded_prompt = self.tokenizer(
                prompt_part,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,  
                padding=False,
                return_overflowing_tokens=False,
            )

            encoded_target = self.tokenizer(
                target_part,
                truncation=True,
                max_length=self.max_length - len(encoded_prompt["input_ids"]),
                add_special_tokens=False, 
                padding=False,
                return_overflowing_tokens=False,
            )

            input_ids = encoded_prompt["input_ids"] + encoded_target["input_ids"]

            labels = [-100] * len(encoded_prompt["input_ids"]) + encoded_target["input_ids"]
            tokenized_batch.append({
                "input_ids": input_ids,
                "labels": labels,
            })

        saved_labels = [item.pop("labels") for item in tokenized_batch]

        batch_dict = self.tokenizer.pad(
            tokenized_batch,
            padding="longest",
            return_tensors="pt"
        )
        max_seq_len = batch_dict["input_ids"].shape[1]

        padded_labels = []
        for lbl in saved_labels:
            needed = max_seq_len - len(lbl)
            if needed > 0:
                lbl = lbl + ([-100] * needed)
            elif needed < 0:
                lbl = lbl[:max_seq_len]
            padded_labels.append(lbl)
        batch_dict["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        batch_dict["label"] = target

        return batch_dict

    def _make_sample(self, current_Q, hist_Qs, hist_As, labels_01, if_visual):
        return {
            "current_Q": current_Q,
            "hist_Qs": hist_Qs,
            "hist_As": hist_As,
            "labels": labels_01,
            "if_visual": if_visual,
        }

class ConstrainedTrainer(Trainer):
    def __init__(self, *args, logits_processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.logits_processor = logits_processor or LogitsProcessorList()

    def training_step(self, model, inputs, *args):
        try:
            loss = super().training_step(model, inputs)
            return loss
        except RuntimeError as e:
            torch.cuda.empty_cache()
            raise e
            
    def compute_loss(self, model, inputs, return_outputs=False, print_test=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        if self.logits_processor is not None:
            logits = outputs.logits
            for processor in self.logits_processor:
                logits = processor(inputs["input_ids"], logits)
            outputs.logits = logits

        shift_logits = outputs.logits[..., :-1, :]
        shift_labels = inputs["labels"][..., 1:].contiguous()
        if print_test:
                print(f"Shift Labels Length: {len(shift_labels[0])}")
                print(f"Shift Labels: {shift_labels[0].tolist()}")
                pred_ids = shift_logits[0].argmax(-1).tolist()
                print(f"Predicted IDs: {pred_ids}")
                print(f"Decoded Pred: {tokenizer.decode(pred_ids)}")
                valid_count = (shift_labels[0] != -100).sum().item()
                print(f"Valid tokens in shift_labels: {valid_count}")

        def decode_valid_predictions(shift_logits, shift_labels, tokenizer, num_samples=2):
            sample_idx = 0
            sample_labels = shift_labels[sample_idx]
            valid_positions = (sample_labels != -100).nonzero().squeeze(-1)
            
            if valid_positions.nelement() == 0:
                print(f"样本 {sample_idx}: 无有效标签")

                
            print(f"\n=== 样本 {sample_idx} 诊断 ===")
            for pos in valid_positions:
                true_token_id = sample_labels[pos].item()
                pred_token_id = shift_logits[sample_idx, pos].argmax(-1).item()

                input_token_id = inputs["input_ids"][sample_idx, pos+1].item()
                
                print(
                    f"位置 {pos} | "
                    f"输入 token: {tokenizer.decode(input_token_id)} | "
                    f"预测 token: {tokenizer.decode(pred_token_id)} ({pred_token_id}) | "
                    f"标签 token: {tokenizer.decode(true_token_id)} ({true_token_id}) | "
                    f"正确/错误: {pred_token_id == true_token_id}"
                )

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        if print_test:
            decode_valid_predictions(shift_logits, shift_labels, tokenizer)
            print(f"Loss: {loss.item()}")
        outputs.loss = loss
        return (outputs.loss, outputs) if return_outputs else outputs.loss

class FlattenedQADataset(torch.utils.data.Dataset):
    def __init__(self, all_data, shuffle=True, audit_file="qa_format_audit.json"):
 
        self.samples = []
        audit_report = {
            "rescued_dict_files": [],
            "rescued_1d_list_files": [],
            "skipped_entries": [],
        }

        if shuffle:
            random.shuffle(all_data)
        for entry in all_data:
            if isinstance(entry, tuple) and len(entry) == 2:
                json_file, original_data = entry
            else:
                json_file, original_data = "unknown", entry

            # 1. If wrapped in a dict with 'Data', unwrap it
            if isinstance(original_data, dict):
                audit_report["rescued_dict_files"].append(json_file)
                original_data = original_data.get("Data", [original_data])
            # 2. If it's a flat 1D list of dicts [ {"Q": ...}, ... ], wrap into 2D [ [ {"Q": ...} ] ]
            if isinstance(original_data, list) and len(original_data) > 0 and isinstance(original_data[0], dict):
                audit_report["rescued_1d_list_files"].append(json_file)
                original_data = [original_data]

            if not isinstance(original_data, list):
                audit_report["skipped_entries"].append({
                    "file": json_file,
                    "reason": f"Expected list at top-level but got {type(original_data).__name__}"
                })
                continue

            if shuffle:
                random.shuffle(original_data)
                
            for group_idx, group in enumerate(original_data):
                if not isinstance(group, list):
                    audit_report["skipped_entries"].append({
                        "file": json_file,
                        "group_idx": group_idx,
                        "reason": f"Expected group to be list but got {type(group).__name__}"
                    })
                    continue

                for i, qa_pair in enumerate(group):
                    if not isinstance(qa_pair, dict) or "Q" not in qa_pair or "info" not in qa_pair:
                        audit_report["skipped_entries"].append({
                            "file": json_file,
                            "group_idx": group_idx,
                            "item_idx": i,
                            "reason": "Missing required 'Q' or 'info' dictionary keys",
                            "raw": str(qa_pair)[:100]
                        })
                        continue

                    current_Q = qa_pair["Q"]
                    current_ID = qa_pair["info"]["ID"]
                    if_visual = qa_pair["info"]["is_visual"]
                    relevance = qa_pair["info"]["relevance"]


                    if isinstance(relevance, str):
                        relevance = json.loads(relevance)
                    labels_01 = [int(r) for r in relevance]  # 0/1
                    if len(labels_01) != qa_pair["info"]["ID"]:
                        print(qa_pair["Q"])

                    hist_Qs = [x["Q"] for x in group[:i]]
                    hist_As = [x["A"] for x in group[:i]]
                    
                    if current_ID == 0:
                        continue

                    if len(labels_01) != len(hist_Qs):
                        audit_report["skipped_entries"].append({
                            "file": json_file,
                            "group_idx": group_idx,
                            "item_idx": i,
                            "question": current_Q,
                            "reason": f"Mismatched label length: len(labels_01)={len(labels_01)} vs len(hist_Qs)={len(hist_Qs)}",
                            "raw": str(qa_pair)[:100]
                        })
                        continue
                    item = {
                        "current_Q": current_Q,
                        "hist_Qs": hist_Qs,
                        "hist_As": hist_As,
                        "labels_01": labels_01,   # [0,1,0,...]
                        "ID": current_ID,
                        "if_visual": if_visual,
                    }
                    self.samples.append(item)
        
        # Save audit report to disk
        audit_report["total_samples_loaded"] = len(self.samples)
        audit_report["total_skipped"] = len(audit_report["skipped_entries"])
        try:
            with open(audit_file, "w", encoding="utf-8") as f:
                json.dump(audit_report, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save audit file {audit_file}: {e}")

        print(f"[FlattenedQADataset] total {len(self.samples)} items.")
        print(f"[FlattenedQADataset] Formats rescued: {len(audit_report['rescued_dict_files'])} dicts, {len(audit_report['rescued_1d_list_files'])} 1D lists.")
        if audit_report["total_skipped"] > 0:
            print(f"[FlattenedQADataset] ⚠️ Warning: {audit_report['total_skipped']} non-QA entries skipped. Details saved to {audit_file}")
        else:
            print(f"[FlattenedQADataset] Audit report saved to {audit_file}.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class StructuredLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.allowed_token_ids = self._get_allowed_token_ids(tokenizer)
        self.mask_1d = None

    def _get_allowed_token_ids(self, tokenizer):
        special_tokens = [str(i) for i in range(10)] + ['[', ']', ',', '<|im_end|>', 'no', 'yes']
        allowed_ids = set()
        for t in special_tokens:
            token_ids = tokenizer.encode(t, add_special_tokens=False)
            for idx in token_ids:
                if idx >= 0:
                    allowed_ids.add(idx)
        return sorted(list(allowed_ids))

    def __call__(self, input_ids, scores):
        vocab_size = scores.shape[-1]
        if (
            self.mask_1d is None
            or self.mask_1d.device != scores.device
            or self.mask_1d.dtype != scores.dtype
            or self.mask_1d.shape[0] != vocab_size
        ):
            mask = torch.full((vocab_size,), float('-inf'), dtype=scores.dtype, device=scores.device)
            mask[self.allowed_token_ids] = 0
            self.mask_1d = mask
        return scores + self.mask_1d

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # preds: [batch_size, seq_len, vocab_size]
    # labels: [batch_size, seq_len]
    preds = np.argmax(preds, axis=-1)

    exact_match = 0
    token_acc = 0
    total_tokens = 0

    for pred_seq, label_seq in zip(preds, labels):
        pred_text = tokenizer.decode(pred_seq, skip_special_tokens=True)
        true_text = tokenizer.decode(label_seq, skip_special_tokens=True)
        pred_digits = re.findall(r'\d', pred_text)
        true_digits = re.findall(r'\d', true_text)
        if pred_digits == true_digits:
            exact_match += 1
        min_len = min(len(pred_digits), len(true_digits))
        token_acc += sum(p == t for p, t in zip(pred_digits[:min_len], true_digits[:min_len]))
        total_tokens += min_len
    return {
        "exact_match": exact_match / len(preds),
        "token_accuracy": token_acc / total_tokens if total_tokens > 0 else 0
    }

def setup(rank, world_size):
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.manual_seed(42 + rank)

if __name__ == "__main__":
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    setup(global_rank, world_size)
    os.makedirs(os.path.expanduser("~/.triton/autotune"), exist_ok=True)
    parser = argparse.ArgumentParser(description="Run VideoLLaMA3 QA selection module training.")
    parser.add_argument("--model_path", type=str, default="model", help="Path to the base model directory.")
    parser.add_argument("--QA_path", type=str, help="Path to the QA directory.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to previous LoRA adapter or trainer checkpoint to continue training.")
    args = parser.parse_args()
    model_path = args.model_path
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer


    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    target_modules = []
    for i in range(28):
        target_modules.extend([
            f"model.layers.{i}.self_attn.q_proj",
            f"model.layers.{i}.self_attn.k_proj",
            f"model.layers.{i}.self_attn.v_proj",
            f"model.layers.{i}.self_attn.o_proj",
            f"model.layers.{i}.mlp.gate_proj",
            f"model.layers.{i}.mlp.up_proj",
            f"model.layers.{i}.mlp.down_proj"
        ])
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    base_model.enable_input_require_grads()
    has_adapter = (
        args.resume_from_checkpoint
        and os.path.isdir(args.resume_from_checkpoint)
        and os.path.isfile(os.path.join(args.resume_from_checkpoint, "adapter_config.json"))
    )
    if has_adapter:
        print(f"Loading existing LoRA weights from {args.resume_from_checkpoint} to continue training...")
        model = PeftModel.from_pretrained(base_model, args.resume_from_checkpoint, is_trainable=True)
    else:
        print("Initializing fresh LoRA weights...")
        model = get_peft_model(base_model, lora_config)
    
    model = model.to(local_rank)
    model.config.use_cache = False
    for name, param in model.named_parameters():
        if "lora" not in name:
            assert not param.requires_grad, f"Non-LoRA parameter {name} is not frozen!"
        else:
            assert param.requires_grad, f"LoRA parameter {name} does not have gradients enabled!"
    trainable_params, total_params = count_trainable_parameters(model)
    print("total_params:", total_params, "trainable_params:", trainable_params)
    train_dataset = load_json(args.QA_path)
    train_dataset = FlattenedQADataset(train_dataset)
    logits_processor = LogitsProcessorList([StructuredLogitsProcessor(tokenizer)])
    training_args = TrainingArguments(
        output_dir="./stage1_lora_weights_custom",
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        max_grad_norm=0.5,
        warmup_ratio=0.08,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        logging_strategy="steps",
        logging_steps=20,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,  
        dataloader_num_workers=0,     
        ddp_find_unused_parameters=False,
        report_to=None
    )
    trainer = ConstrainedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        logits_processor=logits_processor,
        data_collator=CustomDataCollator(tokenizer, eval_mode=False),
        callbacks=[EpochLossCallback()]
    )
    resume_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        checkpoints = [
            os.path.join(training_args.output_dir, d)
            for d in os.listdir(training_args.output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(training_args.output_dir, d))
        ]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            resume_checkpoint = checkpoints[-1]
            print(f"Found existing training checkpoint: {resume_checkpoint}. Resuming training...")

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
