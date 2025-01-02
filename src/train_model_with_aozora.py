import chardet
import os

from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import wandb

from decoder_only_model import DecoderOnlyModel


class ChatLLM:
    def __init__(self, vocab_size, embed_dim, num_layers, max_seq_len, num_heads, temperature=1.0):
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        self.model = DecoderOnlyModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            num_heads=num_heads
            )
        self.temperature = temperature
        self.eos_token_id = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id

    def chat(self, prompt):
        # 入力をトークン化
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]

        # 生成結果を保持する
        generated_ids = input_ids.clone()

        # 最大長までのトークンを生成
        for _ in range(self.max_seq_len - input_ids.size(1)):
            
            # モデルからロジットを取得
            logits = self.model(generated_ids)

            # 最新トークンのロジットを取得
            next_token_logits = logits[:, -1, :]

            # ソフトマックスを適用して確率を計算
            probs = F.softmax(next_token_logits / self.temperature, dim=-1)

            # 確率に基づいて次のトークンをサンプリング
            next_token_id = torch.multinomial(probs, num_samples=1)

            # 生成結果に次のトークンを追加
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
            
            # 終了トークンを検出した場合に終了
            if next_token_id.item() == self.eos_token_id:
                break
        # 生成されたトークンをデコードして返す
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        input_ids = self.tokenizer.encode(text, truncation=True, max_length=self.max_seq_len, return_tensors="pt").squeeze(0)
        labels = input_ids.clone()
        return input_ids, labels


def collate_fn(batch):
    """
    バッチ内のサンプルをパディングして整列
    """
    input_ids, labels = zip(*batch)
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100はCrossEntropyLossで無視される
    return input_ids_padded, labels_padded

def train_model(chat_model, dataset, num_epochs=3, batch_size=8, learning_rate=5e-5, save_path="chat_llm_12.pth"):
    # Initialize W&B
    wandb.init(project="chat-llm-training", config={
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_seq_len": chat_model.max_seq_len,
    })
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(chat_model.model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    chat_model.model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total_steps = len(dataloader)

        for step, (input_ids, labels) in enumerate(dataloader):
            # シフトして次のトークンを予測
            optimizer.zero_grad()
            logits = chat_model.model(input_ids)

            # CrossEntropyLoss expects (batch_size, vocab_size, seq_len), so reshape logits
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Log the loss to W&B
            wandb.log({"loss": loss.item(), "epoch": epoch + 1, "step": step + 1})

            # ステップごとの損失を表示
            if step % 10 == 0 or step == total_steps - 1:  # 10ステップごとまたは最後のステップで表示
                print(f"Epoch {epoch + 1}, Step {step + 1}/{total_steps}, Loss: {loss.item()}")

        # エポックごとの平均損失を表示
        average_loss = total_loss / total_steps
        print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")
        wandb.log({"average_loss": average_loss, "epoch": epoch + 1})

    # モデルの保存
    torch.save(chat_model.model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    wandb.finish()

# モデルのパラメータを指定
vocab_size = 32000
embed_dim = 512
num_layers = 12
max_seq_len = 128
num_heads = 8

# ChatLLM インスタンスを生成
chat_model = ChatLLM(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_layers=num_layers,
    max_seq_len=max_seq_len,
    num_heads=num_heads,
    temperature=0.8
)

# サンプルデータセット
texts = []
for file_name in ["data/cleaned_吾輩は猫である.txt", "data/cleaned_坊ちゃん.txt"]:
    with open(file_name, "r", encoding="utf-8") as f:
        texts.extend(f.readlines())

dataset = TextDataset(texts, chat_model.tokenizer, max_seq_len=max_seq_len)

# モデルを学習
train_model(chat_model, dataset, num_epochs=5, batch_size=2, learning_rate=5e-5)

# サンプルプロンプトでチャット
response = chat_model.chat("今日はどんな話をしましょうか？")
print(response)