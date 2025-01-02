from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

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

def train_model(chat_model, dataset, num_epochs=3, batch_size=8, learning_rate=5e-5):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(chat_model.model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    chat_model.model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for input_ids, labels in dataloader:
            optimizer.zero_grad()
            logits = chat_model.model(input_ids)

            # CrossEntropyLoss expects (batch_size, vocab_size, seq_len), so reshape logits
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# モデルのパラメータを指定
vocab_size = 32000
embed_dim = 512
num_layers = 6
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
texts = [
    "こんにちは、今日はどんな話をしましょうか？",
    "機械学習は面白いです。",
    "天気が良いので散歩に行きたいです。",
    "新しい技術を学ぶのは楽しいですね。",
    "AIは多くの分野で活用されています。",
    "プログラミングは創造的な作業です。",
    "最近読んだ本はとても面白かったです。",
    "旅行の計画を立てるのはワクワクしますね。",
    "数学の問題を解くのはチャレンジングです。",
    "スポーツ観戦は良いリフレッシュになります。",
    "音楽を聴くとリラックスできます。",
    "料理をするのは趣味の一つです。",
    "映画を見るのは楽しいですね。",
    "自然の中で過ごすとリフレッシュできます。",
    "友達との会話は楽しい時間です。"
]

dataset = TextDataset(texts, chat_model.tokenizer, max_seq_len=max_seq_len)

# モデルを学習
train_model(chat_model, dataset, num_epochs=100, batch_size=2, learning_rate=5e-5)

# サンプルプロンプトでチャット
response = chat_model.chat("今日はどんな話をしましょうか？")
print(response)