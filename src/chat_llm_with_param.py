from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

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

    def load_model(self, path):
        """モデルのパラメータをロード"""
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.model.eval()  # 推論モードに設定

    def chat(self, prompt):
        # 入力をトークン化
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]

        # 生成結果を保持する
        generated_ids = input_ids.clone()

        # 最大長までのトークンを生成
        for _ in range(self.max_seq_len - input_ids.size(1)):
            
            # モデルからロジットを取得
            logits = self.model(generated_ids)

            print(logits.size())

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

# 保存済みモデルのパス
model_path = "chat_llm_6.pth"

# モデルのパラメータをロード
chat_model.load_model(model_path)

# サンプルプロンプトでチャット
response = chat_model.chat("こんにちは、今日はどんな話をしましょうか？")
print(response)