from transformers import AutoTokenizer

# トークナイザーのロード
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

# テキストをトークン化
text = "トークナイザーを使ってみましょう。"
tokens = tokenizer.tokenize(text)  # トークン化
print("トークン化された結果:", tokens)

# トークンをIDに変換
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print("トークンID:", token_ids)

# トークンIDから元のテキストにデコード
decoded_text = tokenizer.decode(token_ids)
print("復元されたテキスト:", decoded_text)