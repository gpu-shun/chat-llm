# chat-llm

## 各ファイルの説明

multi_head_attention.py
マルチヘッドアテンションを実装

transformer.py
トランスフォーマーを実装（マルチヘッドアテンションを使用して）

decoder_only_model.py
デコーダーオンリーモデルを実装（トランスフォーマーを使用して）

chat_llm.py
チャットモデルを実装（デコーダーオンリーモデルを使用して）

chat_llm_with_param.py
学習済みのパラメータを使用して推論をさせるコード

train_model.py
チャットモデルを事前学習させるコード（学習データはハードコーディング）

train_model_with_aozora.py
チャットモデルを事前学習させるコード（学習データは青空文庫のデータ）

.pthファイル
実際に学習をさせたパラメータ

tokenizer_test.py
transformersライブラリに公開されているトークナイザーを実際に使ってみたコード