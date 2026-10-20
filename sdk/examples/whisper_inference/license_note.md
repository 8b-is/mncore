# Whisper サンプルのライセンスとコードの由来

## SDK に同梱するソースコード

このサンプルには、[openai/whisper](https://github.com/openai/whisper) の
ソースコードをコピー・改変・移植した部分と、独自に実装した部分が含まれます。
Whisper 由来の部分の著作権表示は `Copyright (c) 2022 OpenAI` で、
MIT License が適用されます。著作権表示とライセンス全文は、同梱の
[LICENSE.whisper](./LICENSE.whisper) に収録しています。

コードの由来と改変内容は次のとおりです。

- `whisper_utils.py`、`whisper_inference.py` の推論処理：
  データセットのラッパー、推論の流れ、モデル出力の後処理、Word Error Rate
  の算出処理には、Whisper のコードのコピー・改変・移植に該当する部分が含まれます。
  MLSDK 用のモデル実装に合わせて変更しており、独自に実装しなおした部分も含まれます。
  参照元は
  [notebooks/LibriSpeech.ipynb](https://github.com/openai/whisper/blob/c0d2f624c09dc18e709e37c2ad90c039a4eb72a2/notebooks/LibriSpeech.ipynb)
  と
  [whisper/decoding.py](https://github.com/openai/whisper/blob/c0d2f624c09dc18e709e37c2ad90c039a4eb72a2/whisper/decoding.py)
  です。後者の `DecodingTask`、`PyTorchInference`、`decode()` を使用・参照し、
  デコーディングループと候補の選択・テキスト化を MLSDK 用に変更しています。
- `static_whisper.py` のモデル実装：
  Whisper のアルゴリズムや仕様を参考に独自に再実装した部分に加え、
  上流のモデル処理をコピー・改変した部分も含まれます。派生元・参照元は
  [whisper/model.py](https://github.com/openai/whisper/blob/c0d2f624c09dc18e709e37c2ad90c039a4eb72a2/whisper/model.py)
  の `Whisper`、`AudioEncoder`、`TextDecoder`、`ResidualAttentionBlock`、
  `MultiHeadAttention` です。具体的には、`AudioEncoder.forward` の処理を
  `AudioEncoderHead` と `AudioEncoderTransformer` に分け、
  `ResidualAttentionBlock.forward` と `MultiHeadAttention.forward`／`qkv_attention`
  の処理をブロック・attention の関数に移植・改変しています。
  `TextDecoder.forward` は `TextDecoderEmbedding` と `TextDecoderTransformer`
  に分け、KV キャッシュや forward の差し替えを MLSDK 向けに変更しています。

## 実行時に取得するパッケージと学習済みの重み

サンプルの実行スクリプトは `openai-whisper` などの依存パッケージをインストールし、
推論プログラムは Whisper の学習済みの重みをダウンロードして読み込みます。
モデル内部の forward を差し替えるため、`openai-whisper` は参照元の
ソースに記載されたバージョン `20250625` に固定しています。
Whisper パッケージと重みは SDK に同梱せず、実行時に取得します。
Whisper の実装と学習済みの重みは MIT License で公開されています。
各依存パッケージには、それぞれのライセンスが適用されます。

## 実行時に取得するデータセット

このサンプルは
[LibriSpeech ASR corpus](https://www.openslr.org/12) の `test-clean` を既定で使用します。
データセットは SDK に同梱せず、実行時に `torchaudio.datasets.LIBRISPEECH` が取得します。
LibriSpeech のライセンスは
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.ja) です。
これはデータセットのライセンスであり、上記の Whisper 由来のコードの
MIT License とは別に適用されます。
