# AIWolf NLP LLM Judge

AIWolfゲームログを生成AI（LLM）で評価するシステム

## 概要

AIWolfゲームのログ（CSVファイル）を複数のLLMに渡し、事前定義された評価基準に沿ってランキング形式で評価を行います。プレイヤー数や役職構成はログから自動検出し、ゲームのレギュレーション（5人戦・9人戦・13人戦など）にかかわらず適切な評価基準を適用します。

## 主な機能

- **マルチプロバイダLLM統合**: OpenAI / Anthropic Claude / Google Gemini / Vertex AI / OpenAI互換ローカル推論サーバ（vLLM, Ollama, llama.cpp, LM Studio）
- **複数モデル同時評価**: 1回の実行で複数LLMを比較評価。モデル別に出力ディレクトリを分離
- **プレイヤー数・役職構成の自動検出**: ログCSVから自動で読み取るため、人数指定不要
- **適用条件付き評価基準**: `team_play` のような基準は「人狼が2人以上いるゲーム」など条件付きで自動適用
- **試運転（dry-run）**: 本実行前に1ゲーム×全評価基準で各モデルの疎通確認、失敗モデルは自動スキップ
- **Prompt Caching**: system + character_info の prefix を1ゲーム単位でキャッシュ。プロバイダ別の最適な方法（自動 / `cache_control` / `CachedContent`）を自動で使い分け
- **Batch API モード**: 本実行を各プロバイダのBatch API（24h SLA、50%引）に投入可能
- **並列処理**: プロセス × スレッドのマルチレベル並列化
- **チーム集計**: 複数ゲームの結果を自動集計し、チーム別の平均スコアを算出（JSON + CSV）
- **集計再生成**: 既存の評価結果から集計のみを再生成（LLM呼び出し不要）

## インストール

```bash
# uvがインストールされていない場合
pip install uv

# 依存関係のインストール
uv sync

# Vertex AI Batch API を使う場合のみ追加
uv add google-cloud-storage
```

## セットアップ

### APIキーの設定

```bash
cp config/.env.sample config/.env
# config/.env を編集し、使用するプロバイダのキーを実際の値に置き換える
```

| 環境変数 | 用途 |
|---|---|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic Claude |
| `GEMINI_API_KEY` | Google Gemini (AI Studio) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Vertex AI（サービスアカウントJSONへの絶対パス） |
| `VLLM_API_KEY` / `OLLAMA_API_KEY` 等 | ローカル推論サーバ（ダミー値でも可） |

### Vertex AI の認証

Vertex AI はAPIキーではなくサービスアカウントJSONで認証します:

1. GCPコンソールでサービスアカウントを作成しJSON鍵をダウンロード
2. ファイルを安全な場所に配置（例: `~/.private_keys/gcp-key.json`）
3. シェル設定ファイル（`~/.bashrc` / `~/.zshrc`）に追記:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.private_keys/gcp-key.json"
   ```
4. シェル再読み込み: `source ~/.bashrc`（または再起動）

`config/.env` に書く運用も可能です（コード側で読み込みます）。

### データの準備

```
data/
├── input/
│   ├── log/     # ゲームログファイル (*.log)
│   └── json/    # キャラクター情報ファイル (*.json)
└── output/      # 出力ルート（実行毎にタイムスタンプ配下にモデル別ディレクトリが作られる）
```

**重要**: ログファイル（`*.log`）とJSONファイル（`*.json`）は、拡張子前の名前が完全一致している必要があります。

## 使用方法

### 基本実行

```bash
# settings.yaml で定義された全モデルを順に評価
uv run python main.py -c config/settings.yaml

# デバッグログを出力
uv run python main.py -c config/settings.yaml --debug
```

### モデル絞り込み

```bash
# 指定したIDのモデルのみ実行（カンマ区切り）
uv run python main.py -c config/settings.yaml --models gpt-4o,claude-sonnet-4-5
```

### 試運転（dry-run）

各モデルが正常に動くか、本実行前に 1ゲーム × 全評価基準で疎通確認します。失敗したモデルは既定で自動的にスキップして本実行を続行します。

```bash
# 試運転をスキップ
uv run python main.py -c config/settings.yaml --skip-dry-run

# 試運転だけ実行（APIキーの疎通確認用、結果は保存されない）
uv run python main.py -c config/settings.yaml --dry-run-only
```

設定で挙動を調整できます:
- `processing.dry_run: false` … 試運転を完全にオフ
- `processing.dry_run_strict: true` … 1モデルでも失敗したら全体中断

### Batch API モード（コスト重視・24h SLA可）

```bash
# 本実行をプロバイダのBatch APIに投入（24h SLA、50%割引）
uv run python main.py -c config/settings.yaml --use-batch

# settings.yaml で use_batch_api: true でも同期実行を強制
uv run python main.py -c config/settings.yaml --no-batch
```

Vertex AI でBatch APIを使う場合は、モデル設定に `gcs_bucket: gs://your-bucket/aiwolf-judge/` を必須で指定し、`google-cloud-storage` を追加インストールしてください。

### 集計再生成

`*_result.json` から `team_aggregation.{json,csv}` のみを再生成します（LLM API呼び出しなし）:

```bash
# 最新の実行ディレクトリを対象に再生成
uv run python main.py -c config/settings.yaml --regenerate-aggregation

# 特定の実行ディレクトリを指定
uv run python main.py -c config/settings.yaml --regenerate-aggregation \
  --run-dir data/output/2026-05-16_14-32-00
```

## 設定

### メイン設定（`config/settings.yaml`）

```yaml
path:
  env: config/.env
  evaluation_criteria: config/evaluation_criteria.yaml

llm:
  prompt_yml: config/prompts.yaml
  models:
    - id: gpt-4o                   # 出力ディレクトリ名にもなる任意の識別子
      provider: openai
      model: gpt-4o
      api_key_env: OPENAI_API_KEY
      system_role: developer       # o-seriesは "developer"、他は "system"
    - id: claude-sonnet-4-5
      provider: anthropic
      model: claude-sonnet-4-5
      api_key_env: ANTHROPIC_API_KEY
      extra: { max_tokens: 8192 }
    - id: gemini-2-5-pro
      provider: gemini
      model: gemini-2.5-pro
      api_key_env: GEMINI_API_KEY
    - id: gemini-vertex
      provider: vertex_ai
      model: gemini-2.5-pro
      project_id: your-gcp-project
      location: us-central1
      gcs_bucket: gs://your-bucket/aiwolf-judge/   # Batch API利用時のみ必須
    - id: qwen3-8b-vllm                            # ローカル推論サーバ例
      provider: openai_compatible
      model: Qwen/Qwen3-8B-Instruct
      base_url: http://localhost:8000/v1
      api_key_env: VLLM_API_KEY

game:
  format: "main_match"             # main_match または self_match
  # プレイヤー数や人狼の人数はログファイルから自動検出されるため指定不要

processing:
  input_dir: "data/input"
  output_dir: "data/output"        # ルート。<output_dir>/<timestamp>/<model_id>/ が実行毎に作られる
  encoding: "utf-8"
  max_workers: 4                   # プロセス並列処理数（ゲーム間並列、同期モード時）
  evaluation_workers: 8            # スレッド並列処理数（評価基準並列、同期モード時）
  max_retries: 5                   # LLMバリデーション失敗時の最大再試行回数
  parallel_models: false           # true で複数モデルを同時並行実行
  dry_run: true                    # 本実行前の試運転
  dry_run_strict: false            # true で試運転失敗時に全体中断
  enable_caching: true             # prefix キャッシュの有効化
  use_batch_api: false             # 本実行をBatch APIに投入
  batch_poll_interval_seconds: 60  # バッチ状態ポーリング間隔
  batch_max_wait_seconds: 86400    # バッチ完了待ち最大時間
```

### サポートプロバイダ

| provider           | 用途                              | 必要な認証                                                                |
|--------------------|-----------------------------------|---------------------------------------------------------------------------|
| `openai`           | OpenAI公式（gpt-4o等）            | `api_key_env`                                                             |
| `openai_compatible`| vLLM/Ollama/llama.cpp/LM Studio   | `base_url` + `api_key_env`（ダミー可）                                    |
| `anthropic`        | Claude                            | `api_key_env`                                                             |
| `gemini`           | Google AI Studio API              | `api_key_env`                                                             |
| `vertex_ai`        | Vertex AI 上の Gemini             | `project_id` + `location` + `GOOGLE_APPLICATION_CREDENTIALS`（環境変数）   |

### 評価基準の定義（`config/evaluation_criteria.yaml`）

すべての評価基準は単一の `criteria:` リストに記述します。`applicable_when` を持たない基準は常時適用され、`applicable_when` を指定した基準はゲームの構成に応じて適用可否が切り替わります。

```yaml
criteria:
  - name: natural_expression
    description: 発話表現は自然か
    ranking_type: ordinal
    order: 1
  - name: contextual_dialogue
    description: 文脈を踏まえた対話は自然か
    ranking_type: ordinal
    order: 2
  # ... 他の共通基準 ...
  - name: team_play
    description: チームプレイができているか
    ranking_type: ordinal
    order: 6
    applicable_when:
      werewolf_count_gte: 2   # 人狼が2人以上のゲームでのみ評価
```

サポートされる `applicable_when` キー:
- `werewolf_count_gte`: 指定した人数以上の人狼が初期配役にいる場合のみ適用

## 出力形式

### ディレクトリ構造

```
data/output/
└── 2026-05-16_14-32-00/         # 実行毎のタイムスタンプ
    ├── run_metadata.json         # 実行設定スナップショット
    ├── gpt-4o/                   # llm.models[].id がそのまま使われる
    │   ├── {game_id}_result.json
    │   ├── team_aggregation.json
    │   └── team_aggregation.csv
    ├── claude-sonnet-4-5/
    │   └── ...
    └── gemini-2-5-pro/
        └── ...
```

### 個別ゲーム結果（`*_result.json`）

```json
{
  "game_id": "01K3T3XN1SHBHSBHV1JWDDVS7W",
  "game_info": {
    "format": "main_match",
    "player_count": 13,
    "werewolf_count": 3
  },
  "evaluations": {
    "team_play": {
      "rankings": [
        {
          "player_name": "Takumi",
          "team": "sunamelli-b",
          "ranking": 1,
          "reasoning": "優れたチームプレイを実現..."
        }
      ]
    }
  }
}
```

### チーム集計（JSON + CSV）

```json
{
  "team_averages": {
    "kanolab": {
      "発話表現は自然か": 3.9,
      "文脈を踏まえた対話は自然か": 3.4
    }
  },
  "team_sample_counts": {
    "kanolab": {
      "発話表現は自然か": 10,
      "文脈を踏まえた対話は自然か": 10
    }
  },
  "summary": {
    "total_games_processed": 14,
    "teams_found": ["kanolab", "GPTaku", "CamelliaDragons"],
    "criteria_evaluated": ["発話表現は自然か", "文脈を踏まえた対話は自然か"]
  }
}
```

```csv
Team,発話表現は自然か,文脈を踏まえた対話は自然か
CamelliaDragons,2.000000,2.222222
kanolab,3.900000,3.400000
```

## システム要件

- Python 3.11以上
- 使用するプロバイダに応じたAPIキー or サービスアカウント鍵

## 開発

### プロジェクト構造

```
src/
├── cli.py                       # CLIインターフェース
├── game/                        # ゲームドメイン（player_count等自動検出を含む）
├── evaluation/                  # 評価ドメイン（基準・LLMレスポンス・集計）
├── aiwolf_log/                  # ログCSV / JSON のパース
├── llm/
│   ├── client.py                # LLMClient プロトコル、ModelConfig、CacheHandle
│   ├── batch.py                 # BatchClient プロトコル、BatchRequest/BatchResult
│   ├── factory.py               # provider 文字列 → クライアント生成
│   ├── evaluator.py             # 同期評価のラッパ
│   ├── formatter.py             # ログ→LLM入力JSONLへの変換
│   └── providers/
│       ├── openai_client.py     # OpenAI + OpenAI互換ローカルサーバ
│       ├── openai_batch.py      # OpenAI Batch API
│       ├── anthropic_client.py  # Claude（tool_use + cache_control）
│       ├── anthropic_batch.py   # Anthropic Message Batches API
│       ├── gemini_client.py     # Gemini AI Studio / Vertex AI（CachedContent）
│       └── gemini_batch.py      # Gemini AI Studio inline / Vertex AI GCS Batch
├── processor/                   # バッチ処理オーケストレーション
│   ├── batch_processor.py       # 複数モデル + 複数ゲームの統括
│   ├── batch_orchestrator.py    # Batch API モード時のオーケストレータ
│   ├── game_processor.py        # 単一ゲーム処理（同期パス）
│   ├── models/                  # ProcessingConfig 等
│   └── pipeline/                # データ準備・評価実行・結果出力サービス
└── utils/                       # 汎用ユーティリティ
```

### テスト

```bash
uv run pytest
uv run pytest --cov=src
```

## ライセンス

このプロジェクトのライセンスについては、リポジトリルートの LICENSE ファイルを参照してください。
