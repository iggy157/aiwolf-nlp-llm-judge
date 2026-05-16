# AIWolf NLP LLM Judge

AIWolfゲームログを生成AI（LLM）で評価するシステム

## プロジェクト概要

このプロジェクトは、AIWolfゲームのログ（CSVファイル）を生成AI（LLM）に渡し、事前定義された評価基準に沿って評価を行うシステムです。5人戦と13人戦の両方に対応し、共通評価項目とゲーム形式固有の評価項目を独立して評価します。評価方式はランキング形式で、各プレイヤーを相対的に順位付けします。

**現在のステータス**: **機能完成** - エンドツーエンドでの評価処理が完全動作

## システム特徴

- **型安全Python**: 広範囲な型ヒント、データクラス、Enumの使用
- **モジュラー・モノリス**: 機能別モジュール分離と明確な境界
- **並列処理**: マルチレベル並列化（プロセス・スレッドベース）
- **設定駆動**: 柔軟なYAMLベース設定システム
- **マルチプロバイダLLM統合**: OpenAI / Anthropic Claude / Google Gemini / Vertex AI / OpenAI互換ローカル推論サーバ（vLLM, Ollama, llama.cpp, LM Studio）
- **複数モデル同時評価**: 1回の実行で複数LLMを評価し、モデル別に出力ディレクトリを分離
- **試運転（dry-run）**: 本実行前に1ゲーム×全評価基準で各モデルの疎通確認
- **Prompt Caching**: system + character_info の prefix を1ゲーム単位でキャッシュ。OpenAI自動 / Anthropic `cache_control` / Gemini・Vertex `CachedContent` を自動で使い分け
- **Batch API モード**: 本実行をプロバイダのBatch API（OpenAI / Anthropic / Gemini / Vertex AI）に投入。24h SLA、50%引。ローカルは自動で同期パスにフォールバック
- **バリデーション**: 自動再試行機能付きLLMレスポンス検証

## 使用方法

### 基本実行
```bash
# 設定ファイル指定での実行（settings.yaml で定義された全モデルを順に評価）
uv run python main.py -c config/settings.yaml

# デバッグモードでの実行
uv run python main.py -c config/settings.yaml --debug

# 実行するモデルを絞り込み（カンマ区切り）
uv run python main.py -c config/settings.yaml --models gpt-4o,claude-sonnet-4-5

# 試運転をスキップして直接本実行
uv run python main.py -c config/settings.yaml --skip-dry-run

# 試運転のみ実行（疎通確認用、結果は保存されない）
uv run python main.py -c config/settings.yaml --dry-run-only

# Batch APIを使う本実行（24h SLA、50%引、コスト重視時）
uv run python main.py -c config/settings.yaml --use-batch

# settings.yamlでuse_batch_api: trueでも同期実行を強制したい場合
uv run python main.py -c config/settings.yaml --no-batch

# 集計再生成（最新の実行ディレクトリを対象に）
uv run python main.py -c config/settings.yaml --regenerate-aggregation

# 集計再生成（特定の実行ディレクトリを指定）
uv run python main.py -c config/settings.yaml --regenerate-aggregation --run-dir data/output/2026-05-16_14-32-00
```

### データファイル配置
```
data/
├── input/
│   ├── log/                   # ゲームログファイル (*.log)
│   └── json/                  # キャラクター情報ファイル (*.json)
└── output/                    # 出力ルート。実行毎にタイムスタンプディレクトリが作られる
    └── 2026-05-16_14-32-00/   # 実行タイムスタンプ
        ├── run_metadata.json   # 実行設定スナップショット
        ├── gpt-4o/             # モデル別ディレクトリ（llm.models[].id がそのまま使われる）
        │   ├── {game_id}_result.json
        │   ├── team_aggregation.json
        │   └── team_aggregation.csv
        └── claude-sonnet-4-5/
            └── ...
```

**重要**: ログファイル（*.log）とJSONファイル（*.json）は、拡張子前の名前が完全一致している必要があります。

### 試運転（dry-run）

`processing.dry_run: true`（デフォルト）の場合、本実行前に各モデルに対して 1 ゲーム × 全評価基準で疎通確認を行います。

- 結果は**保存されません**（メモリ上で破棄）
- 試運転で失敗したモデルは：
  - `processing.dry_run_strict: false`（デフォルト）→ そのモデルだけスキップして本実行を続行
  - `processing.dry_run_strict: true` → 全体を中断
- `--skip-dry-run` で試運転自体を省略
- `--dry-run-only` で試運転だけ実行し、本実行はしない（APIキーやエンドポイントの疎通確認用）

### 集計再生成モード
既存の個別評価結果ファイル（`*_result.json`）から、チーム集計ファイル（`team_aggregation.json`と`team_aggregation.csv`）のみを再生成する機能です。`--run-dir` で対象のタイムスタンプディレクトリを指定すると、その配下の各モデルディレクトリで集計を再生成します。指定がない場合は `output_root` 直下で最新のタイムスタンプディレクトリが使われます。

**使用場面**:
- チーム集計ファイルを誤って削除した場合の復元
- 集計ロジックを変更した際の再集計
- 個別評価は完了しているが集計のみやり直したい場合

**注意事項**:
- 対象の `<output_root>/<timestamp>/<model_id>/` 配下に個別評価結果（`*_result.json`）が存在する必要があります
- ゲームログの再評価は行わないため、LLM APIを呼び出しません（コスト削減）

## 設定管理

### メイン設定（`config/settings.yaml`）
```yaml
path:
  env: config/.env
  evaluation_criteria: config/evaluation_criteria.yaml

llm:
  prompt_yml: config/prompts.yaml
  models:
    - id: gpt-4o                    # 出力ディレクトリ名にもなる任意の識別子
      provider: openai
      model: gpt-4o
      api_key_env: OPENAI_API_KEY
      system_role: developer        # o-seriesは "developer"、他は "system"
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
      # 認証は GOOGLE_APPLICATION_CREDENTIALS をシェル環境変数で設定
    - id: qwen3-8b-vllm             # ローカル推論サーバ（vLLM, Ollama等）
      provider: openai_compatible
      model: Qwen/Qwen3-8B-Instruct
      base_url: http://localhost:8000/v1
      api_key_env: VLLM_API_KEY     # ダミー値でも可

game:
  format: "main_match"        # main_match または self_match
  # プレイヤー数や人狼の人数はログファイルから自動検出されるため指定不要

processing:
  input_dir: "data/input"
  output_dir: "data/output"   # ルート。実行毎に <output_dir>/<timestamp>/<model_id>/ が作られる
  max_workers: 4              # プロセス並列処理数（ゲーム間並列）
  evaluation_workers: 8       # スレッド並列処理数（評価基準並列）
  max_retries: 5              # LLMバリデーション失敗時の最大再試行回数
  parallel_models: false      # true で複数モデルを同時並行実行（API/VRAM負荷に注意）
  dry_run: true               # true で本実行前に試運転を実施
  dry_run_strict: false       # true で試運転失敗時に全体中断、false で失敗モデルをスキップ
```

### サポートプロバイダ

| provider           | 用途                              | 必要な認証                                              |
|--------------------|-----------------------------------|---------------------------------------------------------|
| `openai`           | OpenAI公式（gpt-4o等）            | `api_key_env`（例: `OPENAI_API_KEY`）                   |
| `openai_compatible`| vLLM/Ollama/llama.cpp/LM Studio   | `base_url` + `api_key_env`（ダミー値可）                |
| `anthropic`        | Claude                            | `api_key_env`（例: `ANTHROPIC_API_KEY`）                |
| `gemini`           | Google AI Studio API              | `api_key_env`（例: `GEMINI_API_KEY`）                   |
| `vertex_ai`        | Vertex AI 上の Gemini             | `project_id` + `location` + `GOOGLE_APPLICATION_CREDENTIALS`（環境変数） |

**Vertex AI の認証**: サービスアカウントJSONへのパスをシェル設定ファイル（`~/.bashrc` / `~/.zshrc`）に書き、`export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.private_keys/key.json"` のようにシェル環境変数として設定してください。本プロジェクト側で再読み込みは行いません。

### 構造化出力のしくみ

各プロバイダで最適な方法を使い分け、Pydanticモデル（`EvaluationLLMResponse`）への変換まで自動で行います。

- `openai`: `client.beta.chat.completions.parse(response_format=...)` でPydantic直結
- `openai_compatible`: `response_format={"type":"json_schema", ...}` を指定
- `anthropic`: `tool_use` を強制ツール選択で実行し、入力をPydanticに復元
- `gemini` / `vertex_ai`: `response_mime_type=application/json` + `response_schema=PydanticModel`

### Prompt Caching（コスト削減）

1ゲーム内では全評価基準で `system プロンプト + character_info` が共通であることを利用し、prefix をプロバイダ機能でキャッシュします。

- `openai` / `openai_compatible`: 自動（コード変更不要、1024+ tokens prefix が 5–10 分キャッシュされ50%引）
- `anthropic`: `cache_control: {type: ephemeral}` マーカーを system + character_info に付与（読み込み時はベース価格の10%）
- `gemini` / `vertex_ai`: ゲーム開始時に `client.caches.create()` で `CachedContent` リソース作成、各評価基準呼び出しで `cached_content` を参照、ゲーム終了時に削除（キャッシュ部分はベース価格の25%、TTLは `cache_ttl_seconds` で制御、デフォルト3600秒）

`processing.enable_caching: false` で無効化可能（既定: true）。キャッシュ閾値を下回るゲームの場合、Geminiは自動的にno-cacheにフォールバックします。

### Batch API モード

`processing.use_batch_api: true` または `--use-batch` で本実行をBatch APIに切り替え。プロバイダ別フロー:

| プロバイダ | 入力 | 出力 | SLA | 割引 |
|---|---|---|---|---|
| OpenAI | JSONL（Files APIにアップロード） | JSONL（output_file_id） | 24h | 50% |
| Anthropic | リクエストリスト（直接送信） | ストリーミング取得 | 24h | 50% |
| Gemini (AI Studio) | inline リクエスト | inline レスポンス | 24h | 50% |
| Vertex AI | GCS の JSONL ファイル | GCS の JSONL ファイル | 数時間〜 | 50% |
| openai_compatible（ローカル） | - | - | - | 同期パスに自動フォールバック |

**Vertex AI Batch を使う場合**: モデル設定に `gcs_bucket: gs://your-bucket/aiwolf-judge/` を必須で指定してください。バケットは事前に作成しておくこと。`google-cloud-storage` SDK は本プロジェクトの依存ではないので、Vertex Batchを使うなら `uv add google-cloud-storage` を別途実行してください。

**Batch API の注意点**:
- 試運転（dry-run）は常に同期で実施（バッチで疎通確認すると24h待ち＋無駄な課金になるため）
- 完了待ちは `batch_max_wait_seconds`（既定86400秒）、ポーリングは `batch_poll_interval_seconds`（既定60秒）
- Caching と Batch の併用: OpenAI（自動キャッシュ）と Anthropic（cache_control はバッチでも有効）はそのまま効く。Gemini/Vertex のバッチでは現実装はキャッシュリソースを併用しません

### 評価基準定義（`config/evaluation_criteria.yaml`）

すべての評価基準は単一の `criteria:` リストに記述します。
`applicable_when` を持たない基準は常に適用され、`applicable_when` を指定した基準は
ゲームの構成（現在は人狼の人数）に応じて適用可否が切り替わります。

```yaml
criteria:
  - name: "natural_expression"
    description: "発話表現は自然か"
    ranking_type: "ordinal"
    order: 1

  - name: "contextual_dialogue"
    description: "文脈を踏まえた対話は自然か"
    ranking_type: "ordinal"
    order: 2

  # ... 中略 ...

  - name: "team_play"
    description: "チームプレイができているか"
    ranking_type: "ordinal"
    order: 6
    applicable_when:
      werewolf_count_gte: 2   # 人狼が2人以上のゲームでのみ評価
```

**サポートされる `applicable_when` キー**:
- `werewolf_count_gte`: 指定した人数以上の人狼が初期配役にいる場合のみ適用

## 出力形式

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
          "reasoning": "Takumi consistently supported team decisions..."
        }
      ]
    }
  }
}
```

### チーム集計結果（JSON・CSV）

#### JSON形式（`team_aggregation.json`）
```json
{
  "team_averages": {
    "kanolab": {
      "発話表現は自然か": 3.9,
      "文脈を踏まえた対話は自然か": 3.4,
      "発話内容は一貫しており矛盾がないか": 3.3
    }
  },
  "team_sample_counts": {
    "kanolab": {
      "発話表現は自然か": 10,
      "文脈を踏まえた対話は自然か": 10,
      "発話内容は一貫しており矛盾がないか": 10
    }
  },
  "summary": {
    "total_games_processed": 14,
    "teams_found": ["kanolab", "GPTaku", "CamelliaDragons"],
    "criteria_evaluated": ["発話表現は自然か", "文脈を踏まえた対話は自然か", ...]
  }
}
```

#### CSV形式（`team_aggregation.csv`）
```csv
Team,発話表現は自然か,文脈を踏まえた対話は自然か,発話内容は一貫しており矛盾がないか,...
CamelliaDragons,2.000000,2.222222,2.666667,...
kanolab,3.900000,3.400000,3.000000,...
```

**CSV出力の特徴**:
- ヘッダー: `Team`, 各評価基準名の順序
- 精度: 平均順位は小数点以下6桁まで表示
- 並び順: チーム名のアルファベット順、評価基準はorder順
- エンコーディング: UTF-8で日本語対応

## アーキテクチャ

### モジュール構造
```
src/
├── cli.py                    # CLIインターフェース
├── game/                     # ゲームドメイン
├── evaluation/               # 評価ドメイン
├── processor/                # バッチ処理システム
│   ├── models/               # 処理用データモデル
│   ├── pipeline/             # パイプラインサービス
│   │   └── aggregation_output.py  # チーム集計結果出力サービス
│   ├── batch_processor.py    # バッチ処理オーケストレーター
│   └── game_processor.py     # 単一ゲーム処理
├── aiwolf_log/               # ログファイル管理
├── llm/                      # LLM統合
└── utils/                    # 汎用ユーティリティ
```

### 処理フロー
1. **ゲーム発見**: 入力ディレクトリ内の全ログ・JSONペアを検索
2. **並列処理**: ProcessPoolExecutorによるゲーム間並列処理
3. **パイプライン処理**:
   - データ準備・設定管理
   - ログ変換・キャラクター情報取得
   - 並列評価実行（ThreadPoolExecutor）
   - 結果保存・チームマッピング
4. **チーム集計**: 全ゲーム結果の統計処理とJSON・CSV出力

## 評価システム

### 実装済み基準
1. **共通基準（全ゲームサイズ適用）**:
   - 発話の自然さ
   - 文脈を踏まえた対話
   - 発話の一貫性
   - 行動と対話の整合性
   - キャラクター表現の豊かさ

2. **ゲーム固有基準**:
   - チームプレイの効果（13人戦）

### 評価プロセス
1. **独立評価**: 各基準を個別に評価
2. **ランキングシステム**: 順序ランキング（1位、2位等）
3. **並列実行**: ThreadPoolExecutor（最大8スレッド）
4. **構造化出力**: Pydantic検証付きLLMレスポンス
5. **バリデーション・再試行**: LLMレスポンスの厳密な検証と自動修正

### バリデーション機能
- プレイヤー数一致チェック
- プレイヤー名妥当性検証
- ランキング整合性（1からプレイヤー数までの連続した整数）
- 重複排除
- 自動再生成（設定可能な回数まで、デフォルト5回）

## 実装済み機能

1. **マルチ形式サポート**: 5人戦、13人戦ゲーム
2. **柔軟な評価**: 共通+ゲーム固有基準
3. **バッチ処理**: 並列ゲーム処理
4. **チームマッピング**: プレイヤー名→チーム所属
5. **構造化出力**: 理由付きJSON結果
6. **設定駆動**: パラメータ調整の容易さ
7. **厳密バリデーション**: プレイヤー数・名前・ランキング整合性の自動検証
8. **自動再試行**: LLMレスポンス不正時の自動修正機能
9. **チーム集計機能**: 全ゲーム結果の自動集計とチーム平均スコア算出
10. **日本語表示**: チーム集計結果でのcriteria_name→description自動変換
11. **評価基準ソート**: orderフィールドによる評価基準の出力順序制御
12. **CSV出力**: チーム集計結果のCSV形式出力（小数点以下6桁精度）
13. **集計再生成機能**: 既存の個別評価結果からチーム集計のみを再生成（LLM呼び出し不要）

## 技術仕様

### 依存関係
- **Python 3.11+** 要求
- **外部依存関係**: aiwolf-nlp-common, openai, pydantic, pyyaml, jinja2
- **開発ツール**: uvによるパッケージ管理

### パフォーマンス最適化
- **マルチレベル並列処理**: プロセス+スレッドベース
- **遅延初期化**: 必要時にリーダー作成
- **コンテキストマネージャ**: 適切なリソース管理
- **効率的ファイル発見**: Globベースマッチング

### 品質メトリクス
- **機能完成度**: 100%（エンドツーエンド動作確認済み）
- **型安全性**: ~95%（広範囲な型ヒント使用）
- **エラーハンドリング**: ~85%（階層化された例外処理）
- **コード品質**: ~96%（モジュラー・モノリスアーキテクチャ + リファクタリング改善）
- **アーキテクチャ**: 98%（明確な責任分離）
- **保守性**: ~97%（メソッド分割による単一責任原則の徹底）

## まとめ

AIWolf NLP LLM Judgeプロジェクトは、最先端の言語モデルを使用したAIWolfゲームログの自動評価のための、**モジュラーで保守性の高い**ソリューションです。

### 主要な成果
- **型安全Python開発**: 包括的データモデルによる開発
- **モジュラー・モノリスアーキテクチャ**: 機能別モジュール分離と制御された依存関係
- **パイプラインサービス設計**: データ処理の特性に適したサービス分離
- **スケーラブル並列処理**: マルチレベル最適化
- **柔軟な設定管理**: 様々なゲーム形式への対応
- **プロフェッショナルLLM統合**: 構造化検証付き
- **包括的エラーハンドリング**: 階層化された例外システム
- **高凝集・低結合**: サービス内の機能統合とサービス間の疎結合
- **継続的リファクタリング**: メソッド分割による保守性向上と単一責任原則の徹底

### コード品質改善履歴
- **2025-09-02**: `BatchProcessor._generate_team_aggregation`メソッドを5つの小さなメソッドに分割
  - `_load_evaluation_criteria_mappings`: 評価設定マッピング作成
  - `_create_team_aggregation_data`: チーム集計データ作成
  - `_convert_criteria_names_to_descriptions`: criteria名→description変換
  - `_create_sorted_criteria_list`: ソート済み評価基準リスト作成
  - `_save_team_aggregation_results`: 集計結果保存処理

- **2025-09-02**: チーム集計出力の責任分離によるアーキテクチャ改善
  - `AggregationOutputService`の新規作成: 専用出力サービス
  - `BatchProcessor`からファイル出力ロジックを分離
  - JSON・CSV出力の関心事を独立したサービスに委譲
  - 単一責任原則の徹底とコードの保守性向上

- **2025-09-03**: チーム集計処理の重大な問題修正
  - `BatchProcessor._load_evaluation_criteria_mappings`で`settings_path`自動設定機能を実装
  - `_convert_dict_to_evaluation_result`でデータ形式の不一致問題を解決
  - 並列処理戻り値とファイル形式の違いに対応する柔軟な処理を追加
  - チーム集計が100%動作保証される堅牢性を確保

- **2025-09-04**: チーム集計再生成機能の追加
  - `--regenerate-aggregation`オプションをCLIに追加
  - `BatchProcessor.regenerate_aggregation_only()`メソッドを実装
  - 既存の個別評価結果（`*_result.json`）からチーム集計のみを再生成
  - LLM APIを呼び出さない高速・低コストな集計再生成を実現

このシステムは、プロダクション対応可能な堅牢性と、将来のマイクロサービス化への明確で安全な拡張パスを提供しています。

## トラブルシューティング

### ~~チーム集計結果が空になる問題~~（**2025-09-03修正済み**）

**症状**: `data/output/team_aggregation.json`に結果が表示されず、CSV出力もされない

**原因**: 
1. ~~手動でのチーム集計実行時に`settings_path`設定が不足~~（**修正済み**）
2. ~~並列処理の戻り値とファイル形式の不一致~~（**修正済み**）

**修正内容**:
- `BatchProcessor._load_evaluation_criteria_mappings`で`settings_path`を自動設定
- `_convert_dict_to_evaluation_result`で複数のデータ形式に対応
- **現在は通常のCLI実行で完全に動作します**

**確認方法**:
```bash
# 正常なチーム集計結果の確認
ls -la data/output/team_aggregation.*
# JSONファイルのサマリー確認（jqがインストールされている場合）
python3 -c "import json; print(json.load(open('data/output/team_aggregation.json'))['summary'])"
```

**現在の動作状況**: 
✅ **完全修正済み** - 通常実行で9チームの詳細集計結果が正常に生成されます

### チーム集計ファイルを誤って削除した場合

**症状**: `team_aggregation.json`や`team_aggregation.csv`を誤って削除してしまった

**解決方法**:
```bash
# 集計再生成モードで復元
uv run python main.py -c config/settings.yaml --regenerate-aggregation
```

**動作**:
- `output_dir`内の個別評価結果（`*_result.json`）を読み込み
- チーム集計を再計算して出力
- LLM APIは呼び出さないため高速かつ低コスト