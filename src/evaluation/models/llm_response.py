"""LLMレスポンス関連のデータモデル."""

from typing import Self
from pydantic import BaseModel, Field, model_validator


class EvaluationElement(BaseModel):
    """個々のプレイヤーに対する評価要素."""

    player_name: str = Field(description="評価対象者の名前")
    reasoning: str = Field(description="各評価対象に対する順位付けの理由")
    ranking: int = Field(
        description="評価対象者の順位(他のプレイヤーとの重複はなし)", ge=1
    )


class EvaluationLLMResponse(BaseModel):
    """LLMからの評価レスポンス全体."""

    rankings: list[EvaluationElement] = Field(description="各プレイヤーに対する評価")

    @model_validator(mode="after")
    def validate_rankings_consistency(self) -> Self:
        """ランキングの整合性を検証.

        競技ランキング(1224)方式:
        - 最上位は必ず 1。
        - 同順位（タイ）を許容。
        - タイの分は次の順位を飛ばす。例: 1, 1, 1, 4, 5 は valid、1, 1, 1, 2, 3 は invalid。
        - 全員同順位（区別を放棄した評価）は invalid。少なくとも2階位は必要。

        同順位を許容する理由: 同質な reasoning が複数得られた場合、強制的に
        1/2/3 と差別化することで生じる「相対効果アーティファクト」を防ぐ。
        """
        # 空のランキングリストの検証
        if not self.rankings:
            raise ValueError("ランキングリストは空にできません")

        ranking_values = [elem.ranking for elem in self.rankings]
        sorted_ranks = sorted(ranking_values)
        n = len(sorted_ranks)

        # 最上位は必ず 1（昇順ソートの先頭が 1 でなければ不整合）
        if sorted_ranks[0] != 1:
            raise ValueError(
                f"最上位の順位は1である必要があります。実際: {sorted_ranks[0]}"
            )

        # 競技ランキング方式の整合性:
        # 各順位 r は「r より厳密に小さい値の数 + 1」と一致するべき。
        # 例: [1,1,1,4,5] では 4 の前に 3 人いるので 3+1=4 で OK。
        # [1,1,1,2,3] では 2 の前に 3 人いるので 3+1=4 が期待されるが 2 → invalid。
        for r in sorted_ranks:
            expected = sum(1 for x in sorted_ranks if x < r) + 1
            if r != expected:
                raise ValueError(
                    f"順位 {r} は競技ランキング方式と整合しません（期待値: {expected}）。"
                    f"同順位の分は次の順位を飛ばしてください（例: 1,1,1,4,5）。"
                    f"実際の順位列: {sorted_ranks}"
                )

        # 全員同順位（評価放棄）は禁止
        if len(set(ranking_values)) < 2 and n > 1:
            raise ValueError(
                "全員を同順位にすることはできません。少なくとも2つの異なる順位が必要です。"
            )

        # 順位値が N を超えないこと（上限チェック）
        if max(sorted_ranks) > n:
            raise ValueError(
                f"順位は1から{n}までの範囲である必要があります。"
                f"実際の最大値: {max(sorted_ranks)}"
            )

        return self

    @classmethod
    def create_with_validation(
        cls,
        rankings: list[EvaluationElement],
        player_count: int,
        valid_player_names: set[str],
    ) -> Self:
        """バリデーション付きでインスタンスを作成

        Args:
            rankings: ランキングデータ
            player_count: 期待するプレイヤー数
            valid_player_names: 有効なプレイヤー名のセット

        Returns:
            検証済みのEvaluationLLMResponseインスタンス

        Raises:
            ValueError: バリデーションに失敗した場合
        """
        # プレイヤー数の検証
        if len(rankings) != player_count:
            raise ValueError(
                f"ランキング数（{len(rankings)}）がプレイヤー数（{player_count}）と一致しません"
            )

        # プレイヤー名の検証
        response_player_names = {elem.player_name for elem in rankings}
        invalid_names = response_player_names - valid_player_names
        if invalid_names:
            raise ValueError(
                f"無効なプレイヤー名が含まれています: {invalid_names}. "
                f"有効な名前: {valid_player_names}"
            )

        missing_names = valid_player_names - response_player_names
        if missing_names:
            raise ValueError(f"不足しているプレイヤー名があります: {missing_names}")

        # 基本的なPydanticバリデーションを実行
        return cls(rankings=rankings)

    def __iter__(self):
        """リストのように反復処理可能にする."""
        return iter(self.rankings)

    def __len__(self):
        """リストのようにlen()を使用可能にする."""
        return len(self.rankings)

    def __getitem__(self, index):
        """リストのようにインデックスアクセス可能にする."""
        return self.rankings[index]
