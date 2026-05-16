from typing import Self, Optional, TypeAlias
from pydantic import BaseModel, Field
from src.evaluation.models.llm_response import EvaluationLLMResponse, EvaluationElement


class EvaluationResultElement(BaseModel):
    """チーム情報を含む評価結果要素."""

    player_name: str = Field(description="評価対象者の名前")
    reasoning: str = Field(description="各評価対象に対する順位付けの理由")
    ranking: int = Field(description="評価対象者の順位(他のプレイヤーとの重複はなし)")
    team: str = Field(description="プレイヤーの所属チーム名")

    @classmethod
    def from_evaluation_element(cls, element: EvaluationElement, team: str) -> Self:
        """EvaluationElementからチーム情報付きの結果要素を作成

        Args:
            element: LLMからの評価要素
            team: チーム名

        Returns:
            チーム情報付きの評価結果要素
        """
        return cls(
            player_name=element.player_name,
            reasoning=element.reasoning,
            ranking=element.ranking,
            team=team,
        )

    def to_dict(self) -> dict:
        """要素を辞書形式に変換

        Returns:
            辞書形式の要素
        """
        return {
            "player_name": self.player_name,
            "team": self.team,
            "ranking": self.ranking,
            "reasoning": self.reasoning,
        }


class CriteriaEvaluationResult(list[EvaluationResultElement]):
    """単一の評価基準に対する結果を表すクラス."""

    def __init__(
        self,
        criteria_name: str,
        elements: Optional[list[EvaluationResultElement]] = None,
    ):
        """初期化

        Args:
            criteria_name: 評価基準名
            elements: 評価結果要素のリスト
        """
        super().__init__(elements or [])
        self.criteria_name = criteria_name

    @classmethod
    def from_llm_response(
        cls,
        criteria_name: str,
        llm_response: EvaluationLLMResponse,
        agent_to_team_mapping: dict[str, str],
    ) -> Self:
        """LLMレスポンスから評価結果を作成

        Args:
            criteria_name: 評価基準名
            llm_response: LLMからの評価レスポンス
            agent_to_team_mapping: エージェント名→チーム名のマッピング

        Returns:
            評価結果
        """
        result_elements = []
        for element in llm_response.rankings:
            team = agent_to_team_mapping.get(element.player_name, "unknown")
            result_element = EvaluationResultElement.from_evaluation_element(
                element, team
            )
            result_elements.append(result_element)

        return cls(criteria_name=criteria_name, elements=result_elements)

    def to_dict(self) -> dict:
        """評価結果を辞書形式に変換

        Returns:
            辞書形式の評価結果 {"rankings": [...]}
        """
        return {"rankings": [elem.to_dict() for elem in self]}


class EvaluationResult(list[CriteriaEvaluationResult]):
    """全評価基準の結果を管理するクラス（リストを継承）."""

    def append(self, criteria_result: CriteriaEvaluationResult) -> None:
        """評価結果を追加（重複チェック付き）

        Args:
            criteria_result: 追加する評価結果

        Raises:
            ValueError: 同一のcriteria_nameが既に存在する場合
        """
        if self.get_result_by_criteria_name(criteria_result.criteria_name) is not None:
            raise ValueError(
                f"Criteria '{criteria_result.criteria_name}' already exists in EvaluationResult"
            )
        super().append(criteria_result)

    def add_result(self, criteria_result: CriteriaEvaluationResult) -> None:
        """評価結果を安全に追加（appendのエイリアス）

        Args:
            criteria_result: 追加する評価結果

        Raises:
            ValueError: 同一のcriteria_nameが既に存在する場合
        """
        self.append(criteria_result)

    def get_result_by_criteria_name(
        self, criteria_name: str
    ) -> Optional[CriteriaEvaluationResult]:
        """指定された評価基準名の結果を取得

        Args:
            criteria_name: 評価基準名

        Returns:
            該当する評価結果、見つからない場合はNone
        """
        for result in self:
            if result.criteria_name == criteria_name:
                return result
        return None

    def get_criteria_names(self) -> list[str]:
        """全ての評価基準名を取得

        Returns:
            評価基準名のリスト
        """
        return [result.criteria_name for result in self]

    def to_dict(self) -> dict:
        """評価結果を辞書形式に変換

        Returns:
            辞書形式の評価結果 {criteria_name: {"rankings": [...]}}
        """
        return {
            criteria_result.criteria_name: criteria_result.to_dict()
            for criteria_result in self
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationResult":
        """辞書形式の評価結果を EvaluationResult に復元.

        ``data`` は次の形式を受け入れる:
          - ``{"evaluations": {criteria_name: {"rankings": [...]}}}`` (ファイル形式)
          - ``{criteria_name: {"rankings": [...]}}`` (並列処理戻り値の形)
        """
        evaluations_data = data.get("evaluations", data)

        evaluation_result = cls()
        for criteria_name, criteria_data in evaluations_data.items():
            elements = [
                EvaluationResultElement(
                    player_name=ranking_data["player_name"],
                    reasoning=ranking_data["reasoning"],
                    ranking=ranking_data["ranking"],
                    team=ranking_data["team"],
                )
                for ranking_data in criteria_data.get("rankings", [])
            ]
            criteria_result = CriteriaEvaluationResult(
                criteria_name=criteria_name, elements=elements
            )
            evaluation_result.append(criteria_result)
        return evaluation_result


# 型エイリアス定義
TeamResultsDict: TypeAlias = dict[str, dict[str, list[EvaluationResultElement]]]


class TeamAggregator(TeamResultsDict):
    """チーム別集計データを管理するクラス

    構造: {team_name: {criteria_name: [EvaluationResultElement]}}
    """

    def add_game_result(self, evaluation_result: EvaluationResult) -> None:
        """ゲーム結果をチーム別に集約

        Args:
            evaluation_result: 単一ゲームの評価結果
        """
        for criteria_result in evaluation_result:
            criteria_name = criteria_result.criteria_name

            for element in criteria_result:
                team = element.team

                # チームが存在しない場合は初期化
                if team not in self:
                    self[team] = {}

                # 評価基準が存在しない場合は初期化
                if criteria_name not in self[team]:
                    self[team][criteria_name] = []

                # 評価要素を追加
                self[team][criteria_name].append(element)

    def calculate_team_averages(self) -> dict[str, dict[str, float]]:
        """チーム別平均順位を算出

        Returns:
            {team_name: {criteria_name: average_ranking}}
        """
        team_averages = {}

        for team, criteria_dict in self.items():
            team_averages[team] = {}

            for criteria_name, elements in criteria_dict.items():
                if elements:  # 空でない場合
                    rankings = [element.ranking for element in elements]
                    team_averages[team][criteria_name] = sum(rankings) / len(rankings)
                else:
                    team_averages[team][criteria_name] = 0.0

        return team_averages

    def get_team_count_by_criteria(self) -> dict[str, dict[str, int]]:
        """チーム別・評価基準別のサンプル数を取得

        Returns:
            {team_name: {criteria_name: sample_count}}
        """
        team_counts = {}

        for team, criteria_dict in self.items():
            team_counts[team] = {}

            for criteria_name, elements in criteria_dict.items():
                team_counts[team][criteria_name] = len(elements)

        return team_counts
