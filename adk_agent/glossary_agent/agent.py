"""第3回 宿題（発展 a）: 演習のエージェントを ADK で組み立て直す。

指示（INSTRUCTIONS）とツールは app/ のものをそのまま使い、組み立てる部分だけを書き換える。

使い方（リポジトリの直下で）:
    PYTHONPATH=. uv run --with google-adk adk run adk_agent/glossary_agent
"""

from google.adk.agents import Agent

from app import tools
from app.agent import INSTRUCTIONS
from app.config import MODEL

root_agent = Agent(
    model=MODEL,
    name="glossary_agent",
    instruction=INSTRUCTIONS,
    tools=[tools.search_glossary, tools.get_lecture_summary, tools.make_quiz],
)
