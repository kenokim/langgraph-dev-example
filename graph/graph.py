import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from typing import Sequence, Annotated, TypedDict
from langchain_core.messages import BaseMessage

load_dotenv()  # .env 파일에서 환경 변수 로드

# 1) 모델 정의 ---------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3,
)

# 2) 그래프 정의 --------------------------------------------------------------
class State(TypedDict):
    # LangGraph가 메시지를 누적 관리할 수 있도록 add_messages 애너테이션 사용
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # create_react_agent가 필요로 하는 remaining_steps 필드 추가
    remaining_steps: int
    # 추가 입력 예시: 응답 언어 (필수가 아님)
    language: str | None

workflow = StateGraph(state_schema=State)


# 별도 도구 없이 가장 단순한 에이전트
agent_node = create_react_agent(
    model=llm,
    tools=[],
    prompt="",
    state_schema=State,  # custom State 사용
)

# 그래프에 agent 노드 연결
workflow.add_edge(START, "agent")
workflow.add_node("agent", agent_node)

# 3) 그래프 컴파일 ------------------------------------------------------------
app = workflow.compile() 