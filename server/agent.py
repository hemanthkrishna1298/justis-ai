# %%
import os

from dotenv import load_dotenv
load_dotenv()

# %% [markdown]
# # Utilities

# %%
def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

# %% [markdown]
# # Graph

# %%
from typing import Annotated, Literal
from langgraph.graph import MessagesState
from langgraph.graph.message import AnyMessage, add_messages

# class ChatState(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]

class ChatState(MessagesState):
    chat_control: Literal["ai", "human_expert"]

# %%
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: ChatState, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            client_id = configuration.get("client_id", None)
            state = {**state, "user_info": client_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

# %% [markdown]
# ### Primary Assistant

# %%
from pydantic import BaseModel, Field

class ToIssueIdentifyingAssistant(BaseModel):
    """Transfers work to a specialized assistant to identify the legal issue to provide information on."""

    request: str = Field(
        description="Any additional information or requests from the user."
    )

class ToEscalationAssistant(BaseModel):
    """Transfers work to an assistant that escalates the issue to a human expert."""

    request: str = Field(
        description="Any additional information or requests from the user."
    )

class ToFormFillingAssistant(BaseModel):
    """Transfers work to a specialized assistant that helps the user fill a legal document."""

    request: str = Field(
        description="Any additional information or requests from the user."
    )

# %%
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))

primary_assisant_prompt = ChatPromptTemplate(
    [
        ("system",
         """You are a helpful assistant helping a user by helping them get to the right legal aid information and/or help them fill up legal forms. These are your guidelines:
         1. Your primary goal is to converse with the user and help transfer them to the right agent based on what they require. There are four possibilities:
            a. The user needs some legal help. In this case, transfer to the issue identifying agent to help identify the specific legal issue they are facing.
            b. The user needs to fill a legal document/form. In this case, transfer them to the form filling agent.
            c. The user needs to escalate the issue to a human expert. In this case, transfer them to the escalation agent.
            d. The user is just talking without it being clear that they need legal help. In this case, continue the conversation with them.
         2. Do not let the conversation stray off-topic. Do not explicitly mention your role of identifying the agent to transfer to.
         3. Only when you have successfully identified the type of agent to transfer to, call a function to transfer to the appropriate agent. Tell the user that you are transferring them to appropriate resources.
         4. If the user asks for help pertaining to a type outside of the specified types, tell them that you will escalate the issue to a human expert and escalate the conversation to a human."""
         ),
         ("placeholder", "{messages}"),
    ]
)

primary_assisant_tools = [
    ToIssueIdentifyingAssistant,
    ToEscalationAssistant,
    ToFormFillingAssistant
]

primary_assistant_runnable = primary_assisant_prompt | llm.bind_tools(primary_assisant_tools)

# %% [markdown]
# ### Issue Identifying Assistant

# %%
from typing import Literal

class ToLegalInfoAndGuidanceAssistant(BaseModel):
    """Transfers work to a specialized assistant to provide legal aid based on the issue identified."""

    issue: Literal["Housing Disputes", "Wage Theft", "Immigration Concerns"] = Field(
        description="The type of legal issue identified."
    )

# %%
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))

issue_identifying_prompt = ChatPromptTemplate(
    [
        ("system",
         """You are a helpful assistant helping a user by helping them get to the right legal aid information. These are your guidelines:
         1. Your primary goal is to converse with the user and identify the type of legal help they need. The legal help needed can be one of three types:
            a. Housing Disputes
            b. Wage Theft
            c. Immigration Concerns
         2. Do not let the conversation stray off-topic. Do not explicitly mention your role of identifying the type of legal help needed.
         3. Only when you have successfully identified the type of legal help needed, call a function to transfer to the appropriate legal help agent. Tell the user that you are transferring them to appropriate resources.
         4. If the user asks for help pertaining to a type outside of the specified three types, tell them that you will escalate the issue to a human expert and escalate the conversation to a human."""
         ),
         ("placeholder", "{messages}"),
    ]
)

issue_identifying_assistant_tools = [
    ToLegalInfoAndGuidanceAssistant
]

issue_identifying_assistant_runnable = issue_identifying_prompt | llm.bind_tools(issue_identifying_assistant_tools)

# %% [markdown]
# ### Form Filling Assistant

# %%
from langchain_core.tools import tool
import fitz



@tool
def fill_pdf_form(pdf_type: Literal["eviction_response", "wage_claim"], field_value_dict: dict) -> str:
    """Fill a PDF form with the given field values and return a message indicating the success of the form filling.
    Args:
        pdf_type (Literal["eviction_response", "wage_claim"]): The type of the PDF form.
        field_value_dict (dict): The field values to fill in the PDF form.
    Returns:
        str: A message indicating the success of the form filling.
    """


    def get_fields(pdf_document):
        # Extract field names, current values, and allowed values
        fields = []
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            for widget in page.widgets():
                field_name = widget.field_name

                if("#pageSet" in field_name):
                    continue

                allowed_values = None

                if widget.field_type_string == "CheckBox":
                    allowed_values = widget.button_states()['normal']
                elif widget.field_type_string == "ComboBox":
                    allowed_values = widget.choice_values

                fields.append({
                    "field_type": widget.field_type_string,
                    "field_name": field_name,
                    "field_label": widget.field_label,
                    "field_value": widget.field_value or "",
                    "allowed_values": allowed_values
                })

        return fields
    
    def write_to_pdf(pdf_document, fields):
        with fitz.open(pdf_document) as doc:
            for page in doc: 
                widgets = page.widgets()
                for widget in widgets:
                    matching_field = [field for field in fields if field["field_name"] == widget.field_name]

                    if(len(matching_field) == 0):
                        continue
                    
                    field = matching_field[0]
                    widget.field_value = field["value"]
                    widget.update()
            doc.saveIncr()
    
    pdf_path = f"{pdf_type}.pdf"
    pdf_document = fitz.open(pdf_path)
    fields = get_fields(pdf_document)

    # Update fields with provided values
    for field in fields:
        if field["field_name"] in field_value_dict:
            field["value"] = field_value_dict[field["field_name"]]
        else:
            field["value"] = ""

    write_to_pdf(pdf_path, fields)
    return "PDF form filled successfully."

# %%
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))

form_filling_assistant_prompt = ChatPromptTemplate(
    [
        ("system",
         """You are a helpful assistant helping a user by guiding them through filling a legal form/document and filling it for them. 
         1. Help a user fill a legal form/document by following these steps:
            a. Identify which form/document needs to be filled and call a function to get back the fields that need to be filled.
            b. Converse with the user and ask for the necessary information required to fill the form.
            c. Fill the form with the information provided by the user.
            d. Once the form is filled, tell the user that the form has been filled and you are providing them with the filled form.
         2. Do not let the conversation stray off-topic. Do not explicitly mention your role as the form filling assistant."""
         ),
         ("placeholder", "{messages}"),
    ]
)

form_filling_assistant_tools = [
    get_pdf_fields,
    fill_pdf_form
]

form_filling_assistant_runnable = form_filling_assistant_prompt | llm.bind_tools(form_filling_assistant_tools)

# %% [markdown]
# ### Legal Info and Guidance Assistant

# %%
# from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
import requests
from bs4 import BeautifulSoup
import urllib
import math

@tool
def legal_info_getter(search_query: str)->str:
    """Get legal information based on the search query.
    Args:
        search_query (str): The search query.
    Returns:
        str: The search results containing legal information.
    """

#     search_engine = DuckDuckGoSearchResults()

#     return search_engine.invoke(search_query)

# def get_legal_info(search_query):

    query = search_query
    def get_search_results(query, num_pages=5):
        """Retrieve links to articles from search results for the query."""
        
        
        base_url = "https://www.law.cornell.edu"
        search_url_template = f"{base_url}/search/wex/{{query}}?page={{page}}"

        article_links = []
        for page in range(num_pages):
            search_url = search_url_template.format(query=query, page=page)
            response = requests.get(search_url)
            if response.status_code != 200:
                print(f"Failed to retrieve page {page + 1} of search results.")
                continue
            soup = BeautifulSoup(response.text, "html.parser")
            # Find all search result items in the page
            results = soup.select("li.search-result h3.title a")
            # Extract the href attribute for each result
            for result in results:
                article_links.append(result["href"])
            # time.sleep(1)  # Pause to avoid overloading the server
        return article_links

    def get_article_content(url):
        """Retrieve and parse the main content from an article page."""
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve article at {url}")
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        # Extract the main content from the article page
        main_content_div = soup.find("div", id="main-content")
        if not main_content_div:
            return None
        content = []
        # Extract paragraphs within the main content div
        for paragraph in main_content_div.find_all("p"):
            content.append(paragraph.get_text())
        return "\n".join(content)

    def legal_info_call(query):
        """Perform the search and retrieve the content of the top articles."""
        
        num_articles = 10
        # Step 1: Get the search results (first num_articles results)
        article_links = get_search_results(query, num_pages=math.ceil(num_articles / 10))

        # Step 2: Extract content from each article
        articles_content = {}
        for link in article_links:
            print(f"Retrieving content from {link}")
            content = get_article_content(link)
            if content:
                articles_content[link] = content
            # time.sleep(1)  # Pause to avoid overloading the server

        return articles_content
    
    articles = legal_info_call(urllib.parse.quote_plus(query))

    send_string= ""
    
    for url, content in articles.items():
        send_string+=content

    return send_string

# %%
legal_info_and_guidance_prompt = ChatPromptTemplate(
    [
        ("system",
         """You are a helpful assistant providing legal aid information based on the user's query. 
         1. Understand the user's legal information needs based on the conversation.
         2. Use the provided tool to appropriately search for legal information based on the user's needs.
         3. Provide the user with the relevant legal information obtained from the search.
         4. Do not let the conversation stray off-topic. Do not explicitly mention your role as the legal information assistant."""
         ),
         ("placeholder", "{messages}"),
    ]
)

legal_info_and_guidance_tools = [
    legal_info_getter
]

legal_info_and_guidance_runnable = legal_info_and_guidance_prompt | llm.bind_tools(legal_info_and_guidance_tools)

# %% [markdown]
# ### Escalation Assistant

# %%
class EscalationAssistant(Assistant):
    def __call__(self, state: ChatState, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            client_id = configuration.get("client_id", None)
            state = {**state, "user_info": client_id}
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result, "chat_control": "human_expert"}

# %%
class ToHumanExpert(BaseModel):
    """Transfers work to the human expert, provides them a summary of the conversation so far."""

    summary: str = Field(
        description="Summary of the conversation so far."
    )

# %%
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))

escalation_assistant_prompt = ChatPromptTemplate(
    [
        ("system",
         """You are a helpful assistant that escalates the issue to a human expert by summarizing the conversation so far. 
         1. Summarize the conversation with the user.
         2. Provide the summary to a human expert for further assistance.
         3. Inform the user that their issue is being escalated to a human expert."""
         ),
         ("placeholder", "{messages}"),
    ]
)

escalation_assistant_tools = [
    ToHumanExpert
]

escalation_assistant_runnable = escalation_assistant_prompt | llm.bind_tools(escalation_assistant_tools)

# %% [markdown]
# ### Human Node

# %%
def human_node(state: ChatState)->ChatState:
    return {"chat_control": "ai"}

# %% [markdown]
# ### Graph building

# %%
from typing import Callable

from langchain_core.messages import ToolMessage


def create_entry_node(assistant_name: str) -> Callable:
    def entry_node(state: ChatState) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}."
                    "Reflect on the above conversation between the host assistant and the user."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ]
        }

    return entry_node

# %%
from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph, START

def cond_edge_router(
    state: ChatState,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToEscalationAssistant.__name__:
            return "enter_escalation_assistant"
        elif tool_calls[0]["name"] == ToIssueIdentifyingAssistant.__name__:
            return "enter_issue_identifying_assistant"
        elif tool_calls[0]["name"] == ToFormFillingAssistant.__name__:
            return "enter_form_filling_assistant"
        elif tool_calls[0]["name"] == ToLegalInfoAndGuidanceAssistant.__name__:
            return "enter_legal_info_and_guidance_assistant"
        elif tool_calls[0]["name"] == ToHumanExpert.__name__:
            return "human_expert"

    raise ValueError("Invalid route")

def custom_tools_condition(state: ChatState):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"]=="legal_info_getter":
            return "legal_info_tool"
        elif tool_calls[0]["name"]=="get_pdf_fields" or tool_calls[0]["name"]=="fill_pdf_form":
            return "form_filling_tool"
    
    raise ValueError("Invalid route")

# %%
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode

builder = StateGraph(ChatState)

builder.add_node("primary_assistant", Assistant(primary_assistant_runnable))
builder.add_node("enter_issue_identifying_assistant", create_entry_node("Issue Identifying Assistant"))
builder.add_node("issue_identifying_assistant", Assistant(issue_identifying_assistant_runnable))
builder.add_node("enter_legal_info_and_guidance_assistant", create_entry_node("Legal Info and Guidance Assistant"))
builder.add_node("legal_info_and_guidance_assistant", Assistant(legal_info_and_guidance_runnable))
builder.add_node("legal_info_tool", ToolNode(legal_info_and_guidance_tools))
builder.add_node("enter_form_filling_assistant", create_entry_node("Form Filling Assistant"))
builder.add_node("form_filling_assistant", Assistant(form_filling_assistant_runnable))
builder.add_node("form_filling_tool", ToolNode(form_filling_assistant_tools))
builder.add_node("enter_escalation_assistant", create_entry_node("Escalation Assistant"))
builder.add_node("escalation_assistant", Assistant(escalation_assistant_runnable))
builder.add_node("human_expert", human_node)

builder.add_edge(START, "primary_assistant")
builder.add_conditional_edges("primary_assistant", cond_edge_router, ["enter_issue_identifying_assistant", "enter_form_filling_assistant", "enter_escalation_assistant", END])
builder.add_conditional_edges("issue_identifying_assistant", cond_edge_router, ["enter_legal_info_and_guidance_assistant", "enter_escalation_assistant", END])
builder.add_conditional_edges("legal_info_and_guidance_assistant", custom_tools_condition, ["legal_info_tool", END])
builder.add_edge("legal_info_tool", "legal_info_and_guidance_assistant")
builder.add_conditional_edges("form_filling_assistant", custom_tools_condition, ["form_filling_tool", END])
builder.add_edge("form_filling_tool", "form_filling_assistant")
builder.add_edge("enter_issue_identifying_assistant", "issue_identifying_assistant")
builder.add_edge("enter_legal_info_and_guidance_assistant", "legal_info_and_guidance_assistant")
builder.add_edge("enter_form_filling_assistant", "form_filling_assistant")
builder.add_edge("enter_escalation_assistant", "escalation_assistant")
builder.add_conditional_edges("escalation_assistant", cond_edge_router, ["human_expert", END])
builder.add_edge("human_expert", END)

memory=MemorySaver()
justis_ai_graph = builder.compile(checkpointer=memory, interrupt_before=["human_expert"])
# justis_ai_graph = builder.compile()


# # %%
# from IPython.display import Image, display

# try:
#     display(Image(justis_ai_graph.get_graph(xray=True).draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass
import uuid

questions = [
    "Hi!",
    "Can you help me with something?",
    "I just got evicted by my landlord and I think it was unfair.",
]

thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "client_id": "1234",
        "thread_id": thread_id,
    }
}

_printed = set()

for i, question in enumerate(questions):
    if i==0:
        events = justis_ai_graph.stream(
        {"messages": ("user", question), "chat_control": "ai"}, config, stream_mode="values"
        )
    else:
        events = justis_ai_graph.stream(
            {"messages": ("user", question)}, config, stream_mode="values"
        )

    for event in events:
        _print_event(event, _printed)
        print(event)
