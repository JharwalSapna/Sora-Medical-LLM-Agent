# Import necessary libraries and modules
from langchain_openai import ChatOpenAI
from langchain import LLMChain
from langchain.agents import AgentExecutor, AgentOutputParser, Tool
from langchain.agents import LLMSingleActionAgent
from langchain.tools import DuckDuckGoSearchRun
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferWindowMemory
from typing import List, Union
from langchain.prompts import StringPromptTemplate
import re
from dotenv import load_dotenv

load_dotenv()

# Define the DuckDuckGo search tool
search = DuckDuckGoSearchRun()

# Wrapper function to use the DuckDuckGo search tool
def duck_wrapper(input_text):
    # Use DuckDuckGo to search for information on WebMD
    search_results = search.run(f"site:webmd.com {input_text}")
    return search_results

# Define available tools
tools = [
    Tool(
        name = "Search WebMD",
        func=duck_wrapper,
        description="useful for when you need to answer medical and pharmalogical questions"
    ),
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]

# Set up a prompt template for conversations
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples) and format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


# Define a custom output parser for agent responses, extending from AgentOutputParser
class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            # If the output contains "Final Answer:", create an AgentFinish instance
            return AgentFinish(
                # Return values are typically a dictionary with a single `output` key
                # It is not recommended to change this structure
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # Raise an error if the regular expression does not match the LLM output
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        # Extract the action and action input from the regular expression match
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the parsed action and action input as an AgentAction instance
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()

# Set up LLM model
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Set up a prompt template for conversations with history
template_with_history = """Answer the following questions as best you can, you are Sora, a real medical professional who always speaks compasionately mirroring user style and language perfectly. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as a compasionate medical professional when giving your final answer. If the condition is serious advise they speak to a doctor.

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}"""

# Create a CustomPromptTemplate using a template with history
prompt_with_history = CustomPromptTemplate(
    template=template_with_history,
    tools=tools,
    # Omitting `agent_scratchpad`, `tools`, and `tool_names` because they are generated dynamically
    # Including `intermediate_steps` as it is a necessary variable
    input_variables=["input", "intermediate_steps", "history"]
)

# Set up an LLMChain with the specified language model and prompt template with history
llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)

# Extract tool names from the list of tools
tool_names = [tool.name for tool in tools]

# Create an LLMSingleActionAgent with conversation history and tools
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],    # Specify the stopping condition for the agent
    allowed_tools=tool_names    # Provide the list of allowed tools for the agent
)

# Set up a conversation memory with a buffer window of size 2
memory = ConversationBufferWindowMemory(k=2)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory
    )


# Example usage: agent_executor.run("How can I treat a thyroid?")












