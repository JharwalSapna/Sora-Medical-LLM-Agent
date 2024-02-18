# Import necessary libraries
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from Custom_Agent_with_memory import agent_executor

# Set up Streamlit app
def main():
    # Set page configuration, including title and icon
    st.set_page_config(page_title="Medical Chatbot", page_icon="🏥")
    st.title("Sora - Medical AI Chatbot")


    # Initializing chat history if not already present in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a Sora, your Medical Assistant. I can assist you regarding your health or any medical inquiries you might have."),
        ]
    # Get user input using Streamlit chat_input
    user_query = st.chat_input("Ask a medical question:")


    # Process user query and update chat history
    if user_query:
        # Appending user's message to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        # Execute the agent to get a response
        response = agent_executor.run(user_query)
        # Appending agent's response to chat history
        st.session_state.chat_history.append(AIMessage(content=response))
       
    
    # Display conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            # Display AI messages in a chat message box
            with st.chat_message("AI", avatar="👩🏻‍⚕️"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            # Display human messages in a chat message box
            with st.chat_message("Human", avatar="👩🏻"):
                st.write(message.content)

# Run the Streamlit app when the script is executed
if __name__ == "__main__":
    main()




