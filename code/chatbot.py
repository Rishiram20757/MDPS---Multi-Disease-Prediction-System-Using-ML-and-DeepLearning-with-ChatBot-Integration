import streamlit as st
import ollama
from time import time

def show_chatbot():
    st.caption("Powered by Ollama (monotykamary/medichat-llama3)")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm a Medcare AI assistant. How can I help you today?"}
        ]
        st.session_state.waiting_for_input = True
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.container():
                st.markdown(f"**You:** {message['content']}")
        else:
            with st.container():
                st.markdown(f"**Assistant:** {message['content']}")

    # Chat input using standard text_input
    if st.session_state.waiting_for_input:
        col1, col2 = st.columns([4, 1])
        with col1:
            prompt = st.text_input("Ask your medical question...", key="chat_input", label_visibility="collapsed")
        with col2:
            if st.button("Send") or prompt:
                if prompt:  # Only process if there's actual input
                    st.session_state.waiting_for_input = False
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.experimental_rerun()

    # Generate assistant response
    if not st.session_state.waiting_for_input and st.session_state.messages[-1]["role"] == "user":
        user_message = st.session_state.messages[-1]
        
        with st.spinner("Thinking..."):
            try:
                response = ollama.chat(
                    model="monotykamary/medichat-llama3",
                    messages=st.session_state.messages,
                    stream=False
                )
                full_response = response['message']['content']
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })
                
                st.session_state.waiting_for_input = True
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.session_state.waiting_for_input = True
                st.experimental_rerun()

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm a medical AI assistant. How can I help you today?"}
        ]
        st.session_state.waiting_for_input = True
        st.experimental_rerun()