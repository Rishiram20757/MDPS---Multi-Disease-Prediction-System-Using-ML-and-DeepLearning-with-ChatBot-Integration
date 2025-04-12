import streamlit as st
import ollama
from time import time

def show_chatbot():
    st.caption("Powered by  Ollama (monotykamary/medichat-llama3)")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm glad you're here. How can I assist you today?"}
        ]
        st.session_state.start_time = time()
        st.session_state.waiting_for_input = True
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.container():
                st.markdown("**You:**")
                st.info(message["content"])
        else:
            with st.container():
                st.markdown("**Assistant:**")
                st.success(message["content"])
                if "response_time" in message:
                    st.caption(f"Response time: {message['response_time']:.2f}s")

    # Chat input - only show when waiting for input
    if st.session_state.waiting_for_input:
        prompt = st.text_input("Ask your medical question...", key="chat_input")
        
        if st.button("Send") or prompt:
            if prompt:  # Only process if there's actual input
                st.session_state.waiting_for_input = False
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.experimental_rerun()

    # Generate assistant response when needed
    if not st.session_state.waiting_for_input and st.session_state.messages[-1]["role"] == "user":
        user_message = st.session_state.messages[-1]
        
        with st.spinner("Thinking..."):
            start_time = time()
            full_response = ""
            message_placeholder = st.empty()
            
            # Stream the response
            for chunk in ollama.chat(
                model="monotykamary/medichat-llama3",
                messages=st.session_state.messages,
                stream=True
            ):
                full_response += chunk['message']['content']
                message_placeholder.markdown(f"**Assistant:**\n\n{full_response}▌")
            
            message_placeholder.markdown(f"**Assistant:**\n\n{full_response}")
            response_time = time() - start_time
            st.caption(f"Response time: {response_time:.2f}s")
            
            # Add to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "response_time": response_time
            })
            
            st.session_state.waiting_for_input = True
            st.experimental_rerun()

    # Session info in sidebar
    with st.sidebar:
        st.subheader("Session Info")
        st.write(f"Duration: {(time() - st.session_state.start_time)/60:.1f} minutes")
        st.write(f"Messages: {len([m for m in st.session_state.messages if m['role'] == 'user'])}")
        if st.button("Clear Chat"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi! I'm glad you're here. How can I assist you today?"}
            ]
            st.session_state.start_time = time()
            st.session_state.waiting_for_input = True
            st.experimental_rerun()

if __name__ == "__main__":
    show_chatbot()