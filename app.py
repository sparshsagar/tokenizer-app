import streamlit as st
from tokenizers.gpt_tokenizer import GPT4Tokenizer
from tokenizers.wordpiece_tokenizer import WordPieceTokenizer
from tokenizers.unigram_tokenizer import UnigramTokenizer

# Add the sidebar for tokenizer selection
tokenizer_type = st.sidebar.selectbox(
    "Select Tokenizer",
    ("BPE", "WordPiece", "Unigram")
)

# Initialize tokenizer based on selection
if tokenizer_type == "BPE":
    if "tokenizer" not in st.session_state or not isinstance(st.session_state.tokenizer, GPT4Tokenizer):
        st.session_state.tokenizer = GPT4Tokenizer()
        st.session_state.show_decode = True
elif tokenizer_type == "WordPiece":
    if "tokenizer" not in st.session_state or not isinstance(st.session_state.tokenizer, WordPieceTokenizer):
        st.session_state.tokenizer = WordPieceTokenizer(vocab_size=400)
        st.session_state.show_decode = False
elif tokenizer_type == "Unigram":
    st.warning("Unigram Tokenizer functionality is coming soon!")
    # Optionally disable further actions if you want
    st.stop()  # This stops further execution of the app
    # if "tokenizer" not in st.session_state or not isinstance(st.session_state.tokenizer, UnigramTokenizer):
    #     st.session_state.tokenizer = UnigramTokenizer(2000, 5,shrink_multiplier=0.05, sub_em_steps=3)
    #     st.session_state.show_decode = False

tokenizer = st.session_state.tokenizer

st.title("Tokenizer Visualization App")

# Input text from the user
train_text = st.text_area("Enter text to tokenize", "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception.")
test_text = st.text_area("Enter text to tokenize", "test sentence")

# Add Train Tokenizer button
if st.button("Train Tokenizer"):
    if tokenizer_type == "Unigram":
        tokenizer.train_tokenizer([train_text])  # Pass the training text as a list
        st.success(f"{tokenizer_type} Tokenizer trained on the text!")
    else:
        tokenizer.train(train_text)
        st.success(f"{tokenizer_type} Tokenizer trained on the text!")

# Add Tokenize button
if st.button("Tokenize"):
    col1, col2 = st.columns(2)

    # Define a pre-determined set of hues
    hues = [f"hsl({i}, 100%, 70%)" for i in range(0, 360, 15)]  # Hues spaced every 15 degrees
    hue_index = 0  # Start hue index from 0

    with col1:
        st.subheader("Encoded Tokens")
        # Tokenize the input text and store it in session state
        st.session_state.tokenized_words = st.session_state.tokenizer.encode(test_text)
        st.session_state.token_length = len(st.session_state.tokenized_words)

        st.write("Token length:", st.session_state.token_length)

        # Display tokenized words with colored bounding boxes and curved borders
        colored_output = ""
        for idx, word in enumerate(st.session_state.tokenized_words):
            # Use hue based on the index (and cycle through the hues if necessary)
            color = hues[idx % len(hues)]  # Cycle through the hue list
            # Apply color to the border and fill the background, add curved corners
            colored_output += f'<span style="color:black; padding:5px; border: 2px solid {color}; background-color:{color}; border-radius: 10px; margin: 2px; display:inline-block;">{word}</span> '

        # Display the colorized tokenized words with bounding boxes
        st.markdown(f"<div>{colored_output}</div>", unsafe_allow_html=True)

    if st.session_state.show_decode:
        with col2:
            st.subheader("Decoded Text")
            tokenized_words = st.session_state.tokenized_words

            # Decode the tokenized words back into text
            decoded_text = st.session_state.tokenizer.decode(tokenized_words)

            # Prepare for colored output
            colored_output = ""
            
            # Loop through the tokenized words to apply the same colors (based on index)
            for idx, token in enumerate(tokenized_words):
                # Use hue based on the index (and cycle through the hues if necessary)
                color = hues[idx % len(hues)]  # Same hue assignment as encoding
                
                # Decode each token to get its original text (handle merged tokens as well)
                decoded_word = st.session_state.tokenizer.vocab[token].decode('utf-8', errors="replace")
                
                # Apply color to the border and fill the background, add curved corners
                colored_output += f'<span style="color:black; padding:5px; border: 2px solid {color}; background-color:{color}; border-radius: 10px; margin: 2px; display:inline-block;">{decoded_word}</span> '

            # Display the colorized decoded text with bounding boxes
            st.markdown(f"<div>{colored_output}</div>", unsafe_allow_html=True)
