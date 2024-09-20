# Tokenizer Visualization App

A **Streamlit-based** application that visualizes how different tokenization techniques—**Byte Pair Encoding (BPE), WordPiece, and Unigram**—process and tokenize text. This tool is designed for exploring tokenization strategies used by **large language models (LLMs)**, such as from **Google Gemnini, OpenAI GPT-4**, and others. It provides an interactive interface for training, testing, and visualizing tokenizer outputs in real-time.


## Features

- **Three Tokenization Techniques**: Explore and visualize tokenization for **BPE**, **WordPiece**, and **Unigram** tokenizers.
- **Interactive Training**: Train tokenizers on custom text and understand how tokens are generated.
- **Real-Time Visualization**: View tokenized outputs with color-coded tokens and interactive design.
- **Text Decoding**: Reverse the tokenization process and see how tokens are transformed back into text.
- **Unicode and Special Character Handling**: Handles complex characters like Unicode with ease, giving accurate tokenization insights.
- **Streamlit UI**: A simple, user-friendly interface built on Streamlit for seamless interaction and learning.

## Demo

<!-- If available, replace with a demo GIF or image -->
<!-- ![Demo GIF](demo.gif) -->

## How to Use

1. **Choose a Tokenizer**: Select one of the tokenization techniques from the sidebar—BPE, WordPiece, or Unigram.
2. **Input Training Data**: Enter a piece of text in the "Enter text to tokenize" text area.
3. **Train the Tokenizer**: Click the "Train Tokenizer" button to train the selected tokenizer on the provided text.
4. **Tokenize**: Enter a test sentence and click "Tokenize" to visualize how the trained tokenizer breaks the sentence into tokens.
5. **Visualize**: View the color-coded tokenized output with styled bounding boxes and curved borders.
6. **Decode**: If supported by the tokenizer, view how tokens are decoded back into text in the second column.

> **Note**: Unigram tokenizer’s decoding functionality is coming soon.

