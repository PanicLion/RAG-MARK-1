# How to setup
1. run 
    ```
    git clone https://github.com/PanicLion/RAG-MARK-1.git
    cd RAG-MARK-1
    code .
    ```
2. Download the llmfile from [here](https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-q4.llamafile?download=true).
3. Add .exe extension to this file.
4. execute `llava-v1.5-7b-q4.llamafile.exe -ngl 9999`
5. In VS code terminal create a virtual environment and activate it. 
6. run `pip install -r requirements.txt`
7. to run the streamlit app `streamlit run app.py`
