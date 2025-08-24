# fb-finetuned

This project finetunes **gpt-oss-20b** on 1.5 years of my Facebook texts to replicate my personal texting style.

- Built a dataset generation agent using **LangGraph** and **Ollama (llama3.2b)** to prepare training data.  
- Finetuned **gpt-oss-20b** with **PEFT + LoRA** for efficient training.  
- Planning to integrate a **speech-to-text (STT) pipeline** to enable spoken input/output.