# imports
from util import get_json_file_list, upload_to_s3, read_json_file, has_non_ascii
import os
from ollama import Client
from typing import TypedDict
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import json
import pprint

# setup client
c = Client(host="http://localhost:11434")

# define validation
class ValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether the validation passed")
    feedback: str = Field(default="", description="Feedback message if validation failed, empty if valid")

# define state type
class AgentState(TypedDict):
    original_text: str
    processed_string: str 
    validation_result: dict 
    is_complete: bool
    iteration_count: int
    max_iterations: int

# clean text
def remove_non_ascii(input_string: str) -> str:
    return input_string.encode('ascii', 'ignore').decode('ascii')

# call llm
def invoke_client(query, **generate_kwargs):
    return c.generate(model="llama3.2", prompt=query, **generate_kwargs)["response"]

# convert to dict
def pydantic_to_dict(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

# get structured response
def invoke_structured_client(prompt: str) -> ValidationResult:
    response = invoke_client(prompt, format="json")
    output_parser = PydanticOutputParser(pydantic_object=ValidationResult)
    try:
        return output_parser.parse(response)
    except Exception as e:
        return ValidationResult(is_valid=False, feedback=str(e))

# get prompt text
def prompt(query):
    prompt = """
    You are an expert text editor. Rewrite the input text into fluent, well-structured, grammatically correct English while preserving its original meaning and tone.

    Rules:
    1. Fix grammar, spelling, and punctuation.
    2. Preserve the tone — casual stays conversational, but never sloppy.
    3. Remove extraneous characters, broken encodings, filler words, repeated letters, and emojis.
    4. Capitalize appropriately and ensure proper sentence boundaries.
    5. Condense where possible without removing meaning.
    6. Do not alter factual content.
    7. Recognize proper nouns and do not change them.
    8. Do not paraphrase anything. Do a word to word translation.

    Output format:
    - Return only the rewritten sentence.
    - Do not explain, comment, or include any labels like "Input" or "Output."
    - Do not say anything except the rewritten text. Do not explain anything. Just the rewritten text.

    Examples:
    Input: not a bad thing though, im. enjoying it
    Output: It's not a bad thing, though. I'm enjoying it.

    Input: maybe its just me then hahah i feel like we all go out less often
    Output: Maybe it's just me, but I feel like we all go out less often.

    Input: not that much socializing
    Output: There hasn't been much socializing.

    Input: its good bro super lowkey compared to last year
    Output: It's good, super lowkey compared to last year.

    Important: Your entire reply must be the rewritten sentence only, with no extra text before or after.

    Now rewrite:
    {query}
    """
    return prompt.format(query=query)

# load data
def create_finetune_data():
    bucket = os.getenv("USER_MESSAGES_BUCKET")
    if not bucket:
        return []

    file_list = get_json_file_list(bucket) or []
    messages = []
    for file_name in file_list:
        try:
            response = read_json_file(bucket, file_name) or {}
            user_messages = response.get("data", [])
            if not isinstance(user_messages, list):
                continue
            messages.extend([m for m in user_messages if isinstance(m, str)])
        except Exception as e:
            pass

    dataset = []
    for message in messages:
        message = remove_non_ascii(message)
        dataset.append({"original": message})

    return dataset

# process text
def processor_node(state: AgentState) -> AgentState:
    original_text = state["original_text"]
    current_iterations = state.get("iteration_count", 0)
    
    if current_iterations > 0 and state.get("validation_result", {}).get("feedback"):
        feedback = state["validation_result"]["feedback"]
        previous_attempt = state.get("processed_string", "")
        enhanced_prompt = f"""
        {prompt(original_text)}
        
        Previous attempt: {previous_attempt}
        Previous attempt feedback: {feedback}
        
        Please improve the text based on this feedback while maintaining the original meaning.

        Answer:
        """
        processed_string = invoke_client(enhanced_prompt)
    else:
        processed_string = invoke_client(prompt(original_text))
    
    return {
        **state,
        "processed_string": processed_string,
        "iteration_count": current_iterations + 1,
    }

# validate text
def validator_node(state: AgentState) -> AgentState:
    original_text = state["original_text"]
    processed_string = state["processed_string"]
    
    validation_prompt = f"""
    You are a validation system that checks if a rewritten sentence correctly follows the transformation rules.

    Transformation rules:
    1. The rewritten sentence must preserve the original meaning of the input.
    2. Grammar, spelling, and punctuation must be correct.
    3. Slang and shorthand must be expanded into standard English unless intentionally kept for tone.
    4. Extraneous characters, broken encodings, and meaningless fillers must be removed.
    5. The sentence must be properly capitalized and well-formatted.
    6. The tone should remain natural and conversational.

    Task: Validate if the rewritten text follows all transformation rules.

    Input: {original_text}
    Output: {processed_string}

    Analyze the text and provide your validation decision.

    IMPORTANT: You must respond with a JSON string in exactly this format:
    - If valid: {{"is_valid": true, "feedback": ""}}
    - If not valid: {{"is_valid": false, "feedback": "<your feedback here>"}}
    
    Return only syntactically correct JSON, no prose, no code fences.
    """
    
    validation_result = invoke_structured_client(validation_prompt)
    current_iterations = state.get("iteration_count", 1)
    max_iterations = state.get("max_iterations", 3)
    
    if validation_result.is_valid or current_iterations >= max_iterations:
        return {
            **state,
            "is_complete": True,
            "validation_result": pydantic_to_dict(validation_result),
        }
    else:
        return {
            **state,
            "is_complete": False,
            "validation_result": pydantic_to_dict(validation_result),
        }

# check retry
def should_continue_single_item(state: AgentState) -> str:
    if state.get("is_complete", False):
        return "END"
    
    validation_result = state.get("validation_result", {})
    if not validation_result.get("is_valid") and state.get("iteration_count", 0) < state.get("max_iterations", 3):
        return "retry"
    
    return "END"

# process one text
def process_single_item(original_text: str) -> dict:
    workflow = StateGraph(AgentState)
    
    workflow.add_node("processor", processor_node)
    workflow.add_node("validator", validator_node)
    workflow.set_entry_point("processor")
    workflow.add_edge("processor", "validator")
    
    workflow.add_conditional_edges(
        "validator",
        should_continue_single_item,
        {
            "retry": "processor",
            "END": END
        }
    )
    
    app = workflow.compile()
    
    initial_state = {
        "original_text": original_text,
        "processed_string": "",
        "validation_result": {},
        "is_complete": False,
        "iteration_count": 0,
        "max_iterations": 3
    }
    
    try:
        result = app.invoke(initial_state, config={"recursion_limit": 10})
        return {
            "original": original_text,
            "processed": result["processed_string"],
            "validation_result": result.get("validation_result", {}),
            "iterations_used": result.get("iteration_count", 0)
        }
    except Exception as e:
        return {
            "original": original_text,
            "processed": original_text,
            "validation_result": {"is_valid": False, "feedback": str(e)},
            "iterations_used": 0
        }

# run workflow
def orchestrate():
    dataset = create_finetune_data()
    if not dataset:
        return None
    
    BATCH_SIZE = 1000
    total_items = len(dataset)
    processed_count = 0
    batch_number = 0
    while processed_count < total_items:
        batch_start = processed_count
        batch_end = min(processed_count + BATCH_SIZE, total_items)
        current_batch = dataset[batch_start:batch_end]
        
        batch_results = []
        for item in current_batch:
            original_text = item["original"]
            result = process_single_item(original_text)
            batch_results.append(result)
        try:
            batch_filename = f"finetune_data_batch_{batch_number}.json"
            upload_to_s3(os.getenv("FINE_TUNE_DATA_BUCKET"), batch_filename, {"data": batch_results})
            print(f"Batch {batch_number + 1}: {batch_start} to {batch_end} of {total_items}")
        except Exception as e:
            return {"final_results": batch_results, "error": str(e)}
        
        processed_count += len(current_batch)
        batch_number += 1
    
    return {"total_processed": processed_count, "total_batches": batch_number}

if __name__ == "__main__":
    orchestrate()