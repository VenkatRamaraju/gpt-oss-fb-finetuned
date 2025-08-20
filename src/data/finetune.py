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

c = Client(host="http://localhost:11434")

class ValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether the validation passed")
    feedback: str = Field(default="", description="Feedback message if validation failed, empty if valid")

class AgentState(TypedDict):
    dataset: list 
    current_index: int 
    current_element: dict 
    processed_string: str 
    validation_result: dict 
    final_results: list 
    is_complete: bool
    iteration_count: int
    max_iterations: int

def remove_non_ascii(input_string: str) -> str:
    # strip non-ASCII chars
    return input_string.encode('ascii', 'ignore').decode('ascii')

def invoke_client(query, **generate_kwargs):
    # call ollama API
    return c.generate(model="llama3.2", prompt=query, **generate_kwargs)["response"]

def pydantic_to_dict(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        # newer pydantic
        return model.model_dump()
    # older pydantic
    return model.dict()

def invoke_structured_client(prompt: str) -> ValidationResult:
    # force JSON output
    response = invoke_client(prompt, format="json")
    output_parser = PydanticOutputParser(pydantic_object=ValidationResult)
    # parse structured response
    return output_parser.parse(response)

def prompt(query):
    prompt = """
    You are an expert text editor. Rewrite the input text into fluent, well-structured, grammatically correct English while preserving its original meaning and tone.

    Rules:
    1. Fix grammar, spelling, and punctuation.
    2. Expand slang and shorthand into standard English unless part of the intended tone.
    3. Preserve the tone — casual stays conversational, but never sloppy.
    4. Remove extraneous characters, broken encodings, filler words, repeated letters, and emojis.
    5. Capitalize appropriately and ensure proper sentence boundaries.
    6. Condense where possible without removing meaning.
    7. Do not alter factual content.
    8. Recognize proper nouns and do not change them.

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

def create_finetune_data():
    # get bucket name
    bucket = os.getenv("USER_MESSAGES_BUCKET")
    if not bucket:
        return []

    file_list = get_json_file_list(bucket) or []
    messages = []
    for file_name in file_list:
        try:
            response = read_json_file(bucket, file_name) or {}
            # extract message data
            user_messages = response.get("data", [])
            if not isinstance(user_messages, list):
                continue
            # filter strings only
            messages.extend([m for m in user_messages if isinstance(m, str)])
        except Exception as e:
            pass

    dataset = []
    for message in messages[:100]:
        # limit to 50 messages
        message = remove_non_ascii(message)
        # create training pairs
        dataset.append({"original": message, "cleaned": invoke_client(prompt(message))})

    return dataset

def processor_node(state: AgentState) -> AgentState:
    """Node that handles dataset initialization and element processing."""
    # Initialize dataset if needed
    if not state.get("dataset"):
        dataset = create_finetune_data()
        if not dataset:
            return {**state, "is_complete": True}
        # Initialize with first element
        first_element = dataset[0]
        processed_string = invoke_client(prompt(first_element["original"]))
        return {
            **state,
            "dataset": dataset,
            "current_index": 0,
            "current_element": first_element,
            "processed_string": processed_string,
            "final_results": [],
            "is_complete": False,
            "iteration_count": 1,  # First attempt
            "max_iterations": 3
        }
    
    # Check if we're done with all elements
    if state["current_index"] >= len(state["dataset"]):
        return {**state, "is_complete": True}
    
    current_element = state["dataset"][state["current_index"]]
    current_iterations = state.get("iteration_count", 0)
    
    # Process element (with retry logic if needed)
    if current_iterations > 0 and state.get("validation_result", {}).get("feedback"):
        # Retry with feedback
        feedback = state["validation_result"]["feedback"]
        previous_attempt = state.get("processed_string", "")
        enhanced_prompt = f"""
        {prompt(current_element["original"])}
        
        Previous attempt: {previous_attempt}
        Previous attempt feedback: {feedback}
        
        Please improve the text based on this feedback while maintaining the original meaning.

        Answer:
        """
        processed_string = invoke_client(enhanced_prompt)
    else:
        # First attempt
        processed_string = invoke_client(prompt(current_element["original"]))
    
    return {
        **state,
        "current_element": current_element,
        "processed_string": processed_string,
        "iteration_count": current_iterations + 1,
    }

def validator_node(state: AgentState) -> AgentState:
    """Node that validates processed text and manages workflow decisions."""
    if state.get("is_complete", False) or state["current_index"] >= len(state.get("dataset", [])):
        return {**state, "is_complete": True}

    # Safety check - current_element should be set by processor_node
    if not state.get("current_element") or "original" not in state["current_element"]:
        print(f"Warning: current_element not properly set: {state.get('current_element')}")
        return {**state, "is_complete": True}

    original = state["current_element"]["original"]
    processed_string = state["processed_string"]
    
    # Validate the processed string
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

    Input: {original}
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
    
    # Decide what to do next
    if validation_result.is_valid or current_iterations >= max_iterations:
        # Accept result and move to next element
        new_result = {
            "original": state["current_element"]["original"],
            "processed": processed_string,
        }
        
        return {
            **state,
            "final_results": state["final_results"] + [new_result],
            "current_index": state["current_index"] + 1,
            "iteration_count": 0,  # Reset for next element
            "validation_result": {},
            "processed_string": "",
        }
    else:
        # Keep feedback for retry
        return {
            **state,
            "validation_result": pydantic_to_dict(validation_result),
        }

def should_continue_two_nodes(state: AgentState) -> str:
    """Routing function for two-node workflow."""
    if state.get("is_complete", False):
        return "END"
    
    # If we have validation result that failed and haven't hit max retries
    validation_result = state.get("validation_result", {})
    if (validation_result.get("is_valid") == False and 
        state.get("iteration_count", 0) < state.get("max_iterations", 3)):
        return "retry"
    
    # Otherwise continue to next element
    return "next"

def orchestrate():
    # create workflow graph with 2 nodes
    workflow = StateGraph(AgentState)
    
    workflow.add_node("processor", processor_node)
    workflow.add_node("validator", validator_node)
    
    workflow.set_entry_point("processor")
    
    # processor always goes to validator
    workflow.add_edge("processor", "validator")
    
    # validator decides what to do next
    workflow.add_conditional_edges(
        "validator",
        should_continue_two_nodes,
        {
            "retry": "processor",  # retry current element
            "next": "processor",   # move to next element
            "END": END
        }
    )
    
    # compile workflow
    app = workflow.compile()
    
    initial_state = {
        "dataset": [],
        "current_index": 0,
        "final_results": [],
        "is_complete": False,
        "max_iterations": 3
    }
    
    try:
        # run workflow with properly increased recursion limit
        # 2 nodes per iteration: 100 elements × 3 retries × 2 nodes + overhead
        result = app.invoke(initial_state, config={"recursion_limit": 800})

        pprint.pprint(result["final_results"])

        # Upload to s3
        # upload_to_s3(os.getenv("FINE_TUNE_DATA_BUCKET"), "finetune_data.json", {"data": result["final_results"]})

        # Return results
        return result
    except Exception as e:
        import traceback
        # debug errors
        traceback.print_exc()
        return None

if __name__ == "__main__":
    orchestrate()