from util import get_json_file_list, upload_to_s3, read_json_file, has_non_ascii
import os
from ollama import Client
import pprint
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import json
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate


# local Ollama client
c = Client(host="http://localhost:11434")


# Pydantic model for validation results
class ValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether the validation passed")
    feedback: str = Field(default="", description="Feedback message if validation failed, empty if valid")

# State definition for the graph
class AgentState(TypedDict):
    dataset: list 
    current_index: int 
    current_element: dict 
    processed_string: str 
    validation_result: dict 
    final_results: list 
    is_complete: bool
    iteration_count: int  # Track how many attempts for current element
    max_iterations: int   # Maximum attempts before giving up 


def invoke_client(query):
    return c.generate(model="llama3.2", prompt=query)["response"]

def invoke_structured_client(prompt_template: str, output_parser: PydanticOutputParser, **kwargs) -> BaseModel:
    try:
        # Format the prompt with the output format instructions
        prompt = PromptTemplate(
            template=prompt_template + "\n\n{format_instructions}",
            input_variables=list(kwargs.keys()) + ["format_instructions"]
        )
        
        # Get format instructions from the parser
        format_instructions = output_parser.get_format_instructions()
        
        # Format the complete prompt
        formatted_prompt = prompt.format(
            format_instructions=format_instructions,
            **kwargs
        )
        
        # Get raw response from LLM
        raw_response = invoke_client(formatted_prompt)
        
        # Parse the response into Pydantic object
        parsed_result = output_parser.parse(raw_response)
        return parsed_result
        
    except Exception as e:
        # Fallback: create a default validation result
        print(f"Warning: Failed to parse structured output: {e}")
        print(f"Raw response: {raw_response}")
        
        # Try to extract basic validation info from raw response
        if "VALID" in raw_response.upper():
            return ValidationResult(is_valid=True, feedback="")
        else:
            return ValidationResult(
                is_valid=False, 
                feedback=f"Parsing failed: {str(e)}"
            )

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

    Output format:
    - Return only the rewritten sentence.
    - Do not explain, comment, or include any labels like "Input" or "Output."
    - Do not say anything except the rewritten text.

    Examples:
    Input: not a bad thing though, im. enjoying it
    Output: It's not a bad thing, though. I'm enjoying it.

    Input: maybe its just me then hahah i feel like we all go out less often
    Output: Maybe it's just me, but I feel like we all go out less often.

    Input: not that much socializing
    Output: There hasn't been much socializing.

    Input: its good bro super lowkey compared to last year
    Output: It's good—super low-key compared to last year.

    Important: Your entire reply must be the rewritten sentence only, with no extra text before or after.

    Now rewrite:
    {query}
    """
    return prompt.format(query=query)

def create_finetune_data():
    # Get user messages
    file_list = get_json_file_list(os.getenv("USER_MESSAGES_BUCKET"))
    messages = []
    for file_name in file_list:
        response = read_json_file(os.getenv("USER_MESSAGES_BUCKET"), file_name)
        user_messages = response["data"]
        messages.extend(user_messages)

    # Form dataset
    dataset = []
    for message in messages[:5]:
        if has_non_ascii(message):
            continue
        dataset.append({"original": message, "cleaned": invoke_client(prompt(message))})
    
    return dataset

# Node 1: Get dataset
def get_dataset_node(state: AgentState) -> AgentState:
    """First node: Gets the dataset using create_finetune_data()"""
    return {
        **state,
        "dataset": create_finetune_data(),
        "current_index": 0,
        "final_results": [],
        "is_complete": False,
        "iteration_count": 0,
        "max_iterations": 3  # Allow up to 3 attempts per element
    }

# Node 2: Process element
def process_element_node(state: AgentState) -> AgentState:
    """Second node: Processes current element by calling prompt() on it"""
    if state["current_index"] >= len(state["dataset"]):
        return {**state, "is_complete": True}
    
    current_element = state["dataset"][state["current_index"]]
    
    # If this is a retry, include feedback in the prompt
    if state.get("iteration_count", 0) > 0 and state.get("validation_result", {}).get("feedback"):
        feedback = state["validation_result"]["feedback"]
        previous_attempt = state.get("processed_string", "")
        enhanced_prompt = f"""
        {prompt(current_element["original"])}
        
        Previous attempt: {previous_attempt}
        Previous attempt feedback: {feedback}
        
        Please improve the text based on this feedback while maintaining the original meaning.
        """
        processed_string = invoke_client(enhanced_prompt)
    else:
        # First attempt - use original prompt
        processed_string = invoke_client(prompt(current_element["original"]))
    
    return {
        **state,
        "current_element": current_element,
        "processed_string": processed_string,
        "iteration_count": state.get("iteration_count", 0) + 1
    }

# Node 3: Validate response
def validate_node(state: AgentState) -> AgentState:
    """Third node: Validates the processed string using a validation prompt"""
    
    # Create the output parser for our ValidationResult model
    output_parser = PydanticOutputParser(pydantic_object=ValidationResult)
    
    validation_prompt_template = """
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
    Output: {rewritten_text}

    Analyze the text and provide your validation decision.
    """
    
    # Use structured output parsing to automatically marshal into Pydantic object
    validation_result = invoke_structured_client(
        prompt_template=validation_prompt_template,
        output_parser=output_parser,
        original_text=state["current_element"]["original"],
        rewritten_text=state["processed_string"]
    )
    
    # Update state with validation result
    if validation_result.is_valid:
        # Validation passed - add to results and move to next element
        new_result = {
            "original": state["current_element"]["original"],
            "processed": state["processed_string"],
        }
        
        updated_results = state["final_results"] + [new_result]
        next_index = state["current_index"] + 1
        
        return {
            **state,
            "final_results": updated_results,
            "current_index": next_index,
            "validation_result": validation_result.dict(),
            "iteration_count": 0  # Reset for next element
        }
    else:
        # Validation failed - check if we should retry
        current_iterations = state.get("iteration_count", 1)
        max_iterations = state.get("max_iterations", 3)
        
        if current_iterations < max_iterations:
            # Retry with feedback - go back to process_element
            return {
                **state,
                "validation_result": validation_result.dict()
                # Keep current_index and iteration_count for retry
            }
        else:
            # Max iterations reached - skip this element and move to next
            next_index = state["current_index"] + 1
            return {
                **state,
                "current_index": next_index,
                "validation_result": validation_result.dict(),
                "iteration_count": 0  # Reset for next element
            }

# Decision function to determine next step
def should_continue(state: AgentState) -> str:
    """Determines whether to continue processing or end"""
    if state.get("is_complete", False) or state["current_index"] >= len(state["dataset"]):
        return "END"
    
    # Check if we need to retry the current element
    validation_result = state.get("validation_result", {})
    if validation_result.get("is_valid") == False:
        current_iterations = state.get("iteration_count", 1)
        max_iterations = state.get("max_iterations", 3)
        
        if current_iterations < max_iterations:
            return "retry"  # Go back to process_element for retry
        else:
            return "continue"  # Max iterations reached, move to next element
    
    return "continue"  # Normal flow to next element

def orchestrate():
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("get_dataset", get_dataset_node)
    workflow.add_node("process_element", process_element_node)
    workflow.add_node("validate", validate_node)
    
    # Set entry point
    workflow.set_entry_point("get_dataset")
    
    # Add edges
    workflow.add_edge("get_dataset", "process_element")
    
    # Add conditional edges from validate node
    workflow.add_conditional_edges(
        "validate",
        should_continue,
        {
            "continue": "process_element",  # Move to next element
            "retry": "process_element",     # Retry current element with feedback
            "END": END
        }
    )
    
    # Add edge from process to validate
    workflow.add_edge("process_element", "validate")
    
    # Compile the graph
    app = workflow.compile()
    
    # Initialize state
    initial_state = {
        "dataset": [],
        "current_index": 0,
        "current_element": {},
        "processed_string": "",
        "validation_result": {},
        "final_results": [],
        "is_complete": False,
        "iteration_count": 0,
        "max_iterations": 3
    }
    
    # Run the workflow
    try:
        print("🚀 Starting LangGraph workflow...")
        result = app.invoke(initial_state)
        
        print(f"🎉 Workflow completed!")
        print(f"📊 Processed {len(result['final_results'])} elements")
        print(f"📋 Final results:")
        for i, item in enumerate(result['final_results']):
            print(f"  {i+1}. Original: {item['original'][:50]}...")
            print(f"     Processed: {item['processed'][:50]}...")
            print()
        
        return result
    except Exception as e:
        print(f"❌ Workflow failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    orchestrate()