from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import boto3
from botocore.config import Config
from langchain_core.runnables import RunnablePassthrough
import os
import re
import traceback

import datetime
import json
from langchain.memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory


from rate_limit_handler import retry_with_backoff, invoke_chain_with_retry


from extraction import extract_kernel_from_llm_response, extract_reasoning, read_file, write_file, log_to_file, run, update_function_name_in_text
from doc_grabber import get_available_functions, select_relevant_functions, load_function_documentation
from nki_error_parsing import NKIErrorParser, extract_error_details, get_available_error_codes, select_relevant_errors, load_error_documentation

def log_iteration_data(
    iteration_log_path,
    iteration_number,
    error_message,
    error_line,
    error_description,
    reasoning_text,
    kernel_code,
    test_result,
    change_result=None,
    append=True
):
    """
    Log all data from a kernel generation iteration to a single consolidated file.
    """
    import json
    from datetime import datetime
    
    # Create a structured dictionary for this iteration
    iteration_data = {
        "timestamp": datetime.now().isoformat(),
        "iteration": iteration_number,
        "error": {
            "message": error_message,
            "line": error_line,
            "description": error_description
        },
        "solution": {
            "reasoning": reasoning_text,
            "kernel_code": kernel_code
        },
        "test_result": test_result
    }
    
    # Add change analysis if available
    if change_result:
        iteration_data["change_analysis"] = change_result
    
    # Format the data for human-readable output
    formatted_output = f"\n{'='*80}\n"
    formatted_output += f"ITERATION {iteration_number} - {datetime.now().isoformat()}\n"
    formatted_output += f"{'='*80}\n\n"
    
    # ERROR SECTION
    formatted_output += f"--- ERROR INFORMATION ---\n\n"
    if error_line:
        formatted_output += f"ERROR LINE: {error_line}\n"
    if error_description:
        formatted_output += f"ERROR DESCRIPTION: {error_description}\n"
    formatted_output += f"\nFULL ERROR MESSAGE:\n{error_message}\n\n"
    
    # SOLUTION SECTION
    formatted_output += f"--- SOLUTION INFORMATION ---\n\n"
    if reasoning_text:
        formatted_output += f"REASONING:\n{reasoning_text}\n\n"
    
    # Include truncated kernel code (first 50 lines with indicator if truncated)
    kernel_lines = kernel_code.splitlines()
    max_lines = 50
    if len(kernel_lines) > max_lines:
        kernel_preview = "\n".join(kernel_lines[:max_lines])
        kernel_preview += f"\n\n... [truncated, {len(kernel_lines) - max_lines} more lines] ...\n"
    else:
        kernel_preview = kernel_code
    
    formatted_output += f"GENERATED KERNEL CODE:\n{kernel_preview}\n\n"
    
    # TEST RESULT SECTION
    formatted_output += f"--- TEST RESULT ---\n\n"
    formatted_output += f"{test_result}\n\n"
    
    # CHANGE ANALYSIS SECTION (if available)
    if change_result:
        formatted_output += f"--- CHANGE ANALYSIS ---\n\n"
        formatted_output += f"FIXED PREVIOUS ERROR: {change_result.get('correct', False)}\n"
        formatted_output += f"ANALYSIS: {change_result.get('report', 'No analysis provided')}\n\n"
    
    # Also include the raw JSON data for easier database ingestion later
    json_data = json.dumps(iteration_data, indent=2)
    formatted_output += f"--- RAW JSON DATA ---\n\n"
    formatted_output += f"{json_data}\n\n"
    
    # Write to file
    mode = "a" if append else "w"
    with open(iteration_log_path, mode, encoding="utf-8") as log_file:
        log_file.write(formatted_output)
    
    # Return the data dictionary for potential further processing
    return iteration_data



def generate_kernel_with_direct_docs_and_error_loop(
    kernel_func_name,
    system_prompt_path, 
    user_prompt_path, 
    output_address,
    kernel_module_path,
    test_func_name,
    test_script_output,
    reasoning_log_path,
    error_doc_path,
    docs_dir,
    max_iterations=15
):
    """
    Generate a NKI kernel using direct function documentation access and iteratively 
    improve it based on error feedback with detailed error documentation.
    """
    
    error_parser = NKIErrorParser(error_doc_path)
    

    # Set up consolidated iteration log file
    consolidated_log_path = output_address + ".consolidated_iterations.txt"
    # Initialize with header only on first write (will be overwritten)
    with open(consolidated_log_path, "w", encoding="utf-8") as f:
        f.write(f"=== CONSOLIDATED ITERATION LOG ===\n")
        f.write(f"Started at: {datetime.datetime.now()}\n")
        f.write(f"Output path: {output_address}\n")
        f.write(f"Kernel module path: {kernel_module_path}\n\n")

    # Load the initial prompts
    system_prompt = read_file(system_prompt_path)
    user_prompt = read_file(user_prompt_path)
    
    
    # Initialize LLMs
    query_llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.3
    )
    # kernel_llm = ChatOpenAI(
    #     model="gpt-4o-mini", 
    #     temperature=0.85
    # )
    # Configure boto3 client with custom retry settings
    boto_config = Config(
        region_name="us-west-2",
        retries=dict(
            max_attempts=60,
            mode="adaptive",
            total_max_attempts=60
        )
    )
    
    # Create bedrock client with custom config
    bedrock_client = boto3.client(
        "bedrock-runtime",
        config=boto_config
    )
    
    kernel_llm = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        model_kwargs={"temperature": 0.85},
        client=bedrock_client,
        region_name="us-west-2"
    )
    


    # Get list of available functions
    available_functions = get_available_functions(docs_dir)
   
    # Initial kernel generation with direct documentation
    try:
        # Select relevant functions
        
        selected_functions = select_relevant_functions(
            query_llm,
            user_prompt,
            available_functions
        )
        
    
        function_docs = load_function_documentation(docs_dir, selected_functions)
    
        # Initial kernel generation with function documentation
        initial_generation_prompt = ChatPromptTemplate.from_template(
            "{system_prompt}\n\n"
            "Task: {user_prompt}\n\n"
            "Function Documentation:\n{function_docs}\n\n"
            "Generate a NKI kernel for the task."
        )
        
        # Log the full prompt being sent to the LLM
        full_prompt = initial_generation_prompt.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            function_docs=function_docs
        )
        prompt_path = output_address + ".prompt_path.txt"
        log_to_file(prompt_path, f"FULL PROMPT TO LLM:\n{full_prompt}\n", append = True)
        
        initial_kernel_chain = (
            initial_generation_prompt 
            | kernel_llm 
            | StrOutputParser()
        )
        
        try:
            initial_generation = invoke_chain_with_retry(initial_kernel_chain, {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "function_docs": function_docs
            },
        )
        except Exception as e:
            print(f"Error in initial kernel generation: {e}")
            initial_generation = f"Error occurred: {str(e)}"
        
        # Save raw output
        write_file(output_address, initial_generation)
        
        # Extract the kernel code
        try:
            kernel_code = extract_kernel_from_llm_response(initial_generation)
            kernel_code = update_function_name_in_text(kernel_code, kernel_func_name)
            write_file(kernel_module_path, kernel_code)
        except ValueError as e:
            error_msg = f"Error extracting kernel code: {e}"
            print(error_msg)
            return
        
        # Create previous error context to track history
        previous_error_message = ""
        previous_iteration_info = []
        
        # Create enhanced error re-injection prompt with error documentation and history
        enhanced_error_reinject_prompt = ChatPromptTemplate.from_template(
            "{system_prompt}\n\n"
            "Generate a new improved kernel for this task. Clearly explain your line of reasoning in one sentence, trying"
            "to keep it as brief as possible. Focus on explaining the exact change you will be making to the code."
            "I dont want the actual code, but be specific so someone that sees the same error message on a different line of code"
            "can implement the same fix. Remember to keep it concise, but explanatory as you will be referencing this later to make sure"
            "you are not trying to do the same fixes multiple times. "
            "When you are changing the code, try to only change the line with the error message and maybe code that relates."
            "However, if the error you are facing is that the outputs differ, then you are allowed to change multiple lines."
            "When the outputs differ, most likely the logic is wrong. I want you to notice this and in your reasoning state that the logic is "
            "likely wrong and state which logic you will update. Please clearly state in your reasoning ***i see that the outputs differ***"
            "Your output should include the entire kernel code, NOT just individual fixes. I want to be able to run the code inside the ``` ```"
            "The way I want your response structured is an explanation of your reasoning at the very start inside *** *** triple stars. "
            "Then, immediatly after write the python nki code inside triple backticks ``` ```."
            "I repeat, I only want your output to first be the line of reasoning inside triple stars, then the "
            "nki kernel code inside triple backticks. Do NOT put the reasoning inside the nki kernel code."
            "Everything above this line is the most important information. Please make sure you follow these guidelines."
            "Task: {user_prompt}\n\n"
            
            "{iteration_history}\n\n"
            "Previous error message:\n"
            "--------------------------------------------------\n"
            "{previous_error_message}\n"
            "--------------------------------------------------\n\n"
            "Function Documentation:\n"
            "--------------------------------------------------\n"
            "{function_docs}\n"
            "--------------------------------------------------\n\n"
            
        )
        
        enhanced_error_chain = (
            enhanced_error_reinject_prompt 
            | kernel_llm 
            | StrOutputParser()
        )
        
        # Iterative error correction loop
        for iteration in range(max_iterations):
            print(f"\n=== Iteration {iteration + 1} ===")
            
            # Store the previous error message before running any new tests
            old_error_message = previous_error_message if 'previous_error_message' in locals() else ""
            
            # Run the test script only if this is iteration 0 (initial code) or after we've generated new code
            # For the first iteration, we need to run the script on the initial code
            if iteration == 0:
                # Run the test using the execution server for the initial kernel
                from extraction import run
                error_message = run(test_func_name, kernel_func_name, kernel_module_path, test_script_output)

                previous_error_message = error_message
                
                # If no errors in the initial code, we're done
                if "Error" not in error_message and "error" not in error_message and "ERROR" not in error_message:
                    print("No errors detected in initial kernel! Kernel generation successful.")
                    # Log successful initial generation to the consolidated log
                    log_iteration_data(
                        consolidated_log_path,
                        iteration + 1,
                        "No errors detected",
                        None,
                        None,
                        "Initial generation successful without errors",
                        kernel_code,
                        error_message,
                        None
                    )
                    return 1

            error_line, error_description = extract_error_details(error_message)
            if not error_line and error_description:
                print("\nCould not extract specific error details.")


            # Get all available error codes
            available_errors = get_available_error_codes(error_parser)

            # Select relevant errors using the LLM
            error_selection_prompt = ChatPromptTemplate.from_template(
                "You are helping to identify relevant NKI error codes from error output.\n\n"
                "Here is the error output:\n{error_message}\n\n"
                "Available error codes:\n{error_list}\n\n"
                "Please identify the most relevant error codes in this output. Return your selection as a JSON list "
                "of error codes (without the 'ERROR: ' prefix). For example: [\"INVALID_TYPE\", \"OUT_OF_BOUNDS\"]\n\n"
                "Your entire response must be a valid JSON array. Do not include any explanations, headers, or text before or after the JSON."
                "I repeat your entire response must be a valid JSON array. Do not deviate from this format"
            )

            # Format error list for display
            error_list = "\n".join(sorted(available_errors))

            error_selection_chain = (
                error_selection_prompt
                | query_llm
                | StrOutputParser()
            )

            try:
                error_response = invoke_chain_with_retry(error_selection_chain, {
                    "error_message": previous_error_message,
                    "error_list": error_list
                },
            )
            except Exception as e:
                print(f"Error in error selection: {e}")
                error_response = "[]"  # Default to empty list on error


            # Clean up and parse the response
            try:
                # Clean the response and try to parse it
                cleaned_response = extract_json_array(error_response)
                
                # Handle empty lists represented as empty string, "[]", etc.
                if not cleaned_response or cleaned_response.isspace():
                    selected_errors = []
                elif cleaned_response == "[]":
                    selected_errors = []
                else:
                    selected_errors = json.loads(cleaned_response)
                
                # Validate that all selected errors are in available_errors
                selected_errors = [e for e in selected_errors if e in available_errors]
                
            except Exception as e:
                print(f"Error parsing selected errors: {e}")
                
                # Fallback mechanism: try to extract error codes using regex
                try:
                    pattern = re.compile(r'["\']([\w_-]+)["\']')
                    matches = pattern.findall(error_response)
                    selected_errors = [e for e in matches if e in available_errors]
                    print(f"Using fallback: Extracted errors via regex: {', '.join(selected_errors)}")
                except Exception as fallback_error:
                    print(f"Fallback parsing also failed: {fallback_error}")
                    selected_errors = []


            # Load documentation for selected errors
            error_documentation = load_error_documentation(error_parser, selected_errors)
            # Log the selected errors and their documentation
            with open(f"{output_address}.error_selection", "w") as f:
                f.write(f"ERROR MESSAGE:\n{previous_error_message}\n\n")
                f.write(f"SELECTED ERRORS:\n{', '.join(selected_errors)}\n\n")
                f.write(f"ERROR DOCUMENTATION:\n{error_documentation}\n\n")

            # If no documented errors found, use a fallback message
            if not selected_errors:
                error_documentation = "No specific documentation found for the errors in the output. Please analyze the error message carefully."
            
            additional_functions_prompt = ChatPromptTemplate.from_template(
                "Based on the error message below, do we need to include documentation for any additional NKI functions "
                "that weren't selected earlier?\n\n"
                "Current functions: {current_functions}\n\n"
                "Error message:\n{error_message}\n\n"
                "Available functions: {all_functions}\n\n"
                "Return ONLY a JSON list of additional function names needed (without the 'nki_language_' prefix). "
                "If no additional functions are needed, return an empty list [].\n\n"
                "Your entire response must be a valid JSON array. Do not include any explanations, headers, or text before or after the JSON."
            )

            additional_functions_chain = (
                additional_functions_prompt 
                | query_llm 
                | StrOutputParser()
            )

            try:
                additional_response = invoke_chain_with_retry(additional_functions_chain, {
                    "current_functions": ", ".join(selected_functions),
                    "error_message": previous_error_message,
                    "all_functions": ", ".join(available_functions)
                },
            )
            except Exception as e:
                additional_response = "[]"  # Default to empty list on error


            # Clean up the response to ensure it's valid JSON
            def extract_json_array(text):
                # Remove any non-JSON text before or after the array
                text = text.strip()
                # If text begins with characters before [, remove them
                if '[' in text and text[0] != '[':
                    text = text[text.find('['):]
                # If text has characters after the closing ], remove them
                if ']' in text and text[-1] != ']':
                    text = text[:text.rfind(']')+1]
                # If we still don't have a valid JSON looking text, try regex
                if not (text.startswith('[') and text.endswith(']')):
                    import re
                    json_pattern = re.compile(r'\[.*?\]', re.DOTALL)
                    json_match = json_pattern.search(text)
                    if json_match:
                        text = json_match.group(0)
                return text

            try:
                # Clean the response and try to parse it
                cleaned_response = extract_json_array(additional_response)
                
                # Handle empty lists represented as empty string, "[]", etc.
                if not cleaned_response or cleaned_response.isspace():
                    additional_functions = []
                elif cleaned_response == "[]":
                    additional_functions = []
                else:
                    additional_functions = json.loads(cleaned_response)
                
                # Only include valid functions that weren't already selected
                new_functions = [f for f in additional_functions 
                            if f in available_functions and f not in selected_functions]
                
                if new_functions:
                    print(f"Adding additional functions: {', '.join(new_functions)}")
                    
                    # Add to selected functions
                    selected_functions.extend(new_functions)
                    
                    # Update function documentation
                    additional_docs = load_function_documentation(docs_dir, new_functions)
                    function_docs += "\n\n" + additional_docs
                    
            except Exception as e:
                print(f"Error parsing additional functions: {e}")
                
                # Fallback mechanism: try to extract function names using regex
                try:
                    pattern = re.compile(r'["\']([\w_]+)["\']')
                    matches = pattern.findall(additional_response)
                    valid_matches = [f for f in matches if f in available_functions and f not in selected_functions]
                    
                    if valid_matches:
                        print(f"Using fallback: Adding functions detected via regex: {', '.join(valid_matches)}")
                        
                        # Add to selected functions
                        selected_functions.extend(valid_matches)
                        
                        # Update function documentation
                        additional_docs = load_function_documentation(docs_dir, valid_matches)
                        function_docs += "\n\n" + additional_docs
                except Exception as fallback_error:
                    print(f"Fallback parsing also failed: {fallback_error}")
                    
            # Create iteration history for context
            iteration_history = ""
            if previous_iteration_info:
                iteration_history = "Previous iterations:\n"
                for idx, info in enumerate(previous_iteration_info):
                    iteration_history += f"Iteration {idx + 1}:\n{info}\n\n"
            
            # Generate improved kernel with error feedback, documentation, and history
            print(f"Generating improved kernel (iteration {iteration + 1})...")
            

            # Log the full error prompt being sent to the LLM
            full_error_prompt = enhanced_error_reinject_prompt.format(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                iteration_history="",
                previous_error_message=previous_error_message,
                function_docs=function_docs
            )
            log_to_file(prompt_path, f"FULL ERROR PROMPT TO LLM:\n{full_error_prompt}\n", append=False)
            
            
            
            try:
                improved_generation = invoke_chain_with_retry(enhanced_error_chain, {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "iteration_history": iteration_history,
                    "previous_error_message": previous_error_message,
                    "function_docs": function_docs
                },
            )
            except Exception as e:
                improved_generation = f"Error occurred: {str(e)}"
                
            # Save the raw output
            write_file(output_address, improved_generation)
            
            # Extract reasoning and log it
            reasoning_text = extract_reasoning(improved_generation)
            if reasoning_text:
                # Add reasoning to iteration history
                previous_iteration_info.append(f"Reasoning: {reasoning_text}")
        
            # Extract the updated kernel code
            try:
                kernel_code = extract_kernel_from_llm_response(improved_generation)
                kernel_code = update_function_name_in_text(kernel_code, kernel_func_name)
                write_file(kernel_module_path, kernel_code)
                
                # Add the code snippet to the iteration history
                previous_iteration_info.append(f"Generated code: {kernel_code[:500]}...")
            except ValueError as e:
                error_msg = f"Error extracting kernel code: {e}"
                print(error_msg)
                continue
            
            # Now run the test using the execution server
            from extraction import run
            error_message = run(test_func_name, kernel_func_name, kernel_module_path, test_script_output)

            # Add test results to iteration history
            previous_iteration_info.append(f"Test result: {error_message[:500]}...")
            

            if iteration > 0:  # Skip for the first iteration as we don't have a previous solution to compare
                
                
                # Extract error line from old error message if possible
                old_error_line, _ = extract_error_details(old_error_message)
                new_error_line, _ = extract_error_details(error_message)

                
                
                old_error_line_info = f"Error occurred at line: {old_error_line}" if old_error_line else "Error line could not be determined."
                new_error_line_info = f"Error occurred at line: {new_error_line}" if new_error_line else "Error line could not be determined."
                
                change_report_prompt = ChatPromptTemplate.from_template(
                    "You are analyzing the results of changes made to fix errors in a NKI kernel.\n\n"
                    "Previous error message:\n{old_error_message}\n\n"
                    "Previous error line information:\n{old_error_line_info}\n\n"
                    "Applied solution (reasoning):\n{reasoning}\n\n"
                    "New error message after applying the solution:\n{new_error_message}\n\n"
                    "New error line information:\n{new_error_line_info}\n\n"
                    "Please provide your analysis in the following JSON format:\n"
                    "```json\n"
                    "{{\n"
                    " \"correct\": boolean, // true if the fix resolved the initial problem, false otherwise\n"
                    " \"report\": \"string\" // brief explanation of why the solution worked or didn't work\n"
                    "}}\n"
                    "```\n\n"
                    "The 'correct' field should be true if the exact error we had last time has been fixed."
                    "it is still deemed correct even if a different error arises, we are just focusing on the "
                    "last error we were trying to fix\n"
                    "Remember, if the previous error and the new error are different, that means the solution is correct and should be true"
                    "Keep your report brief and focused on the specific changes and their effects. This is important"
                    "remember to keep the report consise and focused on key words on why it worked or failed"
                )
                change_report_chain = (
                    change_report_prompt
                    | query_llm
                    | StrOutputParser()
                )
                try:
                    change_report_json = invoke_chain_with_retry(change_report_chain, {
                        "old_error_message": old_error_message,
                        "old_error_line_info": old_error_line_info,
                        "reasoning": reasoning_text,
                        "new_error_message": error_message,
                        "new_error_line_info": new_error_line_info
                    },
                )
                except Exception as e:
                    print(f"Error in change report generation: {e}")
                    change_report_json = '{"correct": false, "report": "Error occurred during report generation"}'
                
                # Extract JSON from the response (in case there's additional text)
                json_match = re.search(r'```json\s*(.*?)\s*```', change_report_json, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = change_report_json
                
                # Clean up potential comment lines from the JSON
                json_str = re.sub(r'//.*', '', json_str)
                
                try:
                    report_data = json.loads(json_str)
                    correct = report_data.get("correct", False)
                    report = report_data.get("report", "No explanation provided")
                except json.JSONDecodeError:
                    # Fallback in case JSON parsing fails
                    print("Failed to parse JSON response. Using default values.")
                    correct = False
                    report = change_report_json
                
                
                # Add report to iteration history
                previous_iteration_info.append(f"Change report: correct={correct}, report={report}")
                
                # Log all the data from this iteration to the consolidated log file
                log_iteration_data(
                    consolidated_log_path,
                    iteration + 1,
                    error_message,
                    error_line,
                    error_description,
                    reasoning_text,
                    kernel_code,
                    error_message,
                    report_data if 'report_data' in locals() else None
                )

                # Update the previous error message for the next iteration
                previous_error_message = error_message

                # If no errors, we're done
                if "Error" not in error_message and "error" not in error_message and "ERROR" not in error_message:
                    log_iteration_data(
                        consolidated_log_path,
                        iteration + 1,
                        "Success - No errors detected",
                        None,
                        None,
                        reasoning_text if reasoning_text else "Final successful generation",
                        kernel_code,
                        error_message,
                        {"correct": True, "report": "Final successful iteration with no errors detected."}
                    )
                    print("No errors detected! Kernel generation successful.")
                    return 1
                
                # Pause for review before the next iteration if needed
                if iteration < max_iterations - 1:
                    print("Kernel iteration process completed.")
                    

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in kernel generation pipeline: {e}")
        
        # Save the error
        with open(output_address, "w") as f:
            f.write(f"Error generating kernel: {str(e)}\n\n{error_details}")






















if __name__ == "__main__":
    # Define constant file paths
    #TODO change depending on system
    
    # multi_element_operators = [
    #     "softmax", "log_softmax", "max", "min", "sum", "mean", "var", "std", "norm",
    #     "cumsum", "cumprod", "prod", "round", "floor", "ceil", "trunc", "sign",
    #     "where", "eq", "ne", "gt", "lt", "clamp", "sort", "topk", "kthvalue", "median",
    #     "mode", "percentile", "logsumexp", "amax", "amin", "all", "any", "bincount",
    #     "unique", "unique_consecutive"
    # ]

    # multi_element_test_names = [
    #     "test_torch_softmax",
    #     "test_torch_log_softmax",
    #     "test_torch_max",
    #     "test_torch_min",
    #     "test_torch_sum",
    #     "test_torch_mean",
    #     "test_torch_var",
    #     "test_torch_std",
    #     "test_torch_norm",
    #     "test_torch_cumsum",
    #     "test_torch_cumprod",
    #     "test_torch_prod",
    #     "test_torch_round",
    #     "test_torch_floor",
    #     "test_torch_ceil",
    #     "test_torch_trunc",
    #     "test_torch_sign",
    #     "test_torch_where",
    #     "test_torch_eq",
    #     "test_torch_ne",
    #     "test_torch_gt",
    #     "test_torch_lt",
    #     "test_torch_clamp",
    #     "test_torch_sort",
    #     "test_torch_topk",
    #     "test_torch_kthvalue",
    #     "test_torch_median",
    #     "test_torch_mode",
    #     "test_torch_percentile",
    #     "test_torch_logsumexp",
    #     "test_torch_amax",
    #     "test_torch_amin",
    #     "test_torch_all",
    #     "test_torch_any",
    #     "test_torch_bincount",
    #     "test_torch_unique",
    #     "test_torch_unique_consecutive"
    # ]

    # tests_passed_dict = {}

    # elementwise_operators = [
    #     "add", "sub", "mul", "div", "abs", "exp", "log", "sqrt", "rsqrt",
    #     "pow", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh",
    #     "tanh", "sigmoid", "relu", "threshold"
    # ]
    # elementwise_test_names = [
    #     "test_torch_addition",
    #     "test_torch_subtraction",
    #     "test_torch_multiplication",
    #     "test_torch_division",
    #     "test_torch_absolute",
    #     "test_torch_exponential",
    #     "test_torch_log",
    #     "test_torch_sqrt",
    #     "test_torch_rsqrt",
    #     "test_torch_power",
    #     "test_torch_sine",
    #     "test_torch_cosine",
    #     "test_torch_tangent",
    #     "test_torch_arcsine",
    #     "test_torch_arccosine",
    #     "test_torch_arctangent",
    #     "test_torch_hyperbolic_sine",
    #     "test_torch_hyperbolic_cosine",
    #     "test_torch_hyperbolic_tangent",
    #     "test_torch_sigmoid",
    #     "test_torch_relu",
    #     "test_torch_threshold"
    # ]   

    multi_element_operators = [
        "softmax"
    ]

    multi_element_test_names = [
        "test_torch_softmax",
    ]

    tests_passed_dict = {}

    for i in range(len(multi_element_operators)):
        operator = multi_element_operators[i]
        test_name = multi_element_test_names[i]
        system_prompt_path = f"/home/ubuntu/torch2nki/generation/langchain_single_pass/langchain_files/langchain_prompts/system_prompt_langchain.txt"
        user_prompt_path = f"/home/ubuntu/torch2nki/prompts/{operator}_nki_prompt.txt"
        output_address = f"/home/ubuntu/torch2nki/generation/langchain_single_pass/langchain_files/langchain_outputs/{operator}_nki_kernel.txt"
        kernel_module_path = f"/home/ubuntu/torch2nki/generation/langchain_single_pass/langchain_files/langchain_outputs/{operator}_nki_kernel.py"
        test_script_output = f"/home/ubuntu/torch2nki/generation/langchain_single_pass/langchain_files/langchain_outputs/{operator}_error_message.txt"
        reasoning_log_path = f"/home/ubuntu/torch2nki/generation/langchain_single_pass/langchain_files/langchain_outputs/{operator}_reasoning_log.txt"
        error_doc_path = f"/home/ubuntu/torch2nki/documentation/nki_documentation/nki_error_messages.txt"
        docs_dir = f"/home/ubuntu/torch2nki/documentation/nki_documentation/nki_language_apis_parsed"
        kernel_func_name = f"nki_{operator}"
        # Get credentials
        pinecone_api_key = os.environ.get('PINECONE_API_KEY')
        pinecone_index_name = os.environ.get('PINECONE_INDEX_NAME')

        
        # Run the updated generator with direct documentation and error loop
        result = generate_kernel_with_direct_docs_and_error_loop(
            kernel_func_name,
            system_prompt_path,
            user_prompt_path,
            output_address,
            kernel_module_path,
            test_name,
            test_script_output,
            reasoning_log_path,
            error_doc_path,
            docs_dir,
            max_iterations=10
        )
        if result:
            print(result)
            tests_passed_dict[operator] = True
        else:
            tests_passed_dict[operator] = False

    # Save test_passed_dict to a file, and make the file if it doesn't exist
    with open(f"/home/ubuntu/torch2nki/generation/langchain_single_pass/langchain_files/langchain_outputs/test_passed_dict.json", "w") as f:
        json.dump(tests_passed_dict, f)
    
    