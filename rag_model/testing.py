import rag
import torch

# Should proly move this to another file, with the whole testing pipeline
# THIS PORTION IS ON ANSWER COMPARING WITH GROUND TRUTH 
prompt = rag.PromptTemplate(
    template="""     
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing whether an answer is similar to its ground truth
    Give a binary score 'YES' or 'NO' to indicate whether the answer is truthful.
    Provide the binary score as value to a single key 'score' in JSON format.
    Do not provide any preamble or explanation. 
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the generated answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the ground truth: {ground_truth}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["generation", "ground_truth"],
)

answer_comparer = prompt | rag.llm | rag.JsonOutputParser()

# Should proly move this to another file, with the whole testing pipeline
def load_questions_and_answers(file_path):
    """
    Load questions and answers from a text file.
    Assumes the file is formatted with alternating lines of questions and answers.
    """
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    
    # Remove any empty lines
    lines = [line.strip() for line in lines if line.strip()]
    
    # Separate questions and answers
    questions = lines[::2]  # Even indices (0, 2, 4, ...)
    answers = lines[1::2]   # Odd indices (1, 3, 5, ...)
    
    return questions, answers

# Should proly move this to another file, with the whole testing pipeline
def test_llm_pipeline(workflow, questions_file):
    """
    Test LLM pipeline by comparing generated answers with ground truth.
    
    Args:
    - workflow: Compiled workflow for answer generation
    - questions_file: Path to the text file containing questions and answers
    
    Returns:
    - Accuracy percentage
    - List of incorrectly answered questions
    """
    # Load questions and answers
    questions, ground_truth_answers = load_questions_and_answers(questions_file)
    
    # Tracking variables
    correct_answers = 0
    incorrect_questions = []
    
    # Iterate through questions
    for i, question in enumerate(questions):        
        # Prepare input
        inputs = {"question": question}
        
        # Generate answer
        try:
            for output in workflow.stream(inputs):
                for key, value in output.items():
                    generated_answer = value.get("generation", "")
            
            # Compare generated answer with ground truth
            # You might want to add more sophisticated comparison (e.g., fuzzy matching)
            score = answer_comparer.invoke({"generation": generated_answer.lower().strip(), "ground_truth":  ground_truth_answers[i].lower().strip()})
            grade = score['score']
            if grade.upper() == 'YES':
                is_correct = True
            else:
                is_correct = False
            
            # Track results
            if is_correct:
                correct_answers += 1
            else:
                incorrect_questions.append({
                    "question": question,
                    "generated_answer": generated_answer,
                    "ground_truth": ground_truth_answers[i]
                })
            
            # Optional: Print current question and result
            print(f"Question {i+1}: {'Correct' if is_correct else 'Incorrect'}")
            print(f"{question}")
            print(f"Generated: {generated_answer}")
            print(f"Ground Truth: {ground_truth_answers[i]}\n")
        
        except Exception as e:
            print(f"Error processing question {i+1}: {e}")
    
    # Calculate accuracy
    accuracy = (correct_answers / len(questions)) * 100
    
    return accuracy, incorrect_questions

# Main execution
if __name__ == "__main__":
    # Path to your questions file
    questions_file = "questions.txt"
    
    # Compile workflow (assuming this is already done)
    vectorstore = rag.index_and_embed_cur_docs()
    workflow = rag.create_graph(vectorstore)
    app = workflow.compile()
    
    # Run the test
    accuracy, incorrect_questions = test_llm_pipeline(app, questions_file)
    
    # Print final results
    print("\n--- Test Results ---")
    print(f"Accuracy: {accuracy:.2f}%")
    print("\nIncorrectly Answered Questions:")
    for q in incorrect_questions:
        print(f"Question: {q['question']}")
        print(f"Generated Answer: {q['generated_answer']}")
        print(f"Ground Truth: {q['ground_truth']}\n")

    print("Prog Finished")
