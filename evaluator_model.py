from typing import List


class TPOEvaluatorModel:
    def __init__(self, llm_engine):
        """
        Initialize the evaluator model using OpenAI's API.
        
        Args:
            llm_engine: The LLM engine to use for evaluation
        """
        self.llm_engine = llm_engine
    
    def get_contrastive_samples(self, question: str, golden_answer: str, answers: List[str]) -> tuple[str, str]:
        """
        Get the contrastive samples for the question and answers.
        
        Args:
            question: The question being asked
            golden_answer: The reference/golden answer
            answers: List of candidate answers to evaluate
            
        Returns:
            tuple[str, str]: A tuple containing (best_answer, worst_answer)
        """
        # Construct the prompt for evaluation
        prompt = f"""Given the following question and reference answer, evaluate the candidate answers and identify the best and worst ones.

Question: {question}

Reference Answer: {golden_answer}

Candidate Answers:
{chr(10).join(f"{i+1}. {ans}" for i, ans in enumerate(answers))}

Please analyze these answers and identify:
1. The best answer (most accurate, complete, and well-structured)
2. The worst answer (least accurate, incomplete, or poorly structured)

Return your response in the following format:
Best: [answer number]
Worst: [answer number]

Only return the numbers, nothing else."""

        # Get evaluation from LLM
        response = self.llm_engine.generate(prompt)
        
        # Parse the response to get best and worst indices
        try:
            best_idx = int(response.split("Best:")[1].split()[0]) - 1
            worst_idx = int(response.split("Worst:")[1].split()[0]) - 1
            
            return answers[best_idx], answers[worst_idx]
        except (IndexError, ValueError):
            # If parsing fails, return the first and last answers as fallback
            return answers[0], answers[-1]
        
        
    