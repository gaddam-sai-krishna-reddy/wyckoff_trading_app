"""
RAG Evaluation and Testing Tools
Helps evaluate the effectiveness of your RAG implementation
"""

import pandas as pd
from typing import List, Dict, Tuple
import time
from chatbot_service import get_answer, retrieve_docs, logger

class RAGEvaluator:
    """Evaluate RAG system performance."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_retrieval(self, test_questions: List[str]) -> Dict:
        """Evaluate retrieval quality."""
        results = {
            "total_questions": len(test_questions),
            "avg_docs_retrieved": 0,
            "avg_retrieval_time": 0,
            "questions_with_docs": 0,
            "questions_without_docs": 0
        }
        
        total_docs = 0
        total_time = 0
        
        for question in test_questions:
            start_time = time.time()
            docs = retrieve_docs(question)
            retrieval_time = time.time() - start_time
            
            total_docs += len(docs)
            total_time += retrieval_time
            
            if docs:
                results["questions_with_docs"] += 1
            else:
                results["questions_without_docs"] += 1
        
        if test_questions:
            results["avg_docs_retrieved"] = total_docs / len(test_questions)
            results["avg_retrieval_time"] = total_time / len(test_questions)
        
        return results
    
    def evaluate_generation(self, test_questions: List[str]) -> Dict:
        """Evaluate answer generation quality."""
        results = {
            "total_questions": len(test_questions),
            "avg_generation_time": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "avg_answer_length": 0
        }
        
        total_time = 0
        total_length = 0
        successful = 0
        
        for question in test_questions:
            start_time = time.time()
            try:
                answer = get_answer(question)
                generation_time = time.time() - start_time
                
                # Simple logic: only count as successful if it's a real answer
                if answer and not answer.startswith("Sorry, I encountered an error") and not answer.startswith("I couldn't find any relevant information"):
                    successful += 1
                    total_time += generation_time
                    total_length += len(answer)
                else:
                    results["failed_generations"] += 1
                
            except Exception as e:
                logger.error(f"Error evaluating question '{question}': {e}")
                results["failed_generations"] += 1
        
        results["successful_generations"] = successful
        if successful > 0:
            results["avg_generation_time"] = total_time / successful
            results["avg_answer_length"] = total_length / successful
        
        return results
    
    def run_comprehensive_evaluation(self, test_questions: List[str]) -> Dict:
        """Run comprehensive RAG evaluation."""
        logger.info("Starting RAG evaluation...")
        
        # Evaluate retrieval
        retrieval_results = self.evaluate_retrieval(test_questions)
        logger.info(f"Retrieval Results: {retrieval_results}")
        
        # Evaluate generation
        generation_results = self.evaluate_generation(test_questions)
        logger.info(f"Generation Results: {generation_results}")
        
        # Combined metrics
        combined_results = {
            "retrieval": retrieval_results,
            "generation": generation_results,
            "overall": {
                "total_questions": len(test_questions),
                "retrieval_success_rate": retrieval_results["questions_with_docs"] / len(test_questions) if test_questions else 0,
                "generation_success_rate": generation_results["successful_generations"] / len(test_questions) if test_questions else 0,
                "avg_total_time": retrieval_results["avg_retrieval_time"] + generation_results["avg_generation_time"]
            }
        }
        
        return combined_results

def create_test_questions() -> List[str]:
    """Create test questions for evaluation."""
    return [
        "What is a Spring in Wyckoff methodology?",
        "How does accumulation work in Wyckoff?",
        "What is the difference between a Spring and a Test?",
        "How do you identify a markup phase?",
        "What is distribution in Wyckoff theory?",
        "How do you measure volume in Wyckoff analysis?",
        "What is a climax in Wyckoff methodology?",
        "How do you identify support and resistance in Wyckoff?",
        "What is the purpose of a trading range?",
        "How do you use volume analysis in Wyckoff?"
    ]

def print_evaluation_report(results: Dict):
    """Print a formatted evaluation report."""
    print("\n" + "="*50)
    print("RAG EVALUATION REPORT")
    print("="*50)
    
    # Retrieval metrics
    print("\n📊 RETRIEVAL METRICS:")
    retrieval = results["retrieval"]
    print(f"  • Total Questions: {retrieval['total_questions']}")
    print(f"  • Questions with Docs: {retrieval['questions_with_docs']}")
    print(f"  • Questions without Docs: {retrieval['questions_without_docs']}")
    print(f"  • Avg Docs Retrieved: {retrieval['avg_docs_retrieved']:.2f}")
    print(f"  • Avg Retrieval Time: {retrieval['avg_retrieval_time']:.3f}s")
    
    # Generation metrics
    print("\n🤖 GENERATION METRICS:")
    generation = results["generation"]
    print(f"  • Successful Generations: {generation['successful_generations']}")
    print(f"  • Failed Generations: {generation['failed_generations']}")
    print(f"  • Avg Generation Time: {generation['avg_generation_time']:.3f}s")
    print(f"  • Avg Answer Length: {generation['avg_answer_length']:.0f} chars")
    
    # Overall metrics
    print("\n📈 OVERALL PERFORMANCE:")
    overall = results["overall"]
    print(f"  • Retrieval Success Rate: {overall['retrieval_success_rate']:.1%}")
    print(f"  • Generation Success Rate: {overall['generation_success_rate']:.1%}")
    print(f"  • Avg Total Time: {overall['avg_total_time']:.3f}s")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if overall['retrieval_success_rate'] < 0.8:
        print("  ⚠️  Low retrieval success rate - consider improving document structure or embedding model")
    if overall['generation_success_rate'] < 0.8:
        print("  ⚠️  Low generation success rate - consider improving prompts or model")
    if overall['avg_total_time'] > 5.0:
        print("  ⚠️  Slow response time - consider optimizing model or using caching")
    if overall['retrieval_success_rate'] >= 0.8 and overall['generation_success_rate'] >= 0.8:
        print("  ✅ Good overall performance!")
    
    print("="*50)

if __name__ == "__main__":
    # Run evaluation
    evaluator = RAGEvaluator()
    test_questions = create_test_questions()
    
    print("Running RAG evaluation...")
    results = evaluator.run_comprehensive_evaluation(test_questions)
    print_evaluation_report(results) 