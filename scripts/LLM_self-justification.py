#!/usr/bin/env python3
"""
Enhanced SQL Semantic Validation with Multiple Methods
Extended from existing run_detection.py framework
Follows the existing code structure and patterns
"""

import json
import sqlite3
from openai import OpenAI
import time
import os
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SQLSemanticValidator:
    def __init__(self, db_root: str, model: str = "gpt-4o"):
        """Initialize the validator with OpenAI client"""
        self.client = client
        self.model = model
        self.db_root = db_root
        self.results = []
        
    def load_data(self, file_path: str, sample_size: int = None) -> List[Dict]:
        """Load NL2SQL-BUGs dataset"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if sample_size:
            data = data[:sample_size]
            
        print(f"Loaded {len(data)} samples")
        return data
    
    def get_database_schema(self, db_id: str) -> str:
        """Extract database schema from sqlite file"""
        db_path = f"{self.db_root}/{db_id}/{db_id}.sqlite"
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema_info = []
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                col_info = [f"{col[1]} {col[2]}" for col in columns]
                schema_info.append(f"Table {table_name}: ({', '.join(col_info)})")
            
            conn.close()
            return "\n".join(schema_info)
            
        except Exception as e:
            print(f"Error loading schema for {db_id}: {e}")
            return f"Database: {db_id}"
    
    def call_openai_api(self, prompt: str, max_retries: int = 3) -> str:
        """Call OpenAI API with retry logic using new client"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert SQL semantic validator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
            
            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return f"API_ERROR: {str(e)}"
    

    
    def self_reflection_validation(self, question: str, sql: str, schema: str) -> Dict:
        """Enhanced self-reflection method"""
        prompt = f"""
        You are an expert SQL validator. Analyze if this SQL query correctly answers the natural language question.

        Question: "{question}"
        SQL Query: {sql}
        Database Schema: {schema}

        Please provide detailed analysis:
        1. REASONING: Step-by-step explanation of what the SQL does
        2. ISSUES: Any semantic problems you identify (or "None" if correct)
        3. VERDICT: "CORRECT" or "INCORRECT"
        4. CONFIDENCE: Your confidence score from 0-100

        Focus on semantic correctness - does the SQL answer what was asked?

        Format your response exactly as:
        REASONING: [your step-by-step analysis]
        ISSUES: [problems found or "None"]
        VERDICT: [CORRECT or INCORRECT]
        CONFIDENCE: [0-100]
        """
        
        response = self.call_openai_api(prompt)
        
        # Parse structured response
        verdict_match = re.search(r'VERDICT:\s*(CORRECT|INCORRECT)', response, re.IGNORECASE)
        confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', response)
        issues_match = re.search(r'ISSUES:\s*(.+?)(?=\n|VERDICT|$)', response, re.DOTALL)
        
        is_correct = verdict_match and verdict_match.group(1).upper() == "CORRECT"
        confidence = int(confidence_match.group(1)) if confidence_match else None
        issues = issues_match.group(1).strip() if issues_match else None
        
        return {
            "method": "self_reflection",
            "prediction": is_correct,
            "confidence": confidence,
            "issues_identified": issues,
            "raw_response": response
        }
    
    def chain_of_thought_validation(self, question: str, sql: str, schema: str) -> Dict:
        """Chain of thought reasoning method"""
        prompt = f"""
        Let's analyze this SQL query step by step to determine if it correctly answers the question.

        Question: "{question}"
        SQL: {sql}
        Schema: {schema}

        Step 1: What is the question asking for?
        Step 2: What tables and columns does the SQL use?
        Step 3: What operations does the SQL perform?
        Step 4: Does this match what the question requested?
        Step 5: Are there any logical errors or missing conditions?

        Final Answer: CORRECT or INCORRECT
        """
        
        response = self.call_openai_api(prompt)
        is_correct = response.split("Final Answer:")[-1].strip().upper().startswith("CORRECT")
        
        return {
            "method": "chain_of_thought",
            "prediction": is_correct,
            "raw_response": response,
            "confidence": None
        }
    
    def validate_single_item(self, item: Dict) -> Dict:
        """Validate a single data item using multiple methods"""
        question = item['question']
        sql = item['sql']
        db_id = item['db_id']
        schema = self.get_database_schema(db_id)
        
        # Ground truth (label=False means incorrect SQL)
        ground_truth = item['label']
        
        # Run validation methods
        reflection_result = self.self_reflection_validation(question, sql, schema)
        cot_result = self.chain_of_thought_validation(question, sql, schema)
        
        return {
            "id": item['id'],
            "question": question,
            "sql": sql,
            "db_id": db_id,
            "ground_truth": ground_truth,
            "error_types": item.get('error_types', []),
            "evidence": item.get('evidence', ''),
            "methods": {
                "self_reflection": reflection_result,
                "chain_of_thought": cot_result
            }
        }
    
    def run_evaluation(self, examples: List[Dict], result_file_path: str) -> List[Dict]:
        """Run complete evaluation on dataset"""
        results = []
        
        print(f"Starting evaluation on {len(examples)} samples...")
        
        # Create results directory and open result file
        import os
        os.makedirs("results", exist_ok=True)
        
        with open(result_file_path, "w") as result_file:
            for i, item in enumerate(examples):
                print(f"Processing item {i+1}/{len(examples)} (ID: {item['id']})")
                
                try:
                    result = self.validate_single_item(item)
                    results.append(result)
                    
                    # Write to jsonl file (following your pattern)
                    result_file.write(json.dumps(result) + "\n")
                    result_file.flush()  # Ensure immediate write
                    
                except Exception as e:
                    print(f"Error processing item {item['id']}: {e}")
                    continue
        
        self.results = results
        return results
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate performance metrics for all methods"""
        metrics = {}
        
        for method_name in ["self_reflection", "chain_of_thought"]:
            predictions = []
            ground_truths = []
            confidences = []
            
            for result in results:
                if method_name in result["methods"]:
                    method_result = result["methods"][method_name]
                    predictions.append(method_result["prediction"])
                    ground_truths.append(result["ground_truth"])
                    
                    if method_result["confidence"] is not None:
                        confidences.append(method_result["confidence"])
            
            # Calculate basic metrics
            correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
            total = len(predictions)
            accuracy = correct / total if total > 0 else 0
            
            # Calculate precision, recall for "incorrect" detection
            tp = sum(1 for p, g in zip(predictions, ground_truths) if p == True and g == True)
            fp = sum(1 for p, g in zip(predictions, ground_truths) if p == True and g == False)
            fn = sum(1 for p, g in zip(predictions, ground_truths) if p == False and g == True)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[method_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "total_samples": total,
                "avg_confidence": sum(confidences) / len(confidences) if confidences else None
            }
        
        return metrics
    
    def analyze_by_error_type(self, results: List[Dict]) -> Dict:
        """Analyze performance by error type"""
        error_analysis = defaultdict(lambda: defaultdict(list))
        
        for result in results:
            for error_info in result["error_types"]:
                error_type = error_info["error_type"]
                sub_type = error_info.get("sub_error_type", "Unknown")
                
                for method_name, method_result in result["methods"].items():
                    is_correct = method_result["prediction"] == result["ground_truth"]
                    error_analysis[error_type][method_name].append(is_correct)
        
        # Calculate accuracy by error type
        summary = {}
        for error_type, methods in error_analysis.items():
            summary[error_type] = {}
            for method, results_list in methods.items():
                accuracy = sum(results_list) / len(results_list) if results_list else 0
                summary[error_type][method] = {
                    "accuracy": accuracy,
                    "count": len(results_list)
                }
        
        return summary
    
    def save_results(self, results: List[Dict], suffix: str = "") -> str:
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/validation_results_{timestamp}_{suffix}.json"
        
        Path("results").mkdir(exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {filename}")
        return filename
    
    def print_summary(self, results: List[Dict]):
        """Print evaluation summary"""
        metrics = self.calculate_metrics(results)
        error_analysis = self.analyze_by_error_type(results)
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        # Overall metrics
        print("\nOverall Performance:")
        for method, stats in metrics.items():
            print(f"\n{method.upper()}:")
            print(f"  Accuracy: {stats['accuracy']:.3f}")
            print(f"  Precision: {stats['precision']:.3f}")
            print(f"  Recall: {stats['recall']:.3f}")
            print(f"  F1-Score: {stats['f1_score']:.3f}")
            if stats['avg_confidence']:
                print(f"  Avg Confidence: {stats['avg_confidence']:.1f}")
        
        # Error type analysis
        print("\nPerformance by Error Type:")
        for error_type, methods in error_analysis.items():
            print(f"\n{error_type}:")
            for method, stats in methods.items():
                print(f"  {method}: {stats['accuracy']:.3f} ({stats['count']} samples)")

def main():
    """Main execution function - matches your existing style"""
    # Paths
    DATA_PATH = "bug-data/NL2SQL-Bugs.json"
    DB_ROOT = "BIRD/dev_20240627/dev_databases"
    MAX_EXAMPLES = 50  # Reduced for quick testing
    
    # Result path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULT_PATH = f"results/enhanced_results_{MAX_EXAMPLES}_{timestamp}.jsonl"
    
    # Load data
    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, "r") as f:
        examples = json.load(f)
    
    if MAX_EXAMPLES:
        examples = examples[:MAX_EXAMPLES]
    
    print(f"Loaded {len(examples)} examples")
    
    # Initialize validator (no need to pass API key now)
    validator = SQLSemanticValidator(DB_ROOT)
    
    # Run evaluation
    print("Starting SQL Semantic Validation Evaluation...")
    results = validator.run_evaluation(examples, RESULT_PATH)
    
    # Print summary
    validator.print_summary(results)
    
    print(f"\nEvaluation complete! Results saved to {RESULT_PATH}")
    
    # Additional analysis
    metrics = validator.calculate_metrics(results)
    error_analysis = validator.analyze_by_error_type(results)
    
    # Save summary metrics
    summary_path = f"results/summary_{MAX_EXAMPLES}_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "error_analysis": error_analysis,
            "total_samples": len(results)
        }, f, indent=2)
    
    print(f"Summary saved to {summary_path}")



if __name__ == "__main__":
    main()