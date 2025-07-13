#!/usr/bin/env python3
"""
Uncertainty-Based SQL Semantic Validation
Extends the existing validation framework with uncertainty quantification
Follows the same structure and patterns as the existing code
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
from collections import defaultdict, Counter
from dotenv import load_dotenv
import statistics

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class UncertaintyValidator:
    def __init__(self, db_root: str, model: str = "gpt-4o"):
        """Initialize the uncertainty validator"""
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
        """Extract database schema from sqlite file with better error handling"""
        db_path = f"{self.db_root}/{db_id}/{db_id}.sqlite"
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all table names with better error handling
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema_info = []
            for table in tables:
                table_name = table[0]
                try:
                    # Use quotes around table name to handle reserved keywords
                    cursor.execute(f"PRAGMA table_info(\"{table_name}\");")
                    columns = cursor.fetchall()
                    
                    # Format: column_name data_type
                    col_info = []
                    for col in columns:
                        col_name = col[1]
                        col_type = col[2]
                        # Handle reserved keywords in column names too
                        if col_name.lower() in ['order', 'group', 'select', 'from', 'where', 'join']:
                            col_info.append(f'"{col_name}" {col_type}')
                        else:
                            col_info.append(f"{col_name} {col_type}")
                    
                    schema_info.append(f"Table {table_name}: ({', '.join(col_info)})")
                    
                except Exception as table_error:
                    print(f"Warning: Could not load columns for table {table_name}: {table_error}")
                    schema_info.append(f"Table {table_name}: (columns unavailable)")
            
            conn.close()
            
            if not schema_info:
                return f"Database: {db_id} (no accessible tables)"
            
            return "\n".join(schema_info)
            
        except Exception as e:
            print(f"Error loading schema for {db_id}: {e}")
            # Return a basic schema description as fallback
            return f"Database: {db_id} (schema unavailable due to: {str(e)})"
    

    def call_openai_api(self, prompt: str, temperature: float = 0.1, max_retries: int = 3) -> str:
        """Call OpenAI API with retry logic and configurable temperature"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert SQL semantic validator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
            
            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return f"API_ERROR: {str(e)}"

      



    def consistency_based_uncertainty(self, question: str, sql: str, schema: str, num_samples: int = 5) -> Dict:
        """Method 1: Multiple sampling consistency check"""
        prompt = f"""
        You are an expert SQL validator. Analyze if this SQL query correctly answers the natural language question.

        Question: "{question}"
        SQL Query: {sql}
        Database Schema: {schema}

        Please respond with exactly one word: CORRECT or INCORRECT
        """
        
        predictions = []
        responses = []
        
        # Multiple sampling with higher temperature for variation
        for i in range(num_samples):
            response = self.call_openai_api(prompt, temperature=0.7)
            prediction = 'CORRECT' if 'CORRECT' in response.upper() else 'INCORRECT'
            
            predictions.append(prediction)
            responses.append(response)
        
        # Calculate consistency metrics
        prediction_counts = Counter(predictions)
        most_common_prediction, max_count = prediction_counts.most_common(1)[0]
        consistency_rate = max_count / num_samples
        
        # Uncertainty score (higher = more uncertain)
        uncertainty_score = 1.0 - consistency_rate
        
        # Final prediction based on majority vote
        final_prediction = most_common_prediction == 'CORRECT'
        
        return {
            "method": "consistency_uncertainty",
            "prediction": final_prediction,
            "uncertainty_score": uncertainty_score,
            "consistency_rate": consistency_rate,
            "individual_predictions": predictions,
            "prediction_distribution": dict(prediction_counts),
            "raw_responses": responses,
            "num_samples": num_samples
        }
    
    def confidence_based_uncertainty(self, question: str, sql: str, schema: str) -> Dict:
        """Method 2: Explicit confidence scoring"""
        prompt = f"""
        You are an expert SQL validator. Analyze if this SQL query correctly answers the natural language question.

        Question: "{question}"
        SQL Query: {sql}
        Database Schema: {schema}

        Please provide your analysis in this exact format:
        VERDICT: CORRECT or INCORRECT
        CONFIDENCE: [0-100]
        REASONING: [Brief explanation of your reasoning]

        The confidence score should reflect how certain you are about your verdict.
        """
        
        response = self.call_openai_api(prompt, temperature=0.1)
        
        # Parse structured response
        verdict_match = re.search(r'VERDICT:\s*(CORRECT|INCORRECT)', response, re.IGNORECASE)
        confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', response)
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\n|$)', response, re.DOTALL)
        
        # Extract values
        verdict = verdict_match.group(1).upper() if verdict_match else "INCORRECT"
        confidence = int(confidence_match.group(1)) if confidence_match else 50
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        # Convert to prediction and uncertainty
        prediction = verdict == "CORRECT"
        uncertainty_score = 1.0 - (confidence / 100.0)
        
        return {
            "method": "confidence_uncertainty", 
            "prediction": prediction,
            "uncertainty_score": uncertainty_score,
            "confidence": confidence,
            "verdict": verdict,
            "reasoning": reasoning,
            "raw_response": response
        }
    
    def linguistic_uncertainty_detection(self, question: str, sql: str, schema: str) -> Dict:
        """Method 3: Detect uncertainty through linguistic cues"""
        prompt = f"""
        You are an expert SQL validator. Analyze if this SQL query correctly answers the natural language question.

        Question: "{question}"
        SQL Query: {sql}
        Database Schema: {schema}

        Please provide a detailed analysis explaining your reasoning step by step.
        End your response with: FINAL_VERDICT: CORRECT or INCORRECT
        """
        
        response = self.call_openai_api(prompt, temperature=0.3)
        
        # Uncertainty linguistic indicators
        uncertainty_phrases = [
            "might", "could", "possibly", "perhaps", "seems", "appears", 
            "likely", "probably", "uncertain", "unclear", "ambiguous",
            "not sure", "difficult to", "hard to", "may be", "possibly",
            "可能", "也许", "似乎", "看起来", "不确定", "不太确定", "或许"
        ]
        
        # Count uncertainty phrases (case insensitive)
        response_lower = response.lower()
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in response_lower)
        
        # Extract final verdict
        verdict_match = re.search(r'FINAL_VERDICT:\s*(CORRECT|INCORRECT)', response, re.IGNORECASE)
        verdict = verdict_match.group(1).upper() if verdict_match else "INCORRECT"
        prediction = verdict == "CORRECT"
        
        # Calculate uncertainty score based on linguistic cues
        # Normalize by response length and cap at 1.0
        response_length = len(response.split())
        normalized_uncertainty_count = uncertainty_count / max(response_length / 50, 1)  # Per ~50 words
        linguistic_uncertainty = min(normalized_uncertainty_count, 1.0)
        
        return {
            "method": "linguistic_uncertainty",
            "prediction": prediction,
            "uncertainty_score": linguistic_uncertainty,
            "uncertainty_phrases_found": uncertainty_count,
            "detected_phrases": [phrase for phrase in uncertainty_phrases if phrase in response_lower],
            "verdict": verdict,
            "response_length": response_length,
            "raw_response": response
        }
    
    def ensemble_uncertainty_validation(self, question: str, sql: str, schema: str) -> Dict:
        """Combined uncertainty validation using all methods"""
        # Run all three uncertainty methods
        consistency_result = self.consistency_based_uncertainty(question, sql, schema)
        confidence_result = self.confidence_based_uncertainty(question, sql, schema)
        linguistic_result = self.linguistic_uncertainty_detection(question, sql, schema)
        
        # Collect uncertainty scores and predictions
        uncertainty_scores = [
            consistency_result["uncertainty_score"],
            confidence_result["uncertainty_score"], 
            linguistic_result["uncertainty_score"]
        ]
        
        predictions = [
            consistency_result["prediction"],
            confidence_result["prediction"],
            linguistic_result["prediction"]
        ]
        
        # Calculate ensemble metrics
        avg_uncertainty = statistics.mean(uncertainty_scores)
        max_uncertainty = max(uncertainty_scores)
        min_uncertainty = min(uncertainty_scores)
        
        # Ensemble prediction (majority vote)
        prediction_votes = Counter(predictions)
        ensemble_prediction, vote_count = prediction_votes.most_common(1)[0]
        prediction_agreement = vote_count / len(predictions)
        
        # Overall uncertainty (combine multiple factors)
        overall_uncertainty = (avg_uncertainty + (1 - prediction_agreement)) / 2
        
        return {
            "method": "ensemble_uncertainty",
            "prediction": ensemble_prediction,
            "overall_uncertainty": overall_uncertainty,
            "uncertainty_scores": {
                "consistency": consistency_result["uncertainty_score"],
                "confidence": confidence_result["uncertainty_score"],
                "linguistic": linguistic_result["uncertainty_score"],
                "average": avg_uncertainty,
                "max": max_uncertainty,
                "min": min_uncertainty
            },
            "prediction_agreement": prediction_agreement,
            "individual_predictions": predictions,
            "high_uncertainty": overall_uncertainty > 0.7,  # Threshold can be tuned
            "individual_results": {
                "consistency": consistency_result,
                "confidence": confidence_result,
                "linguistic": linguistic_result
            }
        }
    
    def validate_single_item(self, item: Dict) -> Dict:
        """Validate a single data item using uncertainty methods"""
        question = item['question']
        sql = item['sql']
        db_id = item['db_id']
        schema = self.get_database_schema(db_id)
        
        # Ground truth (label=False means incorrect SQL)
        ground_truth = item['label']
        
        # Run uncertainty validation methods
        ensemble_result = self.ensemble_uncertainty_validation(question, sql, schema)
        
        return {
            "id": item['id'],
            "question": question,
            "sql": sql,
            "db_id": db_id,
            "ground_truth": ground_truth,
            "error_types": item.get('error_types', []),
            "evidence": item.get('evidence', ''),
            "methods": {
                "ensemble_uncertainty": ensemble_result
            }
        }
    
    def run_evaluation(self, examples: List[Dict], result_file_path: str) -> List[Dict]:
        """Run complete evaluation on dataset"""
        results = []
        
        print(f"Starting uncertainty validation on {len(examples)} samples...")
        
        # Create results directory and open result file
        os.makedirs("results", exist_ok=True)
        
        with open(result_file_path, "w") as result_file:
            for i, item in enumerate(examples):
                print(f"Processing item {i+1}/{len(examples)} (ID: {item['id']})")
                
                try:
                    result = self.validate_single_item(item)
                    results.append(result)
                    
                    # Write to jsonl file
                    result_file.write(json.dumps(result) + "\n")
                    result_file.flush()  # Ensure immediate write
                    
                except Exception as e:
                    print(f"Error processing item {item['id']}: {e}")
                    continue
        
        self.results = results
        return results
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate performance metrics for uncertainty methods"""
        metrics = {}
        
        predictions = []
        ground_truths = []
        uncertainty_scores = []
        high_uncertainty_flags = []
        
        for result in results:
            ensemble_result = result["methods"]["ensemble_uncertainty"]
            predictions.append(ensemble_result["prediction"])
            ground_truths.append(result["ground_truth"])
            uncertainty_scores.append(ensemble_result["overall_uncertainty"])
            high_uncertainty_flags.append(ensemble_result["high_uncertainty"])
        
        # Calculate basic metrics
        correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0
        
        # Calculate precision, recall for "correct" detection
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == True and g == True)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == True and g == False)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p == False and g == True)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Uncertainty-specific metrics
        avg_uncertainty = statistics.mean(uncertainty_scores)
        high_uncertainty_rate = sum(high_uncertainty_flags) / len(high_uncertainty_flags)
        
        # Correlation between uncertainty and correctness
        uncertainty_correct_correlation = self._calculate_uncertainty_correlation(
            uncertainty_scores, predictions, ground_truths
        )
        
        metrics["ensemble_uncertainty"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "total_samples": total,
            "avg_uncertainty": avg_uncertainty,
            "high_uncertainty_rate": high_uncertainty_rate,
            "uncertainty_correctness_correlation": uncertainty_correct_correlation
        }
        
        return metrics
    
    def _calculate_uncertainty_correlation(self, uncertainty_scores: List[float], 
                                         predictions: List[bool], 
                                         ground_truths: List[bool]) -> float:
        """Calculate correlation between uncertainty and prediction correctness"""
        correctness_scores = [1.0 if p == g else 0.0 for p, g in zip(predictions, ground_truths)]
        
        if len(uncertainty_scores) != len(correctness_scores):
            return 0.0
            
        # Simple correlation: higher uncertainty should correlate with lower correctness
        try:
            import numpy as np
            correlation = np.corrcoef(uncertainty_scores, correctness_scores)[0, 1]
            return -correlation  # Negative because we want high uncertainty = low correctness
        except:
            # Fallback calculation without numpy
            n = len(uncertainty_scores)
            if n < 2:
                return 0.0
                
            mean_u = sum(uncertainty_scores) / n
            mean_c = sum(correctness_scores) / n
            
            numerator = sum((u - mean_u) * (c - mean_c) for u, c in zip(uncertainty_scores, correctness_scores))
            u_var = sum((u - mean_u) ** 2 for u in uncertainty_scores)
            c_var = sum((c - mean_c) ** 2 for c in correctness_scores)
            
            if u_var == 0 or c_var == 0:
                return 0.0
                
            correlation = numerator / (u_var * c_var) ** 0.5
            return -correlation
    
    def analyze_by_uncertainty_level(self, results: List[Dict]) -> Dict:
        """Analyze performance by uncertainty level"""
        uncertainty_analysis = {
            "low_uncertainty": {"correct": 0, "total": 0},      # < 0.3
            "medium_uncertainty": {"correct": 0, "total": 0},   # 0.3 - 0.7  
            "high_uncertainty": {"correct": 0, "total": 0}      # > 0.7
        }
        
        for result in results:
            ensemble_result = result["methods"]["ensemble_uncertainty"]
            uncertainty = ensemble_result["overall_uncertainty"]
            is_correct = ensemble_result["prediction"] == result["ground_truth"]
            
            if uncertainty < 0.3:
                category = "low_uncertainty"
            elif uncertainty < 0.7:
                category = "medium_uncertainty"
            else:
                category = "high_uncertainty"
            
            uncertainty_analysis[category]["total"] += 1
            if is_correct:
                uncertainty_analysis[category]["correct"] += 1
        
        # Calculate accuracy for each uncertainty level
        for category in uncertainty_analysis:
            total = uncertainty_analysis[category]["total"]
            correct = uncertainty_analysis[category]["correct"]
            uncertainty_analysis[category]["accuracy"] = correct / total if total > 0 else 0
        
        return uncertainty_analysis
    
    def print_summary(self, results: List[Dict]):
        """Print evaluation summary"""
        metrics = self.calculate_metrics(results)
        uncertainty_analysis = self.analyze_by_uncertainty_level(results)
        
        print("\n" + "="*60)
        print("UNCERTAINTY VALIDATION EVALUATION SUMMARY")
        print("="*60)
        
        # Overall metrics
        print("\nOverall Performance:")
        for method, stats in metrics.items():
            print(f"\n{method.upper()}:")
            print(f"  Accuracy: {stats['accuracy']:.3f}")
            print(f"  Precision: {stats['precision']:.3f}")
            print(f"  Recall: {stats['recall']:.3f}")
            print(f"  F1-Score: {stats['f1_score']:.3f}")
            print(f"  Average Uncertainty: {stats['avg_uncertainty']:.3f}")
            print(f"  High Uncertainty Rate: {stats['high_uncertainty_rate']:.3f}")
            print(f"  Uncertainty-Correctness Correlation: {stats['uncertainty_correctness_correlation']:.3f}")
        
        # Uncertainty level analysis
        print("\nPerformance by Uncertainty Level:")
        for level, stats in uncertainty_analysis.items():
            print(f"  {level.replace('_', ' ').title()}: "
                  f"Accuracy {stats['accuracy']:.3f} ({stats['total']} samples)")
        
        # High uncertainty cases
        high_uncertainty_cases = [
            r for r in results 
            if r["methods"]["ensemble_uncertainty"]["high_uncertainty"]
        ]
        print(f"\nHigh Uncertainty Cases: {len(high_uncertainty_cases)}")
        
        if high_uncertainty_cases:
            print("Sample high uncertainty cases:")
            for i, case in enumerate(high_uncertainty_cases[:3]):  # Show first 3
                uncertainty = case["methods"]["ensemble_uncertainty"]["overall_uncertainty"]
                print(f"  {i+1}. ID {case['id']}: Uncertainty {uncertainty:.3f}")
                print(f"     Question: {case['question'][:100]}...")




def create_balanced_sample(examples: List[Dict], n: int) -> List[Dict]:
        """创建平衡的测试集，按错误类型和正确性分层采样"""
        
        # 首先按正确性分类
        correct_examples = [ex for ex in examples if ex.get('label', True) == True]
        incorrect_examples = [ex for ex in examples if ex.get('label', True) == False]
        
        print(f"Total examples: {len(examples)}")
        print(f"Correct examples: {len(correct_examples)}")
        print(f"Incorrect examples: {len(incorrect_examples)}")
        
        # 按错误类型进一步分类 (只对incorrect examples)
        error_type_groups = defaultdict(list)
        for ex in incorrect_examples:
            error_types = ex.get('error_types', [])
            if error_types:
                # 使用第一个错误类型作为主要分类
                primary_error = error_types[0].get('error_type', 'unknown')
                error_type_groups[primary_error].append(ex)
            else:
                error_type_groups['no_error_type'].append(ex)
        
        print(f"Error type distribution:")
        for error_type, examples_list in error_type_groups.items():
            print(f"  {error_type}: {len(examples_list)}")
        
        # 计算采样策略
        target_correct = min(n // 4, len(correct_examples))  # 25% correct examples
        target_incorrect = n - target_correct
        
        # 从correct examples中随机采样
        import random
        random.seed(42)  # 确保可重复性
        selected_correct = random.sample(correct_examples, target_correct) if len(correct_examples) >= target_correct else correct_examples
        
        # 从incorrect examples中按错误类型比例采样
        selected_incorrect = []
        if error_type_groups:
            samples_per_type = target_incorrect // len(error_type_groups)
            remaining_samples = target_incorrect % len(error_type_groups)
            
            for i, (error_type, examples_list) in enumerate(error_type_groups.items()):
                # 为前几个错误类型分配额外的样本
                current_samples = samples_per_type + (1 if i < remaining_samples else 0)
                current_samples = min(current_samples, len(examples_list))
                
                selected_from_type = random.sample(examples_list, current_samples) if len(examples_list) >= current_samples else examples_list
                selected_incorrect.extend(selected_from_type)
        
        # 合并结果
        balanced_sample = selected_correct + selected_incorrect[:target_incorrect]
        random.shuffle(balanced_sample)  # 打乱顺序
        
        print(f"\nBalanced sample created:")
        print(f"  Correct: {len(selected_correct)}")
        print(f"  Incorrect: {len(selected_incorrect)}")
        print(f"  Total: {len(balanced_sample)}")
        
        return balanced_sample



def main():
    """Main execution function"""

    current_dir = Path(__file__).parent  # scripts目录
    parent_dir = current_dir.parent
    # Paths
    DATA_PATH = parent_dir / "bug-data" / "NL2SQL-Bugs.json"
    DB_ROOT = parent_dir / "BIRD" / "dev_20240627" / "dev_databases"
    MAX_EXAMPLES = 200
    
    # Result path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULT_PATH = f"results/uncertainty_results_{MAX_EXAMPLES}_{timestamp}.jsonl"
    
    # Load data
    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, "r") as f:
        examples = json.load(f)
    
    print(f"Loaded {len(examples)} examples")
    
    # Create balanced sample
    examples = create_balanced_sample(examples, MAX_EXAMPLES)
    print(f"Using {len(examples)} balanced examples")
    # Initialize uncertainty validator
    validator = UncertaintyValidator(DB_ROOT)
    
    # Run evaluation
    print("Starting SQL Uncertainty Validation Evaluation...")
    results = validator.run_evaluation(examples, RESULT_PATH)
    
    # Print summary
    validator.print_summary(results)
    
    print(f"\nEvaluation complete! Results saved to {RESULT_PATH}")
    
    # Additional analysis
    metrics = validator.calculate_metrics(results)
    uncertainty_analysis = validator.analyze_by_uncertainty_level(results)
    
    # Save summary metrics
    summary_path = f"results/uncertainty_summary_{MAX_EXAMPLES}_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "uncertainty_analysis": uncertainty_analysis,
            "total_samples": len(results)
        }, f, indent=2)
    
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()





# def create_balanced_sample(examples: List[Dict], n: int = 200) -> List[Dict]:
    """创建平衡的测试集，按错误类型和正确性分层采样"""
    
    # 首先按正确性分类
    correct_examples = [ex for ex in examples if ex.get('label', True) == True]
    incorrect_examples = [ex for ex in examples if ex.get('label', True) == False]
    
    print(f"Total examples: {len(examples)}")
    print(f"Correct examples: {len(correct_examples)}")
    print(f"Incorrect examples: {len(incorrect_examples)}")
    
    # 按错误类型进一步分类 (只对incorrect examples)
    error_type_groups = defaultdict(list)
    for ex in incorrect_examples:
        error_types = ex.get('error_types', [])
        if error_types:
            # 使用第一个错误类型作为主要分类
            primary_error = error_types[0].get('error_type', 'unknown')
            error_type_groups[primary_error].append(ex)
        else:
            error_type_groups['no_error_type'].append(ex)
    
    print(f"Error type distribution:")
    for error_type, examples_list in error_type_groups.items():
        print(f"  {error_type}: {len(examples_list)}")
    
    # 计算采样策略
    target_correct = min(n // 4, len(correct_examples))  # 25% correct examples
    target_incorrect = n - target_correct
    
    # 从correct examples中随机采样
    import random
    random.seed(42)  # 确保可重复性
    selected_correct = random.sample(correct_examples, target_correct) if len(correct_examples) >= target_correct else correct_examples
    
    # 🎯 新增：按原始比例分配错误样本
    selected_incorrect = []
    if error_type_groups:
        total_incorrect_available = sum(len(examples_list) for examples_list in error_type_groups.values())
        
        print(f"\nProportional sampling for {target_incorrect} error samples:")
        
        for error_type, examples_list in error_type_groups.items():
            # 按原始比例计算应该分配的样本数
            original_proportion = len(examples_list) / total_incorrect_available
            proportional_samples = int(target_incorrect * original_proportion)
            
            # 确保不超过该类型的实际样本数
            current_samples = min(proportional_samples, len(examples_list))
            
            # 如果计算出的样本数为0，但该类型有样本，至少分配1个
            if current_samples == 0 and len(examples_list) > 0:
                current_samples = 1
            
            selected_from_type = random.sample(examples_list, current_samples) if len(examples_list) >= current_samples else examples_list
            selected_incorrect.extend(selected_from_type)
            
            print(f"  {error_type}: {len(examples_list)} available → {current_samples} selected (proportion: {original_proportion:.3f})")
        
        # 如果由于四舍五入导致样本不足，从最大的错误类型补充
        if len(selected_incorrect) < target_incorrect:
            shortage = target_incorrect - len(selected_incorrect)
            # 找到样本最多的错误类型
            largest_type = max(error_type_groups.keys(), key=lambda x: len(error_type_groups[x]))
            largest_examples = error_type_groups[largest_type]
            
            # 从已选样本中排除
            already_selected_ids = {ex.get('id') for ex in selected_incorrect}
            available_from_largest = [ex for ex in largest_examples if ex.get('id') not in already_selected_ids]
            
            additional_samples = min(shortage, len(available_from_largest))
            if additional_samples > 0:
                additional = random.sample(available_from_largest, additional_samples)
                selected_incorrect.extend(additional)
                print(f"  Additional {additional_samples} samples from {largest_type} to reach target")
        
        # 如果样本过多，随机移除多余的
        elif len(selected_incorrect) > target_incorrect:
            selected_incorrect = random.sample(selected_incorrect, target_incorrect)
            print(f"  Reduced to {target_incorrect} samples to match target")
    
    # 合并结果
    balanced_sample = selected_correct + selected_incorrect[:target_incorrect]
    random.shuffle(balanced_sample)  # 打乱顺序
    
    print(f"\nBalanced sample created:")
    print(f"  Correct: {len(selected_correct)}")
    print(f"  Incorrect: {len(selected_incorrect)}")
    print(f"  Total: {len(balanced_sample)}")
    
    return balanced_sample