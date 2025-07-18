# semantic_detector.py
import re
from typing import Optional, Dict, Any

def build_rubber_duck_prompt(question: str, schema: str, sql: str) -> str:
    """橡皮鸭调试提示词 - 逐步分析SQL语义正确性"""
    return f"""You are an experienced database administrator. Given the following natural language question, SQL query, and database schema, determine if the SQL **semantically matches** the question.

Answer "Yes" — if semantically correct  
Answer "No" — if semantically incorrect  

Important: Answer "Yes" or "No" first on a separate line.  
Then, **explain your reasoning** in 2–3 concise sentences.
---

Question: {question}

Database Schema: {schema}

SQL Query: {sql}

Analyze step-by-step:

Step 1: Analyze FROM and JOIN clauses - Are correct tables joined with appropriate conditions?
Step 2: Analyze SELECT clause - Are correct columns selected matching the question?  
Step 3: Analyze WHERE clause - Do filter conditions align with question's intent?
Step 4: Analyze ORDER BY/GROUP BY - Is sorting/grouping logic correct?
Step 5: Overall semantic check - Does the query return what the question asks for?

Based on your analysis, is the SQL semantically correct? Answering in yes or no, do not include any other text after your ANSWER line."""


def build_reflexion_prompt(question: str, schema: str, sql: str, previous_analysis: str) -> str:
    """反思机制提示词 - 基于首次分析进行更深入检查"""
    return f"""You are a database expert doing a second review. A previous analysis was done, but you need to double-check for any missed semantic errors.
Answer "Yes" — if semantically correct  
Answer "No" — if semantically incorrect  

Important: Answer "Yes" or "No" first on a separate line.  
Then, **explain your reasoning** in 2–3 concise sentences.
---

Question: {question}

Database Schema: {schema}

SQL Query: {sql}

Previous Analysis Result:
{previous_analysis}


Now carefully re-examine in your mind:
1. Column name mapping - Are column names correctly corresponding to question's intent?
2. Aggregation logic - Are aggregation functions used appropriately?  
3. Join conditions - Are table relationships semantically correct?
4. Filter logic - Do WHERE conditions capture the question's requirements?

INSTRUCTION: Respond with ONLY ONE WORD: "Yes" or "No"
DO NOT provide explanations or analysis.
ONLY respond: Yes or No."""

def extract_confidence_from_response(response: str) -> float:
    """从响应中提取置信度指标"""
    response_lower = response.lower()
    
    # 检查不确定性关键词
    uncertainty_keywords = ['might', 'could', 'possibly', 'unclear', 'ambiguous', 'not sure']
    uncertainty_count = sum(1 for keyword in uncertainty_keywords if keyword in response_lower)
    
    # 检查确定性关键词
    certainty_keywords = ['clearly', 'definitely', 'obviously', 'correct', 'appropriate']
    certainty_count = sum(1 for keyword in certainty_keywords if keyword in response_lower)
    
    # 基于关键词计算基础置信度
    base_confidence = max(0.3, 0.7 + (certainty_count - uncertainty_count) * 0.1)
    
    return min(0.95, base_confidence)

def parse_answer_enhanced(response: str) -> Optional[bool]:
    """解析CORRECT/INCORRECT格式"""
    if not response:
        return None
        
    text = response.strip().lower()
    
    # 查找我们的关键词
    if "correct" in text and "incorrect" not in text:
        return True
    elif "incorrect" in text:
        return False
    
    # Fallback到yes/no
    if "yes" in text and "no" not in text:
        return True
    elif "no" in text and "yes" not in text:
        return False
    
    print(f"[WARN] Could not parse: '{response[:100]}...'")
    return None


def is_low_confidence(response: str) -> bool:
    """判断响应是否显示低置信度"""
    confidence = extract_confidence_from_response(response)
    return confidence < 0.6

class SemanticErrorDetector:
    """语义错误检测器 - 橡皮鸭 + 反思机制"""
    
    def __init__(self, llm_query_func):
        self.llm_query_func = llm_query_func
    
    def detect(self, question: str, schema: str, sql: str) -> Dict[str, Any]:
        """主检测方法 - 自适应使用橡皮鸭和反思"""
        
        # 第一轮：橡皮鸭调试
        duck_prompt = build_rubber_duck_prompt(question, schema, sql)
        duck_response = self.llm_query_func(duck_prompt)
        duck_prediction = parse_answer_enhanced(duck_response)
        
        result = {
            'question': question,
            'sql': sql,
            'method_used': 'rubber_duck',
            'duck_response': duck_response,
            'duck_prediction': duck_prediction,
            'final_prediction': duck_prediction,
            'confidence': extract_confidence_from_response(duck_response)
        }
        
        # 如果置信度低或解析失败，启用反思机制
        if duck_prediction is None or is_low_confidence(duck_response):
            try:
                reflexion_prompt = build_reflexion_prompt(question, schema, sql, duck_response)
                reflexion_response = self.llm_query_func(reflexion_prompt)
                reflexion_prediction = parse_answer_enhanced(reflexion_response)
                
                result.update({
                    'method_used': 'rubber_duck_reflexion',
                    'reflexion_response': reflexion_response,
                    'reflexion_prediction': reflexion_prediction,
                    'final_prediction': reflexion_prediction if reflexion_prediction is not None else duck_prediction,
                    'confidence': max(result['confidence'], extract_confidence_from_response(reflexion_response))
                })
                
            except Exception as e:
                result['reflexion_error'] = str(e)
        
        return result