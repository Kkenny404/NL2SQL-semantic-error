from share_utils import read_file
import os
import glob


# ========== LLM调用器（分离的模块）==========

class LLMErrorInjector:
    """LLM错误注入器 - 独立的模块"""
    
    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: 你的LLM客户端，比如OpenAI客户端
        """
        self.llm_client = llm_client
    
    def inject_error_with_llm(self, prompt: str) -> str:
        """使用LLM注入错误"""
        if self.llm_client is None:
            raise ValueError("LLM client not provided")
        
        # 这里替换为实际的LLM调用
        # response = self.llm_client.chat.completions.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.1
        # )
        # return response.choices[0].message.content
        
        # 临时返回原始SQL作为占位符
        return "-- LLM generated error SQL would be here"
    
    def process_prompts_in_directory(self, 
                                   category_dir: str,
                                   output_sql_dir: str = None):
        """处理目录中的所有prompt文件"""
        if output_sql_dir is None:
            output_sql_dir = os.path.join(category_dir, "error_sqls")
        
        os.makedirs(output_sql_dir, exist_ok=True)
        
        # 查找所有prompt文件
        prompt_files = glob.glob(os.path.join(category_dir, "*_prompt.txt"))
        
        for prompt_file in prompt_files:
            try:
                prompt = read_file(prompt_file)
                error_sql = self.inject_error_with_llm(prompt)
                
                # 生成对应的SQL文件名
                base_name = os.path.basename(prompt_file).replace("_prompt.txt", ".sql")
                sql_path = os.path.join(output_sql_dir, base_name)
                
                with open(sql_path, "w", encoding="utf-8") as f:
                    f.write(error_sql.strip())
                
                print(f"✅ Generated error SQL: {sql_path}")
                
            except Exception as e:
                print(f"⚠️ Error processing {prompt_file}: {e}")
