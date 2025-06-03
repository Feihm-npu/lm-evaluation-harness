from transformers import AutoTokenizer

# 全局变量存储tokenizer（避免重复加载）
_tokenizer = None

def process_docs(dataset):
    """
    预处理数据集，可以用来重命名字段或添加新字段
    
    Args:
        dataset: HuggingFace数据集
    
    Returns:
        dataset: 处理后的数据集
    """
    paragraph_template = """
        # Title: {title}
        ## text: {text}
    """
    def _process_doc(doc):
        # 确保passages字段存在
        formatted_docs = [
            f"[document]{paragraph_template.format(title=ctx['title'], text=ctx['text']).strip()}[/document]" for ctx in doc["ctxs"]
        ]
        docs = "\n".join(formatted_docs)
        doc["passages"] = docs
        
        # 可以在这里添加其他预处理逻辑
        return doc
    return dataset.map(_process_doc)

def get_tokenizer():
    """获取tokenizer实例"""
    global _tokenizer
    if _tokenizer is None:
        # 这里需要替换为你实际使用的模型名称
        model_name = "Qwen/Qwen2.5-7B-Instruct"  # 替换为你的instruct模型
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer

def format_chat_prompt(doc):
    """
    使用tokenizer的apply_chat_template格式化prompt
    """
    question = doc["question"]
    passages = doc["passages"]
    
 
    # 构建对话格式
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on provided context. Give accurate, concise answers using only the information from the context passages."
        },
        {
            "role": "user", 
            "content": f"""Context:
                {passages}

            Question: {question}

            Please provide a direct answer based on the context above."""
        }
    ]
    
    # 使用tokenizer的chat template
    tokenizer = get_tokenizer()
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True  # 添加assistant的开始标记
    )
    
    return formatted_prompt




def custom_exact_match(predictions, references):
    """
    自定义精确匹配函数，可以处理多个参考答案
    """
    correct = 0
    total = len(predictions)
    
    for pred, ref in zip(predictions, references):
        pred = pred.strip().lower()
        # 如果reference是列表（多个可接受的答案）
        if isinstance(ref, list):
            if any(pred == r.strip().lower() for r in ref):
                correct += 1
        else:
            if pred == ref.strip().lower():
                correct += 1
    
    return correct / total if total > 0 else 0