## Instruction

Provide a concise summary of the following text. Place the final summary result at the very end of your response, enclosed in tags like this: <summary>[SUMMARY]</summary>.

## Prompt

The recent surge in remote work has fundamentally altered urban economies. As office buildings in city centers remain under-occupied, local businesses like coffee shops and dry cleaners face declining revenue. Conversely, suburban residential areas are seeing a 'renaissance' of local commerce as workers spend their lunch hours and disposable income closer to home. Economists suggest this shift may be permanent, requiring cities to rethink zoning laws to convert vacant commercial spaces into residential units.

## Answer

<summary>Remote work is shifting economic activity from city centers to suburbs, potentially forcing permanent changes to urban zoning and land use.</summary>

## Response

The recent shift towards remote work has significantly reshaped urban and suburban economies. Previously, city centers thrived as workers commuted in daily, supporting local businesses. However, with workers now spending more time at home, suburban areas have experienced economic revitalization. Coffee shops, restaurants, and retailers in suburban areas are now benefiting from the presence of remote workers who previously worked in downtown offices.

This structural change presents both challenges and opportunities. Cities must adapt by reimagining their commercial spaces, converting underutilized office buildings into residential units or new businesses. Urban planners and policymakers are beginning to recognize that zoning laws may need significant revision to accommodate this permanent shift in work patterns.

<summary>Remote work is shifting economic activity from city centers to suburbs, potentially forcing permanent changes to urban zoning and land use.</summary>

## Reward Function

```python
def reward_fn(model_output, expected_output):
    import re
    from sentence_transformers import SentenceTransformer, util
    
    # 1. Format Objective (Part A): Extract JSON code block
    code_pattern = r"```(?:json)?\\s*(.*?)```"
    match = re.search(code_pattern, model_output, re.DOTALL)
    
    if not match:
        return (0.0, 0.0)
    
    json_block = match.group(1).strip()
    code_format_score = 0.5  # JSON block found
    
    # 2. Format Objective (Part B): Validate JSON format
    try:
        actual_json = json.loads(json_block)
        json_format_score = 0.5  # Valid JSON
        actual_summary = actual_json.get('summary', '')
        
        # 3. Correctness Objective: Semantic similarity
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            expected_json = json.loads(reference_answer.strip())
            expected_summary = expected_json.get('summary', '')
            
            emb_actual = model.encode(actual_summary, convert_to_tensor=True)
            emb_expected = model.encode(expected_summary, convert_to_tensor=True)
            
            correctness_score = util.cos_sim(emb_actual, emb_expected).item()
        except Exception:
            correctness_score = 0.0
    except json.JSONDecodeError:
        json_format_score = 0.0
        correctness_score = 0.0
    
    format_score = code_format_score + json_format_score
    return (format_score, correctness_score)