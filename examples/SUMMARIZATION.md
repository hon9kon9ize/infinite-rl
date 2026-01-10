## Instruction

Provide a concise summary of the following text. Provide your reasoning and analysis of the key points first, but you must place the final summary at the very end of your response, enclosed in tags like this: <summary>[SUMMARY]</summary>. Do not include extra commentary inside the tags.

## Question

The recent surge in remote work has fundamentally altered urban economies. As office buildings in city centers remain under-occupied, local businesses like coffee shops and dry cleaners face declining revenue. Conversely, suburban residential areas are seeing a 'renaissance' of local commerce as workers spend their lunch hours and disposable income closer to home. Economists suggest this shift may be permanent, requiring cities to rethink zoning laws to convert vacant commercial spaces into residential units.

## Answer

Remote work is shifting economic activity from city centers to suburbs, potentially forcing permanent changes to urban zoning and land use.

## Reward Function

```python
def reward_fn(model_output, reference_answer):
    import re
    from sentence_transformers import SentenceTransformer, util
    
    # 1. Format Objective: Check for <summary> tags
    tag_pattern = r"<summary>(.*?)</summary>"
    match = re.search(tag_pattern, model_output, re.DOTALL)
    
    if not match:
        return (0.0, 0.0)
    
    format_score = 1.0
    predicted_summary = match.group(1).strip()
    
    # 2. Correctness (Semantic Similarity) Objective
    try:
        # Load a lightweight model for comparing meanings
        # In a real RL loop, you'd initialize this outside the function for speed
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode summaries into vectors
        emb_pred = model.encode(predicted_summary, convert_to_tensor=True)
        emb_ref = model.encode(reference_answer, convert_to_tensor=True)
        
        # Calculate Cosine Similarity (0 to 1 scale)
        correctness_score = util.cos_sim(emb_pred, emb_ref).item()
        
        # Optional: Apply a length penalty if the summary is too long
        if len(predicted_summary.split()) > len(reference_answer.split()) * 2:
            correctness_score *= 0.5
            
    except Exception:
        correctness_score = 0.0
        
    return (format_score, correctness_score)