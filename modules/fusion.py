def fuse_results(text_logit, image_logit):
    """
    Fusion rule (AND logic):
    - The result is 1 (Not Gambling) only if both inputs are 1.
    - In all other cases, the result is 0 (Gambling).

    Args:
        text_logit (int): Logit from text classification (0 or 1).
        image_logit (int): Logit from image classification (0 or 1).

    Returns:
        int: Final fused logit (0 or 1).
    
    Raises:
        ValueError: If the input is not 0 or 1.
    """
    if text_logit not in [0, 1] or image_logit not in [0, 1]:
        raise ValueError("Input logit must be 0 or 1.")
        
    # AND logic: the result is 1 only if both inputs are 1.
    final_logit = text_logit and image_logit
    
    return final_logit
