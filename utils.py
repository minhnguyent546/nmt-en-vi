
def count_parameters(model):
    return sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    
