def valid_argmax2D(tensor, grid_view):
    valid_flat = ~torch.flatten(torch.tensor(grid_view))
    output_flat = torch.flatten(tensor)

    valid_indices = torch.where(valid_flat)[0]
    valid_max_index = torch.argmax(output_flat[valid_indices])
    valid_indices[valid_max_index]

    if grid_view.shape[0] != grid_view.shape[1]:
        raise NotImplementedError()
    
    max_row = valid_indices[valid_max_index] // grid_view.shape[0]
    max_col = valid_indices[valid_max_index] % grid_view.shape[0]
    return max_row, max_col
