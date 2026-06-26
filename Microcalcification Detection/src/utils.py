def decode_to_bounding_box(state_str, roi_origin, roi_size=64):
    """
    Final Detection Result Layer.
    Converts quantum bitstrings back to (x, y) coordinates.
    """
    # Convert binary state to integer address
    pixel_idx = int(state_str.replace(" ", ""), 2)
    
    # Map 1D index to 2D ROI coordinates
    y_roi, x_roi = divmod(pixel_idx, roi_size)
    
    # Map ROI coordinates back to Full Image coordinates
    global_x = roi_origin[0] + x_roi
    global_y = roi_origin[1] + y_roi
    
    # Create a 5x5 bounding box around the detected center
    bbox = [global_x - 2, global_y - 2, global_x + 2, global_y + 2]
    
    return (global_x, global_y), bbox